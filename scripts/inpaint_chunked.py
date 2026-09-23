"""
inpaint_chunked.py — chunked wrapper around m2svid's inpaint+refine stage.

The M2SVid model has a fixed temporal window (config `num_samples`, 25 frames):
`M2SVidModel.generate` raises NotImplementedError for longer clips. Upstream's
`inpaint_and_refine.py` therefore only supports <= 25-frame videos.

This driver mirrors that preprocessing but loads the model ONCE and runs it over
consecutive <= chunk-frame windows, concatenating the generated frames. Short
tail windows are padded (by repeating the last frame) to the full window size so
the model always sees a training-length clip, then trimmed back.

Run inside the `sgm` conda env, with cwd = m2svid repo root and PYTHONPATH set to
`.:third_party/Hi3D-Official:third_party/pytorch-msssim` (vid2stereo.py does this).
"""

import argparse
import os

import ffmpeg
import numpy as np
import torch
import torchvision.io
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torchvision import transforms

from sgm.util import instantiate_from_config
from m2svid.utils.video_utils import get_video_fps
from m2svid.data.utils import get_video_frames, apply_closing, apply_dilation
from m2svid.utils.anaglyph import make_anaglyph_video


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model_config", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--video_path", type=str, required=True)
    p.add_argument("--reprojected_path", type=str, required=True)
    p.add_argument("--reprojected_mask_path", type=str, required=True)
    p.add_argument("--output_folder", type=str, required=True)
    p.add_argument("--reprojected_closing_holes_kernel", type=int, default=11)
    p.add_argument("--mask_antialias", type=int, default=0)
    p.add_argument("--chunk_frames", type=int, default=25)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def save_video(video: torch.Tensor, fps: float, path: str) -> None:
    """video: [b, c, t, h, w] in [-1, 1]."""
    frames = video.cpu().numpy().transpose(0, 2, 3, 4, 1)
    frames = np.concatenate(frames)
    frames = (((frames + 1) / 2).clip(0, 1) * 255).astype(np.uint8)
    torchvision.io.write_video(path, frames, fps=int(fps), options={"crf": "17"})


def main() -> None:
    args = parse_args()

    seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(2), "big")
    seed_everything(seed)

    config = OmegaConf.load(args.model_config)
    model = instantiate_from_config(config.model).cpu()
    model.init_from_ckpt(args.ckpt)
    model = model.cuda().half().eval()

    # --- preprocess full-length videos (mirrors inpaint_and_refine.py) --------
    input_video = get_video_frames(args.video_path)
    reprojected = get_video_frames(args.reprojected_path)
    reprojected_mask = get_video_frames(args.reprojected_mask_path, video_is_grayscale=True)
    fps = get_video_fps(args.video_path, ffmpeg.probe(args.video_path))

    reprojected_mask = apply_closing(reprojected_mask, args.reprojected_closing_holes_kernel)
    reprojected[reprojected_mask.repeat(1, 3, 1, 1) > 0.5] = 0
    reprojected_mask = apply_dilation(reprojected_mask, 3)
    reprojected_mask = reprojected_mask.repeat(1, 3, 1, 1)

    input_video = input_video.permute(1, 0, 2, 3).float() * 2 - 1   # [c,t,h,w]
    reprojected = reprojected.permute(1, 0, 2, 3).float() * 2 - 1   # [c,t,h,w]
    reprojected_mask = reprojected_mask.permute(1, 0, 2, 3).float() * 2 - 1  # [c,t,h,w]

    _, _, h, w = reprojected_mask.shape
    downsampled_resolution = [int(h / 8), int(w / 8)]
    reprojected_mask = reprojected_mask.permute(1, 0, 2, 3).float()  # [t,c,h,w]
    reprojected_mask = transforms.Resize(
        downsampled_resolution, antialias=bool(args.mask_antialias)
    )(reprojected_mask)
    reprojected_mask = reprojected_mask[:, [0]]
    reprojected_mask = reprojected_mask.permute(1, 0, 2, 3).float()  # [c,t,h,w]

    total = input_video.shape[1]
    window = min(max(1, args.chunk_frames), total)

    generated_parts = []
    for start in range(0, total, window):
        end = min(start + window, total)
        count = end - start

        iv = input_video[:, start:end]
        rv = reprojected[:, start:end]
        rm = reprojected_mask[:, start:end]

        # Pad short tail windows up to the model window by repeating the last
        # frame, so the model always sees a training-length clip.
        pad = window - count
        if pad > 0:
            iv = torch.cat([iv, iv[:, -1:].repeat(1, pad, 1, 1)], dim=1)
            rv = torch.cat([rv, rv[:, -1:].repeat(1, pad, 1, 1)], dim=1)
            rm = torch.cat([rm, rm[:, -1:].repeat(1, pad, 1, 1)], dim=1)

        batch = {
            "video": iv[None].cuda(),
            "video_2nd_view": iv[None].cuda(),
            "reprojected_video": rv[None].cuda(),
            "reprojected_mask": rm[None].cuda(),
            "fps_id": torch.tensor([fps]).cuda(),
            "caption": [""],
            "motion_bucket_id": torch.tensor([127]).cuda(),
        }

        n_win = (total + window - 1) // window
        print(f"[inpaint_chunked] window {start // window + 1}/{n_win} "
              f"(frames {start}-{end - 1})", flush=True)
        with torch.inference_mode():
            g = model.generate(batch)["generated-video"][0].cpu()  # [c,t,h,w]

        if pad > 0:
            g = g[:, :count]
        generated_parts.append(g)

    generated_video = torch.cat(generated_parts, dim=1)  # [c,total,h,w]

    sbs_video = torch.cat([input_video, generated_video], dim=-1)
    anaglyph = make_anaglyph_video(input_video, generated_video, unnormalized_videos=True)

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    os.makedirs(args.output_folder, exist_ok=True)
    save_video(generated_video[None], fps, os.path.join(args.output_folder, f"{video_name}_generated.mp4"))
    save_video(sbs_video[None], fps, os.path.join(args.output_folder, f"{video_name}_sbs.mp4"))
    save_video(anaglyph[None], fps, os.path.join(args.output_folder, f"{video_name}_anaglyph.mp4"))
    print(f"[inpaint_chunked] wrote outputs to {args.output_folder}", flush=True)


if __name__ == "__main__":
    main()

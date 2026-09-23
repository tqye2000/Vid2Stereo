#!/usr/bin/env python3
"""
Vid2Stereo — thin orchestrator around google-research/m2svid.

Turns a single monocular video into a stereo pair by chaining the three
official m2svid stages, each in its own conda environment:

    1. DepthCrafter        (env: depthcrafter)  -> relative depth (.npz)
    2. warping.py          (env: sgm)           -> reprojected right view + mask
    3. inpaint_and_refine  (env: sgm)           -> refined right view

Then it post-processes the side-by-side result into aligned left / right MP4s.

Run this from inside WSL2 (Ubuntu). See setup.sh for one-time environment setup.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NoReturn


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def log(msg: str) -> None:
    print(f"[vid2stereo] {msg}", flush=True)


def die(msg: str, code: int = 1) -> NoReturn:
    print(f"[vid2stereo] ERROR: {msg}", file=sys.stderr, flush=True)
    sys.exit(code)


def require(tool: str) -> None:
    if shutil.which(tool) is None:
        die(f"required tool '{tool}' not found on PATH")


def ffprobe_dims_fps(video: Path) -> tuple[int, int, float]:
    """Return (width, height, fps) for the first video stream."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate",
            "-of", "json", str(video),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    stream = json.loads(out)["streams"][0]
    num, _, den = stream["avg_frame_rate"].partition("/")
    fps = float(num) / float(den) if den and float(den) != 0 else float(num)
    return int(stream["width"]), int(stream["height"]), fps


def run_stage(name: str, cmd: list[str], cwd: Path, env: dict[str, str]) -> None:
    log(f"stage: {name}")
    log("  $ " + " ".join(cmd))
    result = subprocess.run(cmd, cwd=str(cwd), env=env)
    if result.returncode != 0:
        die(f"stage '{name}' failed with exit code {result.returncode}")


def conda_run(env_name: str, py_args: list[str]) -> list[str]:
    """Wrap a python invocation so it runs inside the given conda env."""
    return [
        "conda", "run", "-n", env_name, "--no-capture-output",
        "python", *py_args,
    ]


# --------------------------------------------------------------------------- #
# Core pipeline
# --------------------------------------------------------------------------- #
def prepare_input(src: Path, work: Path, auto_crop: bool) -> Path:
    """Validate resolution (multiple of 64). Optionally center-crop to fit."""
    width, height, _ = ffprobe_dims_fps(src)
    cw, ch = width - (width % 64), height - (height % 64)

    if width % 64 == 0 and height % 64 == 0:
        return src

    if not auto_crop:
        die(
            f"input is {width}x{height}; width & height must be divisible by 64. "
            f"Nearest valid crop is {cw}x{ch}. Re-run without --no-auto-crop to "
            "auto-crop, or resize the video yourself."
        )

    if cw <= 0 or ch <= 0:
        die(f"input {width}x{height} is too small to crop to a multiple of 64")

    cropped = work / f"{src.stem}_crop64.mp4"
    x, y = (width - cw) // 2, (height - ch) // 2
    log(f"auto-cropping {width}x{height} -> {cw}x{ch} (centered)")
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", str(src),
            "-vf", f"crop={cw}:{ch}:{x}:{y}",
            "-c:v", "libx264", "-crf", "17", "-preset", "fast",
            "-an", str(cropped),
        ],
        check=True, capture_output=True, text=True,
    )
    return cropped


def split_sbs(sbs: Path, out_left: Path, out_right: Path, fps: float) -> None:
    """Split a side-by-side [left|right] video into two aligned MP4s."""
    for path, x_expr in ((out_left, "0"), (out_right, "iw/2")):
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(sbs),
                "-vf", f"crop=iw/2:ih:{x_expr}:0",
                "-c:v", "libx264", "-crf", "17", "-preset", "slow",
                "-r", str(fps), "-an", str(path),
            ],
            check=True, capture_output=True, text=True,
        )


def has_audio(video: Path) -> bool:
    """Return True if the video has at least one audio stream."""
    out = subprocess.run(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0", str(video),
        ],
        check=True, capture_output=True, text=True,
    ).stdout
    return bool(out.strip())


def mux_audio(video: Path, source: Path) -> None:
    """Copy the audio track from `source` into `video` (in place).

    The ML pipeline produces silent video, so we re-attach the original
    soundtrack. Video is stream-copied; audio is re-encoded to AAC for
    broad MP4 compatibility. `-shortest` guards against tiny duration
    mismatches between the processed video and the original audio.
    """
    tmp = video.with_name(video.stem + "_muxed" + video.suffix)
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", str(video),
            "-i", str(source),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", str(tmp),
        ],
        check=True, capture_output=True, text=True,
    )
    tmp.replace(video)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a monocular video into a stereo pair via m2svid.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", "-i", required=True, type=Path,
                        help="Path to the input (monocular) video.")
    parser.add_argument("--output-dir", "-o", default=Path("out"), type=Path,
                        help="Directory for final outputs.")
    parser.add_argument("--m2svid-root", default=Path("m2svid"), type=Path,
                        help="Path to the cloned m2svid repository.")
    parser.add_argument("--variant", choices=["full-attn", "no-full-attn"],
                        default="full-attn",
                        help="Which released model weights to use.")
    parser.add_argument("--disparity-perc", type=float, default=0.05,
                        help="Max disparity as a fraction of frame width.")
    parser.add_argument("--chunk-frames", type=int, default=25,
                        help="Temporal window for the inpaint stage. Videos "
                             "longer than this are processed in chunks and "
                             "concatenated (model window is 25).")
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU index (sets CUDA_VISIBLE_DEVICES).")
    parser.add_argument("--depthcrafter-env", default="depthcrafter",
                        help="conda env name for DepthCrafter.")
    parser.add_argument("--sgm-env", default="sgm",
                        help="conda env name for m2svid (sgm).")
    parser.add_argument("--max-res", type=int, default=1024,
                        help="DepthCrafter --max_res.")
    parser.add_argument("--num-inference-steps", type=int, default=25,
                        help="DepthCrafter denoising steps.")
    parser.add_argument("--no-auto-crop", action="store_true",
                        help="Error instead of auto-cropping to a multiple of 64.")
    parser.add_argument("--keep-intermediates", action="store_true",
                        help="Keep depth / reprojected / mask files.")
    parser.add_argument("--ca-bundle", type=Path,
                        default=Path.home() / "corp-ca-bundle.pem",
                        help="CA bundle for HTTPS (HuggingFace) downloads behind a "
                             "TLS-intercepting proxy. Ignored if the file is absent.")
    args = parser.parse_args()

    # --- resolve paths & tools ------------------------------------------------
    for tool in ("conda", "ffmpeg", "ffprobe"):
        require(tool)

    root = args.m2svid_root.resolve()
    if not (root / "inpaint_and_refine.py").is_file():
        die(f"m2svid repo not found at {root} (run setup.sh first?)")

    if args.variant == "full-attn":
        model_config, ckpt = "configs/m2svid.yaml", "ckpts/m2svid_weights.pt"
    else:
        model_config = "configs/m2svid_no_fullatten.yaml"
        ckpt = "ckpts/m2svid_no_full_atten_weights.pt"
    if not (root / ckpt).is_file():
        die(f"weights not found: {root / ckpt} (run setup.sh to download)")

    src = args.input.resolve()
    if not src.is_file():
        die(f"input video not found: {src}")

    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work"
    work.mkdir(exist_ok=True)

    # --- environment for subprocesses ----------------------------------------
    pythonpath = os.pathsep.join([
        ".", "./third_party/Hi3D-Official", "./third_party/pytorch-msssim",
        os.environ.get("PYTHONPATH", ""),
    ])
    # CA bundle for HF/model downloads behind a TLS-intercepting proxy.
    # The inference conda envs are Python 3.10 (not strict), so pointing
    # requests/curl at the corporate bundle is safe here.
    ca_env: dict[str, str] = {}
    if args.ca_bundle and args.ca_bundle.is_file():
        ca = str(args.ca_bundle)
        ca_env = {
            "REQUESTS_CA_BUNDLE": ca,
            "CURL_CA_BUNDLE": ca,
            "SSL_CERT_FILE": ca,
        }
        log(f"using CA bundle for downloads: {ca}")

    env = {
        **os.environ,
        "PYTHONPATH": pythonpath,
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        **ca_env,
    }

    # DepthCrafter needs its own PYTHONPATH (its repo root).
    dc_env = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([
            "./third_party/DepthCrafter", os.environ.get("PYTHONPATH", ""),
        ]),
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        **ca_env,
    }

    stem = src.stem
    depth_npz = work / f"{stem}.npz"
    reproj = work / f"{stem}_reprojected.mp4"
    reproj_mask = work / f"{stem}_reprojected_mask.mp4"

    # --- stage 0: validate / crop input --------------------------------------
    prepared = prepare_input(src, work, auto_crop=not args.no_auto_crop)
    _, _, fps = ffprobe_dims_fps(prepared)

    # --- stage A: DepthCrafter -> depth .npz ---------------------------------
    run_stage(
        "DepthCrafter (depth)",
        conda_run(args.depthcrafter_env, [
            "third_party/DepthCrafter/run.py",
            "--video-path", str(prepared),
            "--save_folder", str(work),
            "--save_npz", "True",
            "--num_inference_steps", str(args.num_inference_steps),
            "--max_res", str(args.max_res),
        ]),
        cwd=root, env=dc_env,
    )
    # DepthCrafter names the npz after the input stem.
    produced_npz = work / f"{prepared.stem}.npz"
    if produced_npz != depth_npz and produced_npz.is_file():
        produced_npz.replace(depth_npz)
    if not depth_npz.is_file():
        die(f"expected depth npz not found: {depth_npz}")

    # --- stage B: depth-based warping ----------------------------------------
    # warping.py uses `ffmpeg -n` and refuses to overwrite; clear any stale
    # outputs left behind by a previous (possibly failed) run first.
    for stale in (reproj, reproj_mask):
        stale.unlink(missing_ok=True)
    run_stage(
        "warping (reproject + mask)",
        conda_run(args.sgm_env, [
            "warping.py",
            "--video_path", str(prepared),
            "--depth_path", str(depth_npz),
            "--output_path_reprojected", str(reproj),
            "--output_path_mask", str(reproj_mask),
            "--disparity_perc", str(args.disparity_perc),
        ]),
        cwd=root, env=env,
    )

    # --- stage C: inpaint + refine (SVD) -------------------------------------
    # The M2SVid model has a fixed temporal window (config `num_samples`, 25).
    # scripts/inpaint_chunked.py loads the model once and runs it over
    # consecutive <= chunk-frame windows, so arbitrary-length videos work.
    inpaint_driver = str((Path(__file__).resolve().parent / "scripts" / "inpaint_chunked.py"))
    run_stage(
        "M2SVid (inpaint + refine)",
        conda_run(args.sgm_env, [
            inpaint_driver,
            "--mask_antialias", "0",
            "--model_config", model_config,
            "--ckpt", ckpt,
            "--video_path", str(prepared),
            "--reprojected_path", str(reproj),
            "--reprojected_mask_path", str(reproj_mask),
            "--output_folder", str(work),
            "--chunk_frames", str(max(1, args.chunk_frames)),
        ]),
        cwd=root, env=env,
    )

    sbs = work / f"{prepared.stem}_sbs.mp4"
    anaglyph = work / f"{prepared.stem}_anaglyph.mp4"
    generated = work / f"{prepared.stem}_generated.mp4"
    if not sbs.is_file():
        die(f"expected SBS output not found: {sbs}")

    # --- post-process: final outputs -----------------------------------------
    final_sbs = out_dir / f"{stem}_sbs.mp4"
    final_left = out_dir / f"{stem}_left.mp4"
    final_right = out_dir / f"{stem}_right.mp4"
    shutil.copy2(sbs, final_sbs)
    split_sbs(sbs, final_left, final_right, fps)
    if anaglyph.is_file():
        shutil.copy2(anaglyph, out_dir / f"{stem}_anaglyph.mp4")
    if generated.is_file() and not final_right.is_file():
        shutil.copy2(generated, final_right)

    # Re-attach the original soundtrack (the ML stages produce silent video).
    if has_audio(src):
        log("muxing original audio into final outputs")
        outputs_with_audio = [final_sbs, final_left, final_right]
        anaglyph_out = out_dir / f"{stem}_anaglyph.mp4"
        if anaglyph_out.is_file():
            outputs_with_audio.append(anaglyph_out)
        for out in outputs_with_audio:
            if out.is_file():
                try:
                    mux_audio(out, src)
                except subprocess.CalledProcessError as exc:
                    log(f"  warning: could not add audio to {out.name}: "
                        f"{exc.stderr.strip() if exc.stderr else exc}")
    else:
        log("source has no audio stream; outputs will be silent")

    if not args.keep_intermediates:
        shutil.rmtree(work, ignore_errors=True)

    log("done. outputs:")
    for p in (final_sbs, final_left, final_right):
        log(f"  {p}")


if __name__ == "__main__":
    main()

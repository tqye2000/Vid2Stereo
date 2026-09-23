# Vid2Stereo

Convert a single **monocular** video into a **stereo pair** using
[google-research/m2svid](https://github.com/google-research/m2svid)
(*M2SVid: End-to-End Inpainting and Refinement for Monocular-to-Stereo Video
Conversion*, 3DV 2026).

This repo is a **thin orchestrator** around the official m2svid code — it chains
the three pipeline stages into one command and post-processes the result into
side-by-side (SBS) plus separate left / right videos. It is **inference-only**
(uses pretrained weights, no training).

```
input.mp4
   ↓  DepthCrafter          → relative depth (.npz)         [env: depthcrafter]
   ↓  depth warping         → reprojected right view + mask  [env: sgm]
   ↓  M2SVid (SVD refine)   → high-quality right view        [env: sgm]
   ↓
*_sbs.mp4 · *_left.mp4 · *_right.mp4 · *_anaglyph.mp4
```

Each stage runs in its **own conda environment** (they use incompatible
dependency stacks). The only interface between DepthCrafter and the rest is the
depth `.npz`, which is format-stable.

## Requirements

- **WSL2 (Ubuntu)** or native Linux. The pipeline targets **CUDA 11.8 / torch 2.0.1**.
  (Development machine: Windows host + WSL2 Ubuntu.)
- **NVIDIA GPU**, ~12 GB+ recommended (tested on RTX 6000 Ada, 49 GB).
- `git`, `curl` — plus `conda`/`ffmpeg`, which `setup.sh` installs into a
  user-space Miniforge if not present.
- Network access to `pypi.org`, `download.pytorch.org`, `huggingface.co`,
  `storage.googleapis.com`, and `github.com`.

### Corporate network / TLS interception (Zscaler etc.)

If you are behind a TLS-intercepting proxy, HTTPS certificate validation will
fail out of the box. This repo includes tooling for it:

1. On Windows, export the trust store: `scripts/export-windows-ca.ps1`
   → produces `corp-ca-bundle.pem`.
2. Copy it to your WSL home as `~/corp-ca-bundle.pem` (strip CRLF).
3. `setup.sh` and `vid2stereo.py` pick it up automatically:
   - `setup.sh` uses it for `curl` / `git` / `pip` / `conda`.
   - `vid2stereo.py` has `--ca-bundle` (default `~/corp-ca-bundle.pem`) and
     injects it into the inference subprocesses so runtime HuggingFace / torch-hub
     downloads succeed.
4. `conda` uses the **prefix.dev** conda-forge mirror (the Anaconda default
   channels return HTTP 403 through the proxy). See `~/.condarc` created by setup.

Diagnostics: `scripts/net-check.sh`, `scripts/ca-diagnose.sh`.

## One-time setup

From the repo root inside WSL2:

```bash
bash setup.sh 2>&1 | tee setup.log
```

This is idempotent / resumable and will:

- install a user-space **Miniforge** (`~/miniforge3`) if conda is missing,
- clone `m2svid` + submodules and `Hi3D-Official`,
- pin the **DepthCrafter** submodule to tag **v1.0.1** (the torch-2.0.1-compatible
  release) and create the `depthcrafter` env from its `requirements.txt`,
- create the `sgm` env (torch 2.0.1+cu118, xformers, open-clip, sgm stack),
- install `ffmpeg` **into both** envs (needed for previews and `ffprobe`),
- download all weights into `m2svid/ckpts/`:
  - `m2svid_weights.pt` (~4.6 GB) from Google Cloud Storage,
  - `open_clip_pytorch_model.bin` (~3.8 GB, laion2b ViT-H/14) from HuggingFace.

No manual weight / `ckpts.zip` step is required — it is fully automated. When it
finishes you'll see `==> Setup complete.`

Optional sanity check of the `sgm` env:

```bash
cd m2svid
conda activate sgm
PYTHONPATH=./:./third_party/Hi3D-Official/:./third_party/pytorch-msssim/ \
  python ../scripts/sanity.py     # prints torch / cuda / gpus + "SANITY OK"
```

## Usage

The wrapper calls `python`, so you must **activate the base env first** (so
`python` is on PATH), then run:

```bash
source ~/miniforge3/etc/profile.d/conda.sh
conda activate base
python vid2stereo.py --input path/to/video.mp4 --output-dir out/
```

One-liner (e.g. from PowerShell driving WSL):

```bash
wsl -d Ubuntu -- bash -lc "cd /mnt/d/TQYE/Experiments/Vid2Stereo; \
  source ~/miniforge3/etc/profile.d/conda.sh; conda activate base; \
  python vid2stereo.py -i m2svid/demo/input.mp4 -o out"
```

Outputs in `out/`:

| File            | Description                              |
| --------------- | ---------------------------------------- |
| `*_sbs.mp4`     | Side-by-side [left \| right] (2× width)  |
| `*_left.mp4`    | Left eye (original view, aligned)        |
| `*_right.mp4`   | Right eye (generated view)               |
| `*_anaglyph.mp4`| Red-cyan preview                         |

Intermediates (depth `.npz`, reprojected view, mask, previews) land in
`out/_work/`.

### Common options

| Flag                    | Default                 | Notes                                                   |
| ----------------------- | ----------------------- | ------------------------------------------------------- |
| `--input` / `-i`        | *(required)*            | Input video path.                                       |
| `--output-dir` / `-o`   | `out`                   | Output directory.                                       |
| `--gpu`                 | `0`                     | GPU index (sets `CUDA_VISIBLE_DEVICES`).                |
| `--disparity-perc`      | `0.05`                  | Max disparity as a fraction of width (stereo strength). |
| `--variant`             | `full-attn`             | `full-attn` or `no-full-attn` weights.                  |
| `--max-res`             | `1024`                  | DepthCrafter resolution cap.                            |
| `--num-inference-steps` | `25`                    | DepthCrafter denoising steps.                           |
| `--no-auto-crop`        | off                     | Error instead of auto-cropping to a multiple of 64.     |
| `--keep-intermediates`  | off                     | Keep the `out/_work/` files.                            |
| `--depthcrafter-env`    | `depthcrafter`          | Conda env name for the depth stage.                     |
| `--sgm-env`             | `sgm`                   | Conda env name for warp + inpaint.                      |
| `--ca-bundle`           | `~/corp-ca-bundle.pem`  | CA bundle for runtime downloads (used if the file exists). |
| `--chunk-frames`        | `25`                    | Inpaint window size in frames (see *Long videos* below). |

### What to expect on first run

- **Input width & height must be divisible by 64.** The wrapper center-crops to
  the nearest valid size by default (`--no-auto-crop` to disable).
- The model was trained at **512×512**. For much higher resolutions, upstream
  recommends StereoCrafter-style tiling (not wired into this wrapper).
- The first **inpaint+refine** run downloads a few auxiliary files at runtime
  (a ~528 MB SVD component, `vgg_lpips` → `ckpts/vgg.pth`, and torchvision's
  `vgg16` weights into `~/.cache/torch/hub`). These honor `--ca-bundle`.
- The inpaint stage's **model-load phase is slow and silent** (~4–5 min, no
  progress bar) before generation begins. Logs sitting at `Global seed set` /
  `TypedStorage` warnings is **normal**, not a hang — confirm with `nvidia-smi`
  (GPU memory allocated) if unsure.
- `Restored from ckpts/m2svid_weights.pt with 0 missing and 0 unexpected keys`
  means the weights loaded correctly.
- The wrapper does **not** cache stages; every run re-runs depth (~25 s on the
  demo clip).

### Long videos (> 25 frames)

The M2SVid model has a **fixed temporal window of 25 frames** — upstream's
inpaint script only supports clips that short. This wrapper works around it:
`scripts/inpaint_chunked.py` loads the model **once** and runs it over
consecutive `--chunk-frames`-sized windows (default 25, padding the final
short window and trimming it back), then concatenates the result. DepthCrafter
and the warping stage already handle arbitrary-length video and are unaffected.

Caveat: windows are non-overlapping, so a subtle seam / disparity discontinuity
can appear every `--chunk-frames` frames. There is no overlap-blend yet.

### Verifying upstream directly (optional)

Upstream's `m2svid/inference.sh` assumes conda at `/opt/conda` and won't work
with the user-space Miniforge here — `vid2stereo.py` uses `conda run -n <env>`
instead. Prefer the wrapper.

## Layout

```
vid2stereo.py                 # orchestrator CLI
setup.sh                      # env + weights bootstrap (WSL2), idempotent
scripts/
  export-windows-ca.ps1       # export Windows trust store → corp-ca-bundle.pem
  net-check.sh, ca-diagnose.sh# proxy / TLS diagnostics
  sanity.py                   # sgm-env import check
docs/IMPLEMENTATION_PLAN.md   # full plan & design notes
m2svid/                       # cloned upstream repo (created by setup.sh)
  ckpts/                      # downloaded weights
  demo/input.mp4              # 25-frame 512×512 demo clip
out/                          # results (out/_work/ holds intermediates)
```

## Troubleshooting

| Symptom | Cause / fix |
| ------- | ----------- |
| `python: command not found` | Activate base first: `conda activate base`. |
| `ModuleNotFoundError: fire` (depth stage) | `depthcrafter` env incomplete — re-run `setup.sh`. |
| `ffmpeg` / `ffprobe` not found under a stage | ffmpeg must be **inside** the env: `conda install -n <env> -c conda-forge ffmpeg`. |
| Cert / SSL errors during download | Set up the CA bundle (see above); use `curl` not `wget`. |
| Anaconda channel 403 | Use the prefix.dev mirror (`setup.sh` configures `~/.condarc`). |
| Inpaint "stuck" for minutes | Normal model load — see *What to expect on first run*. |

## Credits

Method & upstream code: Shvetsova et al., *M2SVid* (Apache-2.0). Depth:
[DepthCrafter](https://github.com/Tencent/DepthCrafter).

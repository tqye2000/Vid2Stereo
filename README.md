# Vid2Stereo

Convert a single **monocular** video into a **stereo pair** using
[google-research/m2svid](https://github.com/google-research/m2svid)
(*M2SVid: End-to-End Inpainting and Refinement for Monocular-to-Stereo Video
Conversion*, 3DV 2026).

This repo is a **thin orchestrator** around the official m2svid code — it chains
the three pipeline stages into one command and post-processes the result into
side-by-side (SBS) plus separate left / right videos. It is **inference-only**
(uses pretrained weights, no training).

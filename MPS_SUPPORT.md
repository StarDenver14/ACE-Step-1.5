# MPS Support (macOS)

Last updated: 2026-02-04

This document captures the recommended changes to improve MPS support on macOS. No code changes are applied yet.

## Scope and goals

- Scope: inference only (training remains CUDA-focused).
- Goal: allow `auto` device selection to pick MPS, and allow explicit MPS selection in CLI/UI.
- Goal: remove CUDA hard-codes that crash on MPS/CPU.

## Device selection rules

- `auto` should prefer CUDA if available, else MPS if available, else CPU.
- `mps` should be an explicit option in CLI and Gradio UI.

## High-priority fixes (crash/incorrect behavior)

- Remove CUDA hard-code in LRC alignment score.
  - Issue: `calculate_score()` forces tensors onto `device='cuda'`, which fails on MPS/CPU.
  - File: `acestep/dit_alignment_score.py`

- Fix device check to recognize MPS (and other non-CUDA devices).
  - Issue: `_is_on_target_device()` treats any non-CPU device as CUDA.
  - File: `acestep/handler.py`

## Device selection / UX

- Allow selecting MPS in CLI and Gradio UI.
  - Issue: `--device` and dropdown only allow `auto/cuda/cpu`.
  - Files: `acestep/acestep_v15_pipeline.py`, `acestep/gradio_ui/interfaces/generation.py`

- Teach LM auto device selection to choose MPS when available.
  - Issue: LLM auto-select only considers CUDA or CPU.
  - File: `acestep/llm_inference.py`

## Defaults / backend behavior

- Default LM backend to PyTorch on non-CUDA systems.
  - Issue: default `vllm` is CUDA-only and falls back noisily on macOS.
  - Files: `acestep/acestep_v15_pipeline.py`, `acestep/gradio_ui/interfaces/generation.py`

## Lower priority / informational

- `acestep/gpu_config.py` is CUDA-only, so MPS machines are treated as "no GPU."
  - Impact: batch size/offload defaults may be conservative on MPS.
  - File: `acestep/gpu_config.py`

- Training code uses CUDA-specific autocast.
  - Impact: training on MPS likely unsupported, but inference is the focus.
  - File: `acestep/training/trainer.py`

## Acceptance criteria

- `--device mps` works end-to-end on macOS with Apple Silicon.
- `--device auto` selects MPS on macOS when CUDA is unavailable.
- No CUDA-only code paths are hit during MPS inference.

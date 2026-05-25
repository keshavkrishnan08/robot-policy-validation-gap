#!/usr/bin/env bash
# Part C (lean): real-robot external validation, PyTorch-only.
# Compute architecture-agnostic offline metrics on public SIMPLER policies and test whether
# they predict the published real-world success ranking better than validation MSE (r=0.308).
# Run inside the SimplerEnv-OpenVLA repo environment (PyTorch). One Kaggle session.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[C] Prereqs (Kaggle session, PyTorch env):"
echo "    git clone https://github.com/DelinQu/SimplerEnv-OpenVLA   # provides RT-2-X / OpenVLA wrappers"
echo "    download openvla/openvla-7b (HF); fits one T4 in bf16"
echo "    held-out demos: BridgeData V2 / Fractal slices from Open X-Embodiment"
echo
echo "[C] Steps:"
echo "  1. For each public policy (OpenVLA, RT-2-X, +RT-1 if reproducible), wrap it behind"
echo "     drc.external_validation.PolicyAdapter (predict / sample) and compute metrics:"
echo "       python -c \"from drc.external_validation import compute_external_metrics; ...\""
echo "     -> write external/<policy>_metrics.json"
echo "  2. Fill external/real_success.json from SIMPLER's released paired sim-real data"
echo "     (per-policy real-world success). DO NOT fabricate these numbers."
echo "  3. Combine + correlate (CPU):"
echo "       python scripts/06_external_validation.py"
echo
echo "Expected headline: a rollout-free coherence metric (or M5) predicts the real ranking"
echo "with |pearson| > 0.308 (SIMPLER's validation-MSE baseline)."

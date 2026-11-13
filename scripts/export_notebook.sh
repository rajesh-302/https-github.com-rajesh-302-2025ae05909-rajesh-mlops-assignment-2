#!/usr/bin/env bash
cd "$(dirname "$0")/.." || exit 1
jupyter nbconvert --to html notebooks/2025ae05909_mlops_assignment_2.ipynb \
  --output LLMOps_Assignment2.html || exit 1
if grep -E -q "(sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,}|gsk_[A-Za-z0-9]{20,})" \
     notebooks/2025ae05909_mlops_assignment_2.ipynb notebooks/LLMOps_Assignment2.html; then
  echo "WARNING: something that looks like an API key is in the notebook/HTML!"
  exit 1
fi
ls -la notebooks/LLMOps_Assignment2.html
echo "EXPORT OK (no API key found in notebook or HTML)"

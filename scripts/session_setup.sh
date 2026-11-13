#!/usr/bin/env bash
# Run at the start of EVERY lab session:   source scripts/session_setup.sh
PROJECT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$HOME/.venvs/llmops"
cd "$PROJECT" || return 1
if ! "$VENV/bin/python" -c "import eduassist, mlflow, openai, ipykernel" 2>/dev/null; then
  echo "[setup] environment missing or incomplete -> installing (3-6 min)"
  [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
  "$VENV/bin/pip" install --progress-bar off -q --upgrade pip || return 1
  "$VENV/bin/pip" install --progress-bar off -r requirements.txt \
    || { echo "[setup] INSTALL FAILED - copy the error above"; return 1; }
  "$VENV/bin/pip" install --progress-bar off -q -e . || return 1
fi
source "$VENV/bin/activate"
# Jupyter kernel that uses this environment (re-registered because $HOME may be new)
python -m ipykernel install --user --name llmops --display-name "Python (llmops)" >/dev/null 2>&1
[ -f .env ] || echo "[setup] WARNING: .env not found -> cp .env.example .env and fill it in"
echo "[setup] python: $(which python) | $(python --version)"
python -c "import eduassist, mlflow, openai; print('[setup] eduassist', eduassist.__version__, '| mlflow', mlflow.__version__, '| openai', openai.__version__)"

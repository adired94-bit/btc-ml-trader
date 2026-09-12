#!/usr/bin/env bash
# autoloop.sh - continuous self-improvement loop.
#
# Repeatedly launches Claude Code in non-interactive, permission-free mode with the
# improvement protocol from CLAUDE.md. Each iteration reads SYSTEM_LEARNINGS.md, tries
# exactly one hypothesis, validates it with pytest + walk-forward backtest, commits or
# reverts, and records the outcome.
#
# Environment variables:
#   AUTOLOOP_MAX_ITERATIONS  stop after N iterations (default: 0 = infinite)
#   AUTOLOOP_SLEEP_SECONDS   pause between iterations (default: 60)
#   AUTOLOOP_MODEL           optional model override passed to `claude --model`
#
# Usage:  chmod +x autoloop.sh && ./autoloop.sh
set -uo pipefail

cd "$(dirname "$0")"

MAX_ITER="${AUTOLOOP_MAX_ITERATIONS:-0}"
SLEEP="${AUTOLOOP_SLEEP_SECONDS:-60}"
MODEL_FLAG=()
if [[ -n "${AUTOLOOP_MODEL:-}" ]]; then
  MODEL_FLAG=(--model "$AUTOLOOP_MODEL")
fi

mkdir -p logs
LOG="logs/autoloop.log"

if ! command -v claude >/dev/null 2>&1; then
  echo "[autoloop] 'claude' CLI not found in PATH. Install Claude Code first." | tee -a "$LOG"
  exit 1
fi

read -r -d '' PROMPT <<'EOF' || true
You are running one iteration of the autonomous self-improvement loop for the BTC/USDT ML
trading platform in this repository. Follow CLAUDE.md section 5 exactly:

1. Read CLAUDE.md and SYSTEM_LEARNINGS.md. Identify the current best walk-forward metrics
   and the list of open hypotheses.
2. Choose ONE hypothesis (feature engineering, label definition, model regularisation,
   probability threshold, ATR multiplier, take-profit ratio, holding period...). Implement it
   in src/ or config.py only. Never modify tests/ to make them pass.
3. Run: venv/Scripts/python.exe -m pytest tests/  (must be green; otherwise fix or revert).
4. Run: venv/Scripts/python.exe -m src.models.train
   Then: venv/Scripts/python.exe -m src.backtest.engine --mode walk_forward
5. Compare with the current best in SYSTEM_LEARNINGS.md. Keep the change (git commit) only if
   walk-forward Sharpe improves without a worse max drawdown; otherwise revert with git.
6. Append a dated entry to SYSTEM_LEARNINGS.md (hypothesis, metrics, decision, next ideas)
   and commit it. Finish with a short summary of what changed.
EOF

iteration=0
while :; do
  iteration=$((iteration + 1))
  if [[ "$MAX_ITER" -gt 0 && "$iteration" -gt "$MAX_ITER" ]]; then
    echo "[autoloop] reached AUTOLOOP_MAX_ITERATIONS=$MAX_ITER, exiting." | tee -a "$LOG"
    break
  fi
  echo "[autoloop] === iteration $iteration started $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"

  claude --dangerously-skip-permissions "${MODEL_FLAG[@]}" -p "$PROMPT" 2>&1 | tee -a "$LOG"
  status=${PIPESTATUS[0]}
  echo "[autoloop] iteration $iteration finished with exit code $status" | tee -a "$LOG"

  # Safety net: never leave the repo red. If tests fail after an iteration, roll back.
  if ! venv/Scripts/python.exe -m pytest tests/ -q >/dev/null 2>&1; then
    echo "[autoloop] tests failing after iteration $iteration - reverting uncommitted changes" | tee -a "$LOG"
    git checkout -- . 2>>"$LOG" || true
  fi

  sleep "$SLEEP"
done

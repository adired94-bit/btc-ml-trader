"""Cross-platform launcher: trains models if needed, then runs the API and dashboard.

    python run.py            # both servers
    python run.py --api      # API only
    python run.py --ui       # dashboard only
"""

from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from config import settings  # noqa: E402
from src.models.train import DIRECTION_MODEL_FILE  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true", help="start only the FastAPI server")
    parser.add_argument("--ui", action="store_true", help="start only the Streamlit dashboard")
    args = parser.parse_args()
    run_api = args.api or not args.ui
    run_ui = args.ui or not args.api

    py = sys.executable
    if run_api and not (settings.models_dir / DIRECTION_MODEL_FILE).exists():
        print("[run] no trained models found - training now...")
        subprocess.run([py, "-m", "src.models.train"], cwd=ROOT, check=True)

    procs: list[subprocess.Popen] = []
    try:
        if run_api:
            print(f"[run] API -> http://{settings.api_host}:{settings.api_port}")
            procs.append(subprocess.Popen(
                [py, "-m", "uvicorn", "src.api.main:app", "--host", settings.api_host, "--port", str(settings.api_port)], cwd=ROOT
            ))
            time.sleep(3)
        if run_ui:
            print(f"[run] dashboard -> http://localhost:{settings.dashboard_port}")
            procs.append(subprocess.Popen(
                [py, "-m", "streamlit", "run", "app.py", "--server.port", str(settings.dashboard_port), "--server.headless", "true"], cwd=ROOT
            ))
        while procs:
            for p in list(procs):
                if p.poll() is not None:
                    print(f"[run] process {p.pid} exited with {p.returncode}")
                    procs.remove(p)
            time.sleep(1)
        return 0
    except KeyboardInterrupt:
        print("\n[run] stopping...")
        return 0
    finally:
        for p in procs:
            if p.poll() is None:
                try:
                    p.send_signal(signal.SIGTERM)
                except Exception:  # noqa: BLE001
                    p.kill()


if __name__ == "__main__":
    raise SystemExit(main())

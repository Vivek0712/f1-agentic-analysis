"""Dashboard server: race selectors, S3-cached outputs, on-demand generation.

    python scripts/serve_dashboard.py [--port 8099] [--no-agents]

Endpoints:
    GET  /                     the dashboard (dashboard/index.html)
    GET  /api/seasons          season -> race catalog (Ergast 2011+ and OpenF1)
    GET  /api/bundle?key=K     cached outputs for a race: local, then S3, else 404
    POST /api/generate         {"key": K, "force": bool} -> background job
    GET  /api/job?key=K        job state: running | done | error

Cache policy: a visit serves the cached bundle (local or S3). Regenerate
forces a fresh deterministic run; agent briefs run fresh whenever none are
cached for the race (or on force). Configure with F1_OUTPUTS_BUCKET,
AWS_PROFILE / F1_AWS_PROFILE, F1_BEDROCK_MODEL_ID, F1_PROTECT_KEYS.
"""

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from f1agents.pipeline import service, insights  # noqa: E402

JOBS: dict = {}
JOBS_LOCK = threading.Lock()
INVOKE_AGENTS = True


def _run_job(key: str, force: bool):
    def progress(step: str):
        with JOBS_LOCK:
            JOBS[key]["step"] = step
    try:
        result = service.generate(key, force=force, invoke_agents=INVOKE_AGENTS,
                                  progress=progress)
        with JOBS_LOCK:
            JOBS[key] = {"state": "done", "note": result.get("note")}
    except Exception as e:
        with JOBS_LOCK:
            JOBS[key] = {"state": "error", "error": str(e)}


class Handler(BaseHTTPRequestHandler):

    def _json(self, obj, status=200):
        body = json.dumps(obj, default=str).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        url = urlparse(self.path)
        q = parse_qs(url.query)
        if url.path in ("/", "/index.html"):
            page = ROOT / "dashboard" / "index.html"
            if not page.exists():
                self._json({"error": "dashboard/index.html missing; run scripts/build_dashboard.py"}, 500)
                return
            body = page.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/chart.umd.min.js":
            lib = ROOT / "dashboard" / "chart.umd.min.js"
            if not lib.exists():
                self._json({"error": "chart.umd.min.js missing from dashboard/"}, 404)
                return
            body = lib.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif url.path == "/api/seasons":
            self._json(service.list_races())
        elif url.path == "/api/bundle":
            key = q.get("key", [""])[0]
            b = service.bundle(key) if key else None
            if b is None:
                self._json({"error": f"no cached outputs for {key}"}, 404)
            else:
                self._json(b)
        elif url.path == "/api/sessions":
            key = q.get("key", [""])[0]
            self._json({"key": key, "sessions": insights.meeting_sessions(key),
                        "qualifying": insights.ergast_qualifying(key)})
        elif url.path == "/api/session_pace":
            try:
                sk = int(q.get("session_key", ["0"])[0])
                self._json(insights.session_pace(sk))
            except Exception as e:
                self._json({"error": str(e)}, 500)
        elif url.path == "/api/season":
            try:
                self._json(insights.season_overview(int(q.get("year", ["2024"])[0])))
            except Exception as e:
                self._json({"error": str(e)}, 500)
        elif url.path == "/api/driver":
            try:
                self._json(insights.driver_season(int(q.get("year", ["2024"])[0]),
                                                  q.get("code", [""])[0]))
            except Exception as e:
                self._json({"error": str(e)}, 500)
        elif url.path == "/api/predictor":
            try:
                self._json(insights.predictor(int(q.get("year", ["2026"])[0])))
            except Exception as e:
                self._json({"error": str(e)}, 500)
        elif url.path == "/api/predictor_brief":
            try:
                self._json(insights.predictor_brief(int(q.get("year", ["2026"])[0]),
                                                    q.get("driver", [""])[0],
                                                    force=q.get("force", ["0"])[0] == "1",
                                                    cached_only=q.get("cached", ["0"])[0] == "1"))
            except Exception as e:
                self._json({"error": str(e)}, 500)
        elif url.path == "/api/insight_brief":
            try:
                sk = q.get("session_key", [None])[0]
                yr = q.get("year", [None])[0]
                self._json(insights.insight_brief(
                    q.get("kind", [""])[0],
                    session_key=int(sk) if sk else None,
                    year=int(yr) if yr else None,
                    code=q.get("code", [None])[0],
                    force=q.get("force", ["0"])[0] == "1",
                    cached_only=q.get("cached", ["0"])[0] == "1"))
            except Exception as e:
                self._json({"error": str(e)}, 500)
        elif url.path == "/api/job":
            key = q.get("key", [""])[0]
            with JOBS_LOCK:
                self._json(JOBS.get(key, {"state": "unknown"}))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/api/generate":
            self._json({"error": "not found"}, 404)
            return
        length = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            self._json({"error": "invalid JSON"}, 400)
            return
        key = req.get("key")
        force = bool(req.get("force"))
        if not key:
            self._json({"error": "key required"}, 400)
            return
        if service._find_ref(key) is None:
            self._json({"error": f"unknown race key: {key}"}, 400)
            return
        with JOBS_LOCK:
            if JOBS.get(key, {}).get("state") == "running":
                self._json({"state": "running", "note": "already in progress"})
                return
            JOBS[key] = {"state": "running", "step": "queued"}
        threading.Thread(target=_run_job, args=(key, force), daemon=True).start()
        self._json({"state": "running"})

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[serve_dashboard] {fmt % args}\n")


def main():
    global INVOKE_AGENTS
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8099)))
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                    help="bind address; 0.0.0.0 for containers")
    ap.add_argument("--no-agents", action="store_true",
                    help="skip Bedrock agent briefs during generation")
    args = ap.parse_args()
    INVOKE_AGENTS = not args.no_agents and os.environ.get("F1_AGENTS", "1") != "0"

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"dashboard -> http://{args.host}:{args.port}/  "
          f"(bucket s3://{service.S3_BUCKET}, agents {'on' if INVOKE_AGENTS else 'off'})",
          flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

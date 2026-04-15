#!/usr/bin/env python3

import argparse
import json
import mimetypes
import pathlib
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from libero_debug_dashboard_backend import (
    AVAILABLE_LIBERO_TYPES,
    AVAILABLE_SUITES,
    BUILD_TOKEN,
    DebugSessionController,
    _build_state,
    _normalize_model_path,
    _safe_relative_path,
)


def _derive_progress_state(state: dict, control: dict) -> dict:
    live_status = state.get("live_status") or {}
    tail = (control.get("tmux_tail") or "").lower()
    phase = str(live_status.get("phase") or "")
    session_alive = bool(control.get("session_alive"))
    current_run_dir = control.get("current_run_dir")

    if "traceback" in tail or "error:" in tail or "failed to" in tail:
        return {"percent": 100, "label": "Launch error. Check Current Prompt for details."}
    if phase == "finished":
        return {"percent": 100, "label": "Task finished. Use Set Task to keep debugging without reloading."}
    if phase == "awaiting_chunk_execution":
        return {"percent": 100, "label": "Chunk ready. Edit actions, Simulate, or Run."}
    if phase == "simulating_chunk":
        return {"percent": 96, "label": "Running dry-run simulation on the current scene..."}
    if phase == "chunk_complete":
        return {"percent": 100, "label": "Chunk finished."}
    if phase == "planning_next_chunk":
        return {"percent": 88, "label": "Chunk finished. Planning the next policy chunk..."}
    if phase == "switching_task":
        return {"percent": 18, "label": "Switching task without reloading model..."}
    if phase == "executing_chunk":
        chunk_idx = live_status.get("chunk_idx", "?")
        return {"percent": 94, "label": f"Executing chunk {chunk_idx}..."}
    if phase == "setup":
        return {"percent": 82, "label": "Environment ready. Preparing the first scene and chunk..."}
    if "chunk[" in tail or "policy_chunk_preview" in tail:
        return {"percent": 100, "label": "Chunk ready. Edit actions, Simulate, or Run."}
    if "policy setup done" in tail:
        return {"percent": 78, "label": "Policy setup done. Building environment..."}
    if "using bundled lerobot processor norm stats" in tail or "norm stats not found" in tail:
        return {"percent": 62, "label": "Normalization stats loaded."}
    if "checkpoint load finished" in tail or "stripped top-level" in tail:
        return {"percent": 52, "label": "Checkpoint weights loaded."}
    if "disabling torch.compile" in tail or "loading model" in tail:
        return {"percent": 28, "label": "Loading model checkpoint..."}
    if "using python at" in tail or control.get("session_alive"):
        return {"percent": 8, "label": "Launch command sent. Starting worker..."}
    if not session_alive and not current_run_dir:
        return {"percent": 0, "label": "Idle. Choose a task and press Set Task."}
    return {"percent": 0, "label": "Idle"}



from libero_debug_dashboard_frontend import render_dashboard_html


class DebugDashboardHandler(BaseHTTPRequestHandler):
    def _write_bytes(self, payload: bytes, content_type: str, status: int = 200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _write_json(self, payload: dict, status: int = 200):
        self._write_bytes(
            json.dumps(payload, ensure_ascii=True).encode("utf-8"),
            "application/json; charset=utf-8",
            status,
        )

    def _controller(self) -> DebugSessionController:
        return self.server.controller

    def do_GET(self):
        parsed_url = urlsplit(self.path)
        request_path = parsed_url.path
        query = parse_qs(parsed_url.query)

        if request_path.startswith("/api/state"):
            control = self._controller().control_state()
            payload = _build_state(self._controller().current_run_dir, self.server.trace_tail)
            payload["control"] = control
            payload["progress"] = _derive_progress_state(payload, control)
            self._write_json(payload)
            return

        if request_path == "/api/task_catalog":
            libero_type = query.get("libero_type", [self._controller().current_libero_type()])[0]
            payload = self._controller().task_catalog(libero_type)
            self._write_json(payload)
            return

        if request_path == "/" or request_path.startswith("/index.html"):
            self._write_bytes(
                render_dashboard_html(
                    self.server.refresh_ms,
                    BUILD_TOKEN,
                    self._controller().available_libero_types,
                    AVAILABLE_SUITES,
                ).encode("utf-8"),
                "text/html; charset=utf-8",
            )
            return

        if request_path.startswith("/artifacts/"):
            try:
                rel_path = _safe_relative_path(request_path[len("/artifacts/") :])
            except ValueError as exc:
                self._write_bytes(str(exc).encode("utf-8"), "text/plain; charset=utf-8", HTTPStatus.BAD_REQUEST)
                return

            run_dir = self._controller().current_run_dir
            if run_dir is None:
                self._write_bytes(b"no active run_dir", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
                return

            file_path = (run_dir / rel_path).resolve()
            try:
                file_path.relative_to(run_dir.resolve())
            except ValueError:
                self._write_bytes(b"forbidden", "text/plain; charset=utf-8", HTTPStatus.FORBIDDEN)
                return

            if not file_path.exists() or not file_path.is_file():
                self._write_bytes(b"not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)
                return

            content_type, _ = mimetypes.guess_type(str(file_path))
            self._write_bytes(file_path.read_bytes(), content_type or "application/octet-stream")
            return

        self._write_bytes(b"not found", "text/plain; charset=utf-8", HTTPStatus.NOT_FOUND)

    def do_POST(self):
        request_path = urlsplit(self.path).path
        content_length = int(self.headers.get("Content-Length", "0"))
        raw_payload = self.rfile.read(content_length) if content_length > 0 else b"{}"

        try:
            payload = json.loads(raw_payload.decode("utf-8"))
        except json.JSONDecodeError:
            self._write_json({"error": "Invalid JSON body."}, HTTPStatus.BAD_REQUEST)
            return

        try:
            if request_path == "/api/launch":
                result = self._controller().launch(payload)
                self._write_json(result)
                return

            if request_path == "/api/set_task":
                phase = self._controller().current_phase()
                session_alive = self._controller().session_alive()
                current_run_dir = self._controller().current_run_dir
                requested_type = str(payload.get("libero_type") or self._controller().current_libero_type())
                current_type = self._controller().current_libero_type()
                requested_model_path = str(payload.get("model_path") or self._controller().current_model_path())
                current_model_path = self._controller().current_model_path()
                if (
                    session_alive
                    and current_run_dir is not None
                    and
                    requested_type == current_type
                    and requested_model_path == current_model_path
                    and phase not in {"awaiting_chunk_execution", "chunk_complete", "finished"}
                ):
                    raise RuntimeError(f"Set Task is only available when paused at a prompt. current_phase={phase!r}")
                result = self._controller().set_task(
                    libero_type=requested_type,
                    suite_name=str(payload.get("suite_name")),
                    task_id=int(payload.get("task_id")),
                    trial_idx=int(payload.get("trial_idx", 0)),
                    model_path=requested_model_path,
                )
                self._write_json(result)
                return

            if request_path == "/api/reset":
                result = self._controller().reset_session()
                self._write_json(result)
                return

            if request_path == "/api/simulate":
                phase = self._controller().current_phase()
                if phase != "awaiting_chunk_execution":
                    raise RuntimeError(
                        f"Simulate Chunk is only available when a chunk is waiting for execution. current_phase={phase!r}"
                    )
                result = self._controller().simulate_chunk(
                    actions=payload.get("actions"),
                    chunk_idx=payload.get("chunk_idx"),
                )
                self._write_json(result)
                return

            if request_path == "/api/run_chunk":
                phase = self._controller().current_phase()
                if phase != "awaiting_chunk_execution":
                    raise RuntimeError(
                        f"Run Chunk is only available when a chunk is waiting for execution. current_phase={phase!r}"
                    )
                result = self._controller().run_chunk(
                    actions=payload.get("actions"),
                    chunk_idx=payload.get("chunk_idx"),
                )
                self._write_json(result)
                return

            if request_path == "/api/send":
                action = payload.get("action")
                if action == "continue":
                    result = self._controller().send_keys(None, press_enter=True)
                else:
                    result = self._controller().send_keys(
                        payload.get("text"),
                        press_enter=bool(payload.get("press_enter", True)),
                    )
                self._write_json(result)
                return
        except Exception as exc:
            self._write_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        self._write_json({"error": "Unsupported endpoint."}, HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args):
        return


def main():
    repo_root = pathlib.Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True, help="Initial debug run directory to serve.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind.")
    parser.add_argument("--port", type=int, default=8765, help="Port to bind.")
    parser.add_argument("--refresh_ms", type=int, default=1500, help="Browser polling interval in ms.")
    parser.add_argument("--trace_tail", type=int, default=20, help="How many trace records to show.")
    parser.add_argument("--repo_path", type=str, default=str(repo_root), help="RLinf repository root.")
    parser.add_argument(
        "--openpi_venv_python",
        type=str,
        default=str(repo_root / ".venv-openpi-liberoplus" / "bin" / "python"),
        help="Python executable for the OpenPI/libero-plus environment.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/data/models/pi05_libero_finetuned_v044",
        help="Model path used when launching new runs from the web UI.",
    )
    parser.add_argument(
        "--debug_root",
        type=str,
        default="/data/inference_results/liberoplus_action_debug",
        help="Root directory where launched debug runs should be written.",
    )
    parser.add_argument(
        "--launcher_script",
        type=str,
        default=str(repo_root / "examples/embodiment/run_liberoplus_debug_action_chunk_openpi_pi05_base.sh"),
        help="Launcher script used to start new debug runs.",
    )
    parser.add_argument(
        "--session_name",
        type=str,
        default="libero_debug_worker",
        help="tmux session name to control from the web UI.",
    )
    parser.add_argument("--default_libero_type", type=str, default="plus")
    parser.add_argument("--default_suite", type=str, default="libero_goal")
    parser.add_argument("--default_task_id", type=int, default=None)
    parser.add_argument("--default_trial_idx", type=int, default=0)
    parser.add_argument("--default_action_chunk", type=int, default=10)
    parser.add_argument("--default_num_steps", type=int, default=10)
    parser.add_argument("--default_num_steps_wait", type=int, default=10)
    parser.add_argument("--default_video_temp_subsample", type=int, default=1)
    args = parser.parse_args()

    run_dir = pathlib.Path(args.run_dir).resolve()
    if not run_dir.exists():
        run_dir = None

    default_config = {
        "libero_type": args.default_libero_type,
        "task_suite_name": args.default_suite,
        "task_id": args.default_task_id,
        "trial_idx": args.default_trial_idx,
        "model_path": str(_normalize_model_path(args.model_path)),
        "action_chunk": args.default_action_chunk,
        "num_steps": args.default_num_steps,
        "num_steps_wait": args.default_num_steps_wait,
        "stop_after_chunks": None,
        "video_temp_subsample": args.default_video_temp_subsample,
        "interactive_chunk_edit": True,
        "pause_after_action": False,
        "save_chunk_videos": True,
        "save_rollout_video": True,
        "save_live_preview": True,
        "print_policy_chunks": True,
        "print_observation_state": True,
        "print_step_state": True,
        "exp_name": "",
    }

    controller = DebugSessionController(
        run_dir=run_dir,
        repo_path=pathlib.Path(args.repo_path),
        openpi_venv_python=pathlib.Path(args.openpi_venv_python),
        model_path=pathlib.Path(args.model_path),
        debug_root=pathlib.Path(args.debug_root),
        launcher_script=pathlib.Path(args.launcher_script),
        session_name=args.session_name,
        default_config=default_config,
    )
    launch_result = controller.ensure_session_started()

    server = ThreadingHTTPServer((args.host, args.port), DebugDashboardHandler)
    server.controller = controller
    server.refresh_ms = args.refresh_ms
    server.trace_tail = args.trace_tail
    served_run_dir = controller.current_run_dir
    if launch_result is not None:
        served_run_dir = pathlib.Path(launch_result["run_dir"]).resolve()
    print(f"Serving {served_run_dir} at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()

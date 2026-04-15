#!/usr/bin/env python3

import argparse
import datetime as dt
import json
import mimetypes
import os
import pathlib
import posixpath
import re
import shlex
import subprocess
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

AVAILABLE_LIBERO_TYPES = ["plus", "pro", "standard"]
AVAILABLE_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]
BUILD_TOKEN = dt.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def _build_openpi_pythonpath(repo_path: pathlib.Path, openpi_venv_python: pathlib.Path) -> str:
    openpi_venv_root = openpi_venv_python.parent.parent
    pythonpath_parts = [str(repo_path)]
    libero_site = openpi_venv_root / "libero"
    libero_plus_site = openpi_venv_root / "libero_plus"
    if libero_site.exists():
        pythonpath_parts.append(str(libero_site))
    if libero_plus_site.exists():
        pythonpath_parts.append(str(libero_plus_site))
    existing_pythonpath = os.environ.get("PYTHONPATH")
    if existing_pythonpath:
        pythonpath_parts.append(existing_pythonpath)
    return ":".join(pythonpath_parts)


def _safe_relative_path(raw_path: str) -> pathlib.PurePosixPath:
    normalized = posixpath.normpath(unquote(raw_path)).lstrip("/")
    path = pathlib.PurePosixPath(normalized)
    if normalized in {"", "."}:
        return pathlib.PurePosixPath(".")
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid path: {raw_path}")
    return path


def _tail_jsonl(path: pathlib.Path, max_records: int) -> list[dict]:
    if not path.exists():
        return []
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records[-max_records:]


def _empty_run_state(run_dir: pathlib.Path | None) -> dict:
    return {
        "run_dir": str(run_dir) if run_dir else None,
        "last_completed_chunk_gif": None,
        "last_completed_chunk_gif_mtime_ns": None,
        "last_completed_chunk_mp4": None,
        "last_completed_chunk_mp4_mtime_ns": None,
        "latest_chunk_gif": None,
        "chunk_gifs": [],
        "live_chunk_gif": None,
        "live_chunk_gif_mtime_ns": None,
        "live_chunk_video": None,
        "live_chunk_video_mtime_ns": None,
        "live_latest_frame": None,
        "live_latest_frame_mtime_ns": None,
        "live_status": {},
        "latest_chunk_video": None,
        "chunk_videos": [],
        "latest_rollout_video": None,
        "rollout_videos": [],
        "summary": {},
        "trace_tail": [],
    }


def _build_state(run_dir: pathlib.Path | None, trace_tail: int) -> dict:
    if run_dir is None or not run_dir.exists():
        return _empty_run_state(run_dir)

    chunk_gifs = sorted(run_dir.glob("chunk_videos/*.gif"), key=lambda path: path.name)
    chunk_videos = sorted(run_dir.glob("chunk_videos/*.mp4"), key=lambda path: path.name)
    rollout_videos = sorted(run_dir.glob("videos/**/*.mp4"), key=lambda path: path.name)

    live_dir = run_dir / "live"
    summary_path = run_dir / "debug_summary.json"
    trace_path = run_dir / "debug_trace.jsonl"
    live_status_path = live_dir / "status.json"
    last_completed_chunk_gif_path = live_dir / "last_completed_chunk.gif"
    last_completed_chunk_mp4_path = live_dir / "last_completed_chunk.mp4"
    live_chunk_video_path = live_dir / "current_chunk.mp4"
    live_chunk_gif_path = live_dir / "current_chunk.gif"
    live_latest_frame_path = live_dir / "latest_frame.jpg"

    summary = {}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))

    live_status = {}
    if live_status_path.exists():
        live_status = json.loads(live_status_path.read_text(encoding="utf-8"))

    return {
        "run_dir": str(run_dir),
        "last_completed_chunk_gif": (
            str(last_completed_chunk_gif_path.relative_to(run_dir))
            if last_completed_chunk_gif_path.exists()
            else None
        ),
        "last_completed_chunk_gif_mtime_ns": (
            last_completed_chunk_gif_path.stat().st_mtime_ns
            if last_completed_chunk_gif_path.exists()
            else None
        ),
        "last_completed_chunk_mp4": (
            str(last_completed_chunk_mp4_path.relative_to(run_dir))
            if last_completed_chunk_mp4_path.exists()
            else None
        ),
        "last_completed_chunk_mp4_mtime_ns": (
            last_completed_chunk_mp4_path.stat().st_mtime_ns
            if last_completed_chunk_mp4_path.exists()
            else None
        ),
        "latest_chunk_gif": str(chunk_gifs[-1].relative_to(run_dir)) if chunk_gifs else None,
        "chunk_gifs": [str(path.relative_to(run_dir)) for path in chunk_gifs],
        "live_chunk_gif": str(live_chunk_gif_path.relative_to(run_dir)) if live_chunk_gif_path.exists() else None,
        "live_chunk_gif_mtime_ns": (
            live_chunk_gif_path.stat().st_mtime_ns if live_chunk_gif_path.exists() else None
        ),
        "live_chunk_video": str(live_chunk_video_path.relative_to(run_dir)) if live_chunk_video_path.exists() else None,
        "live_chunk_video_mtime_ns": (
            live_chunk_video_path.stat().st_mtime_ns if live_chunk_video_path.exists() else None
        ),
        "live_latest_frame": (
            str(live_latest_frame_path.relative_to(run_dir))
            if live_latest_frame_path.exists()
            else None
        ),
        "live_latest_frame_mtime_ns": (
            live_latest_frame_path.stat().st_mtime_ns
            if live_latest_frame_path.exists()
            else None
        ),
        "live_status": live_status,
        "latest_chunk_video": str(chunk_videos[-1].relative_to(run_dir)) if chunk_videos else None,
        "chunk_videos": [str(path.relative_to(run_dir)) for path in chunk_videos],
        "latest_rollout_video": str(rollout_videos[-1].relative_to(run_dir)) if rollout_videos else None,
        "rollout_videos": [str(path.relative_to(run_dir)) for path in rollout_videos],
        "summary": summary,
        "trace_tail": _tail_jsonl(trace_path, trace_tail),
    }


def _sanitize_exp_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return safe or "libero_debug_web"


class DebugSessionController:
    def __init__(
        self,
        *,
        run_dir: pathlib.Path | None,
        repo_path: pathlib.Path,
        openpi_venv_python: pathlib.Path,
        model_path: pathlib.Path,
        debug_root: pathlib.Path,
        launcher_script: pathlib.Path,
        session_name: str,
        default_config: dict,
    ):
        self.current_run_dir = run_dir.resolve() if run_dir is not None else None
        self.repo_path = repo_path.resolve()
        self.openpi_venv_python = openpi_venv_python if openpi_venv_python.is_absolute() else openpi_venv_python.resolve()
        self.model_path = model_path.resolve()
        self.debug_root = debug_root.resolve()
        self.launcher_script = launcher_script.resolve()
        self.session_name = session_name
        self.default_config = default_config
        self.last_launch_config = default_config.copy()
        self.state_file = self.debug_root / ".dashboard_controller_state.json"
        self.openpi_pythonpath = _build_openpi_pythonpath(self.repo_path, self.openpi_venv_python)
        self.available_libero_types = self._detect_available_libero_types()
        self._task_catalog_cache: dict[str, dict] = {}
        self._load_state_file()

    def _detect_available_libero_types(self) -> list[str]:
        import_scripts = {
            "plus": "import liberoplus.liberoplus",
            "pro": "import liberopro.liberopro",
            "standard": "import libero.libero",
        }
        available = []
        for libero_type in AVAILABLE_LIBERO_TYPES:
            env = dict(os.environ)
            env["PYTHONPATH"] = self.openpi_pythonpath
            result = subprocess.run(
                [str(self.openpi_venv_python), "-c", import_scripts[libero_type]],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
                env=env,
            )
            if result.returncode == 0:
                available.append(libero_type)
        return available or ["plus"]

    def _load_state_file(self):
        if not self.state_file.exists():
            return
        try:
            payload = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return
        run_dir = payload.get("current_run_dir")
        if run_dir:
            path = pathlib.Path(run_dir)
            if path.exists():
                self.current_run_dir = path
        last_launch_config = payload.get("last_launch_config")
        if isinstance(last_launch_config, dict):
            self.last_launch_config.update(last_launch_config)

    def _save_state_file(self):
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.state_file.write_text(
            json.dumps(
                {
                    "current_run_dir": str(self.current_run_dir) if self.current_run_dir else None,
                    "last_launch_config": self.last_launch_config,
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )

    def session_alive(self) -> bool:
        result = subprocess.run(
            ["tmux", "has-session", "-t", self.session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return result.returncode == 0

    def capture_pane(self, lines: int = 80) -> str:
        if not self.session_alive():
            return ""
        result = subprocess.run(
            ["tmux", "capture-pane", "-pt", self.session_name, "-S", f"-{lines}"],
            text=True,
            capture_output=True,
            check=False,
        )
        return result.stdout

    def _infer_run_dir_from_tmux_tail(self, tail: str) -> pathlib.Path | None:
        if not tail:
            return None

        matches = re.findall(r"run_dir=([^\s]+)", tail)
        if matches:
            candidate = pathlib.Path(matches[-1])
            if candidate.exists():
                return candidate

        launcher_match = re.search(r"--log_dir\s+([^\s]+)\s+--exp_name\s+([^\s]+)", tail)
        if launcher_match:
            log_dir = pathlib.Path(launcher_match.group(1))
            exp_name = launcher_match.group(2)
            candidate = log_dir / exp_name
            if candidate.exists():
                return candidate

        return None

    def sync_current_run_dir(self) -> str:
        tail = self.capture_pane(lines=200)
        inferred = self._infer_run_dir_from_tmux_tail(tail)
        if inferred is not None and inferred != self.current_run_dir:
            self.current_run_dir = inferred
            self._save_state_file()
        return tail

    def send_keys(self, text: str | None = None, *, press_enter: bool = True) -> dict:
        if not self.session_alive():
            raise RuntimeError(f"tmux session '{self.session_name}' is not running.")

        cmd = ["tmux", "send-keys", "-t", self.session_name]
        if text:
            cmd.append(text)
        if press_enter:
            cmd.append("C-m")
        subprocess.run(cmd, check=True)
        return {"ok": True, "session_name": self.session_name, "sent": text or "", "press_enter": press_enter}

    def send_commands(self, commands: list[str]) -> dict:
        if not commands:
            raise ValueError("No commands were provided.")
        results = []
        for command in commands:
            if command is None:
                results.append(self.send_keys(None, press_enter=True))
            else:
                results.append(self.send_keys(command, press_enter=True))
        return {
            "ok": True,
            "session_name": self.session_name,
            "commands": commands,
            "results": results,
        }

    def current_live_status(self) -> dict:
        if self.current_run_dir is None:
            return {}
        return _build_state(self.current_run_dir, 0).get("live_status") or {}

    def current_phase(self) -> str:
        return str(self.current_live_status().get("phase") or "")

    def current_libero_type(self) -> str:
        live_status = self.current_live_status()
        live_type = str(live_status.get("libero_type") or "")
        if live_type in self.available_libero_types:
            return live_type
        remembered_type = str(self.last_launch_config.get("libero_type") or "")
        if remembered_type in self.available_libero_types:
            return remembered_type
        default_type = self.default_config["libero_type"]
        if default_type in self.available_libero_types:
            return default_type
        return self.available_libero_types[0]

    def ensure_session_started(self) -> dict | None:
        if self.session_alive():
            return None
        return self.launch(self.last_launch_config)

    def task_catalog(self, libero_type: str) -> dict:
        normalized_type = str(libero_type or self.current_libero_type())
        if normalized_type not in AVAILABLE_LIBERO_TYPES:
            raise ValueError(f"Unsupported libero_type={normalized_type}")
        if normalized_type in self._task_catalog_cache:
            return self._task_catalog_cache[normalized_type]

        catalog_script = r"""
import contextlib
import io
import json
import pathlib
import sys

repo_path = pathlib.Path(sys.argv[1]).resolve()
libero_type = sys.argv[2]
suite_names = sys.argv[3:]
if str(repo_path) not in sys.path:
    sys.path.insert(0, str(repo_path))

from toolkits.eval_scripts_openpi.libero_eval import _import_libero_stack, _load_taxonomy_lookup

benchmark_module, _get_libero_path_fn, _env_cls, package_root = _import_libero_stack(libero_type, None)
taxonomy_lookup = _load_taxonomy_lookup(package_root, suite_names) if libero_type == "plus" else {}
benchmark_dict = benchmark_module.get_benchmark_dict()

payload = {
    "libero_type": libero_type,
    "suite_task_counts": {},
    "task_taxonomy_by_suite": {},
}

for suite_name in suite_names:
    if suite_name not in benchmark_dict:
        payload["suite_task_counts"][suite_name] = None
        payload["task_taxonomy_by_suite"][suite_name] = {}
        continue
    with contextlib.redirect_stdout(io.StringIO()):
        task_suite = benchmark_dict[suite_name]()
    payload["suite_task_counts"][suite_name] = int(task_suite.n_tasks)
    suite_taxonomy = {}
    if libero_type == "plus":
        for task_id in range(task_suite.n_tasks):
            task = task_suite.get_task(task_id)
            task_name = pathlib.Path(task.bddl_file).stem
            suite_taxonomy[str(task_id)] = taxonomy_lookup.get(suite_name, {}).get(task_name, "Unknown")
    payload["task_taxonomy_by_suite"][suite_name] = suite_taxonomy

print(json.dumps(payload))
"""
        catalog_env = dict(os.environ)
        catalog_env["PYTHONPATH"] = self.openpi_pythonpath
        result = subprocess.run(
            [
                str(self.openpi_venv_python),
                "-c",
                catalog_script,
                str(self.repo_path),
                normalized_type,
                *AVAILABLE_SUITES,
            ],
            text=True,
            capture_output=True,
            env=catalog_env,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "Failed to load task catalog: "
                + (result.stderr.strip() or result.stdout.strip() or f"exit code {result.returncode}")
            )
        catalog = json.loads(result.stdout)
        self._task_catalog_cache[normalized_type] = catalog
        return catalog

    def set_task(self, *, libero_type: str, suite_name: str, task_id: int, trial_idx: int) -> dict:
        target_type = str(libero_type or self.current_libero_type())
        if target_type not in self.available_libero_types:
            raise ValueError(f"Unsupported libero_type={target_type}")
        if (not self.session_alive()) or target_type != self.current_libero_type():
            launch_config = self.last_launch_config.copy()
            launch_config.update(
                {
                    "libero_type": target_type,
                    "task_suite_name": suite_name,
                    "task_id": task_id,
                    "trial_idx": trial_idx,
                }
            )
            return self.launch(launch_config)
        command = f"switch {suite_name} {task_id} {trial_idx}"
        return self.send_commands([command])

    def _normalize_actions(self, actions) -> list[list[float]]:
        if not isinstance(actions, list) or not actions:
            raise ValueError("actions must be a non-empty 2D list.")
        normalized = []
        expected_width = None
        for row in actions:
            if not isinstance(row, list) or not row:
                raise ValueError("Each action row must be a non-empty list.")
            normalized_row = [float(value) for value in row]
            if expected_width is None:
                expected_width = len(normalized_row)
            elif len(normalized_row) != expected_width:
                raise ValueError("All action rows must have the same width.")
            normalized.append(normalized_row)
        return normalized

    def _write_chunk_override(self, actions: list[list[float]], chunk_idx: int | None = None) -> pathlib.Path:
        if self.current_run_dir is None:
            raise RuntimeError("No active run directory is available yet.")
        out_dir = self.current_run_dir / "dashboard_temp"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S-%f")
        chunk_suffix = "latest" if chunk_idx is None else f"{int(chunk_idx):03d}"
        out_path = out_dir / f"chunk_{chunk_suffix}_{stamp}.json"
        out_path.write_text(json.dumps({"actions": actions}, indent=2), encoding="utf-8")
        return out_path

    def _dispatch_chunk_command(
        self,
        *,
        command_name: str,
        actions: list[list[float]] | None = None,
        chunk_idx: int | None = None,
    ) -> dict:
        commands = []
        override_path = None
        if actions is not None:
            normalized_actions = self._normalize_actions(actions)
            override_path = self._write_chunk_override(normalized_actions, chunk_idx=chunk_idx)
            commands.append(f"load {shlex.quote(str(override_path))}")
        commands.append(command_name)
        result = self.send_commands(commands)
        if override_path is not None:
            result["override_path"] = str(override_path)
        return result

    def simulate_chunk(self, *, actions: list[list[float]] | None = None, chunk_idx: int | None = None) -> dict:
        return self._dispatch_chunk_command(
            command_name="simulate",
            actions=actions,
            chunk_idx=chunk_idx,
        )

    def run_chunk(self, *, actions: list[list[float]] | None = None, chunk_idx: int | None = None) -> dict:
        return self._dispatch_chunk_command(
            command_name="run",
            actions=actions,
            chunk_idx=chunk_idx,
        )

    def _normalize_launch_config(self, payload: dict) -> dict:
        config = self.default_config.copy()
        config.update(payload or {})

        libero_type = str(config.get("libero_type", self.default_config["libero_type"]))
        suite_name = str(config.get("task_suite_name", self.default_config["task_suite_name"]))
        if libero_type not in self.available_libero_types:
            raise ValueError(f"Unsupported libero_type={libero_type}")
        if suite_name not in AVAILABLE_SUITES:
            raise ValueError(f"Unsupported task_suite_name={suite_name}")

        normalized = {
            "libero_type": libero_type,
            "task_suite_name": suite_name,
            "task_id": int(config.get("task_id", self.default_config["task_id"])),
            "trial_idx": int(config.get("trial_idx", self.default_config["trial_idx"])),
            "action_chunk": int(config.get("action_chunk", self.default_config["action_chunk"])),
            "num_steps": int(config.get("num_steps", self.default_config["num_steps"])),
            "num_steps_wait": int(config.get("num_steps_wait", self.default_config["num_steps_wait"])),
            "video_temp_subsample": int(
                config.get("video_temp_subsample", self.default_config["video_temp_subsample"])
            ),
            "pause_after_action": bool(config.get("pause_after_action", self.default_config["pause_after_action"])),
            "interactive_chunk_edit": bool(
                config.get("interactive_chunk_edit", self.default_config["interactive_chunk_edit"])
            ),
            "save_rollout_video": bool(
                config.get("save_rollout_video", self.default_config["save_rollout_video"])
            ),
            "save_chunk_videos": bool(
                config.get("save_chunk_videos", self.default_config["save_chunk_videos"])
            ),
            "save_live_preview": bool(
                config.get("save_live_preview", self.default_config["save_live_preview"])
            ),
            "print_policy_chunks": bool(
                config.get("print_policy_chunks", self.default_config["print_policy_chunks"])
            ),
            "print_observation_state": bool(
                config.get("print_observation_state", self.default_config["print_observation_state"])
            ),
            "print_step_state": bool(
                config.get("print_step_state", self.default_config["print_step_state"])
            ),
        }

        stop_after_chunks = config.get("stop_after_chunks")
        normalized["stop_after_chunks"] = None if stop_after_chunks in (None, "", "null") else int(stop_after_chunks)

        exp_name = config.get("exp_name")
        if exp_name:
            normalized["exp_name"] = _sanitize_exp_name(str(exp_name))
        else:
            normalized["exp_name"] = _sanitize_exp_name(
                f"web_{libero_type}_{suite_name}_task{normalized['task_id']}_trial{normalized['trial_idx']}"
            )

        return normalized

    def launch(self, payload: dict) -> dict:
        config = self._normalize_launch_config(payload)

        run_stamp = dt.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        run_root = self.debug_root / f"{run_stamp}-{config['exp_name']}"
        run_dir = run_root / config["exp_name"]

        subprocess.run(
            ["tmux", "kill-session", "-t", self.session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

        env_vars = {
            "OPENPI_VENV_PYTHON": str(self.openpi_venv_python),
            "PI05_MODEL_PATH": str(self.model_path),
            "RLINF_DEBUG_ROOT": str(self.debug_root),
            "LIBERO_BENCHMARK_TYPE": config["libero_type"],
            "LIBERO_TASK_SUITE_NAME": config["task_suite_name"],
            "LIBERO_TASK_ID": str(config["task_id"]),
            "LIBERO_TRIAL_IDX": str(config["trial_idx"]),
            "ACTION_CHUNK": str(config["action_chunk"]),
            "NUM_STEPS": str(config["num_steps"]),
            "NUM_STEPS_WAIT": str(config["num_steps_wait"]),
            "VIDEO_TEMP_SUBSAMPLE": str(config["video_temp_subsample"]),
            "INTERACTIVE_CHUNK_EDIT": "1" if config["interactive_chunk_edit"] else "0",
            "PAUSE_AFTER_ACTION": "1" if config["pause_after_action"] else "0",
            "PRINT_POLICY_CHUNKS": "1" if config["print_policy_chunks"] else "0",
            "PRINT_OBSERVATION_STATE": "1" if config["print_observation_state"] else "0",
            "PRINT_STEP_STATE": "1" if config["print_step_state"] else "0",
            "SAVE_ROLLOUT_VIDEO": "1" if config["save_rollout_video"] else "0",
            "SAVE_CHUNK_VIDEOS": "1" if config["save_chunk_videos"] else "0",
            "SAVE_LIVE_PREVIEW": "1" if config["save_live_preview"] else "0",
            "RUN_STAMP": run_stamp,
            "EXP_NAME": config["exp_name"],
        }
        if config["stop_after_chunks"] is not None:
            env_vars["STOP_AFTER_CHUNKS"] = str(config["stop_after_chunks"])

        env_prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in env_vars.items())
        shell_cmd = (
            f"cd {shlex.quote(str(self.repo_path))} && "
            f"{env_prefix} {shlex.quote(str(self.launcher_script))}; "
            "exec bash"
        )
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", self.session_name, "bash", "-lc", shell_cmd],
            check=True,
        )

        self.current_run_dir = run_dir
        self.last_launch_config = config.copy()
        self._save_state_file()
        return {
            "ok": True,
            "session_name": self.session_name,
            "run_dir": str(run_dir),
            "run_root": str(run_root),
            "launch_config": config,
        }

    def control_state(self) -> dict:
        tmux_tail = self.sync_current_run_dir()
        return {
            "session_name": self.session_name,
            "session_alive": self.session_alive(),
            "current_run_dir": str(self.current_run_dir) if self.current_run_dir else None,
            "defaults": self.default_config,
            "last_launch_config": self.last_launch_config,
            "available_libero_types": self.available_libero_types,
            "available_suites": AVAILABLE_SUITES,
            "tmux_tail": tmux_tail,
            "build_token": BUILD_TOKEN,
    }


def _derive_progress_state(state: dict, control: dict) -> dict:
    live_status = state.get("live_status") or {}
    tail = (control.get("tmux_tail") or "").lower()
    phase = str(live_status.get("phase") or "")

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
    return {"percent": 0, "label": "Idle"}


def _render_select_options(options: list[str], selected: str) -> str:
    html_parts = []
    for option in options:
        selected_attr = ' selected="selected"' if option == selected else ""
        html_parts.append(f'<option value="{option}"{selected_attr}>{option}</option>')
    return "".join(html_parts)


def _index_html(refresh_ms: int, build_token: str, available_libero_types: list[str]) -> str:
    default_type_options = _render_select_options(available_libero_types, "plus")
    default_suite_options = _render_select_options(AVAILABLE_SUITES, "libero_goal")
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>LIBERO Debug Control Panel</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #0d1117;
      --panel: #161b22;
      --text: #e6edf3;
      --muted: #8b949e;
      --accent: #58a6ff;
      --border: #30363d;
      --ok: #3fb950;
      --warn: #d29922;
    }}
    body {{
      margin: 0;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      background: var(--bg);
      color: var(--text);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(420px, 1.15fr) minmax(380px, 1fr);
      gap: 16px;
      padding: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
    }}
    h1, h2 {{
      margin: 0 0 10px 0;
      font-size: 16px;
    }}
    .muted {{
      color: var(--muted);
      font-size: 12px;
    }}
    .frontend-error {{
      margin-top: 8px;
      color: #ffb4b4;
      font-size: 12px;
      white-space: pre-wrap;
    }}
    .links {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 8px;
    }}
    .links a {{
      color: var(--accent);
      font-size: 12px;
      text-decoration: none;
    }}
    .links a:hover {{
      text-decoration: underline;
    }}
    .progress-block {{
      margin-top: 10px;
    }}
    .catalog-block {{
      margin-top: 10px;
      display: grid;
      gap: 8px;
    }}
    .catalog-card {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px 10px;
      background: #0b1220;
    }}
    .catalog-title {{
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 4px;
    }}
    .catalog-value {{
      font-size: 12px;
      line-height: 1.45;
      color: var(--text);
      white-space: pre-wrap;
    }}
    .progress-label {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .progress-track {{
      width: 100%;
      height: 10px;
      border-radius: 999px;
      background: #0b1220;
      border: 1px solid var(--border);
      overflow: hidden;
    }}
    .progress-fill {{
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, #3fb950, #58a6ff);
      transition: width 180ms ease;
    }}
    .preview-controls {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 10px 0;
    }}
    .media-stage {{
      display: flex;
      justify-content: center;
      align-items: center;
      width: 100%;
      margin-bottom: 8px;
    }}
    .live-layout {{
      display: grid;
      grid-template-columns: minmax(220px, 280px) minmax(0, 1fr);
      grid-template-areas:
        "media status"
        "editor editor";
      gap: 14px;
      align-items: start;
    }}
    .live-video-wrap {{
      grid-area: media;
      display: grid;
      gap: 8px;
      justify-items: center;
    }}
    .status-wrap {{
      grid-area: status;
      display: grid;
      gap: 10px;
    }}
    .path {{
      word-break: break-all;
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    .status-ok {{
      color: var(--ok);
    }}
    .status-warn {{
      color: var(--warn);
    }}
    .form-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-bottom: 12px;
    }}
    label {{
      display: grid;
      gap: 4px;
      font-size: 12px;
      color: var(--muted);
    }}
    input, select, textarea {{
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #0b1220;
      color: var(--text);
      padding: 8px;
      font: inherit;
    }}
    textarea {{
      min-height: 64px;
      resize: vertical;
    }}
    .checks {{
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 12px;
    }}
    .checks label {{
      display: flex;
      align-items: center;
      gap: 6px;
      color: var(--text);
    }}
    .actions {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    button {{
      background: transparent;
      color: var(--accent);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 8px 10px;
      text-align: left;
      cursor: pointer;
      font: inherit;
    }}
    button:hover {{
      border-color: var(--accent);
    }}
    button.active {{
      border-color: var(--accent);
      color: var(--text);
      background: rgba(88, 166, 255, 0.12);
    }}
    button:disabled {{
      opacity: 0.45;
      cursor: not-allowed;
    }}
    img, video {{
      width: 100%;
      min-width: 0;
      max-width: 360px;
      max-height: 70vh;
      object-fit: contain;
      background: #000;
      border-radius: 8px;
    }}
    .live-frame {{
      max-height: 34vh;
    }}
    .live-video {{
      max-height: 34vh;
    }}
    .status-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin: 10px 0;
    }}
    .status-card {{
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 8px 10px;
      background: #0b1220;
    }}
    .status-card .label {{
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 4px;
    }}
    .status-card .value {{
      font-size: 13px;
      color: var(--text);
      word-break: break-word;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-size: 12px;
      line-height: 1.45;
      margin: 0;
      max-height: 46vh;
      overflow: auto;
    }}
    .list {{
      display: grid;
      gap: 8px;
      max-height: 38vh;
      overflow: auto;
    }}
    .action-log {{
      grid-area: editor;
      margin-top: 10px;
      padding: 10px;
      border: 1px solid var(--border);
      border-radius: 10px;
      background: #0b1220;
      max-height: 38vh;
      overflow-x: auto;
      overflow-y: auto;
    }}
    .action-table {{
      width: 100%;
      min-width: 980px;
      border-collapse: collapse;
      font-size: 12px;
      table-layout: fixed;
    }}
    .action-table th,
    .action-table td {{
      border-bottom: 1px solid rgba(255, 255, 255, 0.07);
      padding: 4px 6px;
      text-align: right;
      white-space: nowrap;
    }}
    .action-table th:first-child,
    .action-table td:first-child {{
      text-align: left;
    }}
    .action-table tr.current-row td {{
      background: rgba(88, 166, 255, 0.12);
      color: var(--text);
    }}
    .action-table tr.executed-row td:first-child {{
      color: var(--ok);
    }}
    .action-meta {{
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .action-toolbar {{
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin-bottom: 10px;
    }}
    .editor-hint {{
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 2px;
    }}
    .action-input {{
      width: 100%;
      min-width: 72px;
      padding: 4px 5px;
      text-align: right;
      border-radius: 6px;
      font-size: 11px;
      appearance: textfield;
      -moz-appearance: textfield;
    }}
    .action-input::-webkit-outer-spin-button,
    .action-input::-webkit-inner-spin-button {{
      -webkit-appearance: none;
      margin: 0;
    }}
    .action-input:disabled {{
      opacity: 0.65;
    }}
    details.raw-status {{
      margin-top: 10px;
    }}
    details.raw-status summary {{
      cursor: pointer;
      color: var(--accent);
      font-size: 12px;
      margin-bottom: 8px;
    }}
    .full {{
      grid-column: 1 / -1;
    }}
    @media (max-width: 900px) {{
      img, video {{
        width: 100%;
        min-width: 0;
        max-width: none;
      }}
      .live-layout {{
        grid-template-columns: 1fr;
        grid-template-areas:
          "media"
          "status"
          "editor";
      }}
      .status-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <section class="panel">
      <h1>Run Controls</h1>
      <div class="muted" id="sessionInfo"></div>
      <div class="muted" id="buildInfo">Build: {build_token}</div>
      <div class="path" id="runDir"></div>
      <div class="frontend-error" id="frontendError"></div>
      <form id="launchForm">
        <div class="form-grid">
          <label>Libero Type
            <select id="liberoType">{default_type_options}</select>
          </label>
          <label>Suite
            <select id="suiteName">{default_suite_options}</select>
          </label>
          <label>Task ID
            <input id="taskId" type="number" min="0" step="1" value="" placeholder="choose task id" />
          </label>
          <label>Trial Index
            <input id="trialIdx" type="number" min="0" step="1" value="0" />
          </label>
        </div>
        <div class="actions">
          <button type="button" id="setTaskBtn">Set Task</button>
        </div>
        <div class="catalog-block">
          <div class="catalog-card">
            <div class="catalog-title">Selected Task Info</div>
            <div class="catalog-value" id="selectedTaskInfo">Loading task metadata...</div>
          </div>
          <div class="catalog-card">
            <div class="catalog-title">Suite Task Counts</div>
            <div class="catalog-value" id="suiteCountsInfo">Loading suite counts...</div>
          </div>
        </div>
      </form>
      <div class="muted" id="launchStatus"></div>
      <div class="progress-block">
        <div class="progress-label" id="launchProgressLabel">Idle</div>
        <div class="progress-track">
          <div class="progress-fill" id="launchProgressFill"></div>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Live Chunk / Status</h2>
      <div class="live-layout">
        <div class="live-video-wrap">
          <div class="media-stage">
            <video id="liveVideo" class="live-video" controls autoplay muted loop playsinline style="display:none;"></video>
          </div>
          <div class="muted" id="liveVideoPath"></div>
          <div class="links">
            <a id="liveVideoLink" target="_blank" rel="noopener noreferrer">Open Live MP4</a>
          </div>
          <div class="action-toolbar">
            <button type="button" id="simulateBtn">Simulate Chunk</button>
            <button type="button" id="runChunkBtn">Run Chunk</button>
            <button type="button" id="resetEditsBtn">Reset Edits</button>
          </div>
        </div>
        <div class="status-wrap">
          <div class="status-grid" id="statusGrid"></div>
        </div>
        <div class="action-log">
          <div class="action-meta" id="actionMeta">No action plan yet</div>
          <div class="editor-hint" id="actionEditorHint">A fresh chunk will become editable here when the policy is ready.</div>
          <div id="actionTableWrap">No action plan yet</div>
        </div>
      </div>
      <details class="raw-status">
        <summary>Raw Status</summary>
        <pre id="liveStatus"></pre>
      </details>
    </section>

    <section class="panel">
      <h1>Completed Chunk Preview</h1>
      <div class="muted" id="selectedChunkLabel">No completed chunk yet</div>
      <div class="preview-controls">
        <button type="button" id="prevChunkBtn">Prev Saved</button>
        <button type="button" id="nextChunkBtn">Next Saved</button>
        <button type="button" id="latestChunkBtn">Latest Chunk</button>
        <button type="button" id="playSequenceBtn">Play 0 → Latest</button>
        <button type="button" id="stopSequenceBtn">Stop Sequence</button>
      </div>
      <div class="media-stage">
        <img id="gif" style="display:none;" />
        <video id="video" controls autoplay muted loop playsinline style="display:none;"></video>
      </div>
      <div class="muted" id="videoPath"></div>
      <div class="links">
        <a id="gifLink" target="_blank" rel="noopener noreferrer">Open GIF</a>
        <a id="mp4Link" target="_blank" rel="noopener noreferrer">Open MP4</a>
      </div>
    </section>

    <section class="panel">
      <h2>Chunk Media</h2>
      <div class="list" id="chunkList"></div>
    </section>

    <section class="panel">
      <h2>Worker Log</h2>
      <pre id="tmuxTail"></pre>
    </section>

    <section class="panel">
      <h2>Summary / Trace</h2>
      <pre id="summary"></pre>
      <hr style="border-color:var(--border);margin:12px 0;">
      <pre id="trace"></pre>
    </section>
  </div>
  <script>
    (function() {{
      function setText(id, text) {{
        var el = document.getElementById(id);
        if (el) el.textContent = text;
      }}
      function setHtml(id, html) {{
        var el = document.getElementById(id);
        if (el) el.innerHTML = html;
      }}
      function getJson(url, cb) {{
        var xhr = new XMLHttpRequest();
        xhr.open('GET', url, true);
        xhr.onreadystatechange = function() {{
          if (xhr.readyState !== 4) return;
          if (xhr.status < 200 || xhr.status >= 300) {{
            cb(new Error('HTTP ' + xhr.status));
            return;
          }}
          try {{
            cb(null, JSON.parse(xhr.responseText));
          }} catch (err) {{
            cb(err);
          }}
        }};
        xhr.send();
      }}
      function updateBasic(state) {{
        var control = state.control || {{}};
        var progress = state.progress || {{ percent: 0, label: 'Idle' }};
        setHtml(
          'sessionInfo',
          "Session: <span class='" + (control.session_alive ? "status-ok" : "status-warn") + "'>" +
          (control.session_name || "n/a") + "</span> | " +
          (control.session_alive ? "alive" : "not running")
        );
        setText('buildInfo', 'Build: ' + (control.build_token || '{build_token}'));
        setText('runDir', state.run_dir || control.current_run_dir || 'No active run directory yet');
        setText('launchProgressLabel', progress.label + ' (' + progress.percent + '%)');
        var fill = document.getElementById('launchProgressFill');
        if (fill) fill.style.width = String(progress.percent) + '%';
      }}
      function poll() {{
        getJson('/api/state?ts=' + Date.now(), function(err, state) {{
          if (err) {{
            setText('frontendError', 'Frontend polling error: ' + err.message);
            return;
          }}
          updateBasic(state);
        }});
      }}
      window.onerror = function(message, source, lineno, colno) {{
        setText('frontendError', 'Frontend JS error: ' + message + ' @ ' + source + ':' + lineno + ':' + colno);
      }};
      poll();
      setInterval(poll, {refresh_ms});
    }})();
  </script>
  <script>
    const refreshMs = {refresh_ms};
    const actionDimLabels = ["eef_x", "eef_y", "eef_z", "rot_x", "rot_y", "rot_z", "gripper"];
    let controlsInitialized = false;
    let currentChunkItems = [];
    let followLatestChunk = true;
    let selectedChunkIndex = null;
    let sequencePlaying = false;
    let sequenceQueue = [];
    let sequenceQueueIndex = -1;
    let previewVideoVersion = null;
    let previewGifVersion = null;
    let liveVideoVersion = null;
    let liveVideoPathValue = null;
    let editableActions = null;
    let editableSourceActions = null;
    let editableChunkKey = null;
    let lastRenderedPlan = null;
    let lastActionEditorRenderKey = null;
    let lastState = null;
    let taskCatalogCache = {{}};
    let taskCatalogPromiseCache = {{}};

    function artifactUrl(relPath) {{
      if (!relPath) return "";
      return "/artifacts/" + relPath.split("/").map(encodeURIComponent).join("/") + "?ts=" + Date.now();
    }}

    function prettyValue(value) {{
      if (value === undefined || value === null) return "-";
      if (typeof value === "number") return value.toFixed(4);
      if (Array.isArray(value)) return value.map((item) => prettyValue(item)).join(", ");
      return String(value);
    }}

    function actionLabelsForWidth(numDims, supplied) {{
      const base = (supplied && supplied.length) ? supplied : actionDimLabels;
      const labels = [];
      for (let idx = 0; idx < numDims; idx += 1) {{
        labels.push(base[idx] || ("a" + idx));
      }}
      return labels;
    }}

    function deepCopyActions(actions) {{
      if (!actions) return null;
      return actions.map((row) => row.map((value) => Number(value)));
    }}

    async function getTaskCatalog(liberoType) {{
      if (taskCatalogCache[liberoType]) {{
        return taskCatalogCache[liberoType];
      }}
      if (taskCatalogPromiseCache[liberoType]) {{
        return taskCatalogPromiseCache[liberoType];
      }}
      taskCatalogPromiseCache[liberoType] = fetch(
        "/api/task_catalog?libero_type=" + encodeURIComponent(liberoType)
      ).then(async function(resp) {{
        const data = await resp.json();
        if (!resp.ok) {{
          throw new Error(data.error || ("task catalog failed: " + resp.status));
        }}
        taskCatalogCache[liberoType] = data;
        delete taskCatalogPromiseCache[liberoType];
        return data;
      }}).catch(function(err) {{
        delete taskCatalogPromiseCache[liberoType];
        throw err;
      }});
      return taskCatalogPromiseCache[liberoType];
    }}

    function phaseAllowsSimulation(phase) {{
      return phase === "awaiting_chunk_execution";
    }}

    function phaseAllowsRun(phase) {{
      return phase === "awaiting_chunk_execution";
    }}

    function phaseAllowsSetTask(phase) {{
      return (
        phase === "awaiting_chunk_execution" ||
        phase === "chunk_complete" ||
        phase === "finished"
      );
    }}

    function extractPlan(state) {{
      const liveStatus = state.live_status || {{}};
      const traceTail = state.trace_tail || [];
      let planActions = liveStatus.current_plan_actions || liveStatus.executed_actions || null;
      let predictedActions = liveStatus.predicted_actions || null;
      let labels = liveStatus.action_dim_labels || actionDimLabels;
      let chunkIdx = liveStatus.chunk_idx;
      let planSource = "live status";

      if (!planActions) {{
        for (let idx = traceTail.length - 1; idx >= 0; idx -= 1) {{
          const record = traceTail[idx];
          if (record.event === "plan") {{
            planActions = record.executed_actions || record.predicted_actions || null;
            predictedActions = record.predicted_actions || null;
            chunkIdx = record.chunk_idx;
            planSource = "trace";
            break;
          }}
        }}
      }}

      if (!planActions || !planActions.length) {{
        return null;
      }}

      return {{
        actions: planActions,
        predictedActions: predictedActions,
        labels: actionLabelsForWidth(planActions[0].length, labels),
        chunkIdx: chunkIdx,
        planSource: planSource,
        lastStep: liveStatus.chunk_step,
        editHistory: liveStatus.edit_history || [],
        phase: String(liveStatus.phase || ""),
      }};
    }}

    function buildPlanKey(state, plan) {{
      const liveStatus = state.live_status || {{}};
      return [
        String(liveStatus.libero_type || ""),
        String(liveStatus.suite_name || ""),
        String(liveStatus.task_id || ""),
        String(liveStatus.trial_idx || ""),
        String(plan.chunkIdx === undefined || plan.chunkIdx === null ? "" : plan.chunkIdx),
        JSON.stringify(plan.actions),
      ].join("|");
    }}

    function syncEditableActions(state, plan) {{
      if (!plan) {{
        editableActions = null;
        editableSourceActions = null;
        editableChunkKey = null;
        lastRenderedPlan = null;
        lastActionEditorRenderKey = null;
        return;
      }}

      const nextKey = buildPlanKey(state, plan);
      if (nextKey !== editableChunkKey) {{
        editableChunkKey = nextKey;
        editableSourceActions = deepCopyActions(plan.actions);
        editableActions = deepCopyActions(plan.actions);
      }}
      lastRenderedPlan = plan;
    }}

    function buildStatusCards(state) {{
      const liveStatus = state.live_status || {{}};
      const summary = state.summary || {{}};
      const cards = [
        ["Phase", liveStatus.phase || summary.stop_reason || "-"],
        ["Libero Type", liveStatus.libero_type || summary.libero_type || "-"],
        ["Suite", liveStatus.suite_name || summary.suite_name || "-"],
        ["Task ID", liveStatus.task_id === undefined ? (summary.task_id === undefined ? "-" : summary.task_id) : liveStatus.task_id],
        ["Trial", liveStatus.trial_idx === undefined ? (summary.trial_idx === undefined ? "-" : summary.trial_idx) : liveStatus.trial_idx],
        ["Chunk", liveStatus.chunk_idx === undefined ? "-" : liveStatus.chunk_idx],
        ["Chunk Step", liveStatus.chunk_step === undefined ? "-" : liveStatus.chunk_step],
        ["Env Step", liveStatus.env_step === undefined ? "-" : liveStatus.env_step],
        ["Task", liveStatus.task_name || summary.task_name || "-"],
        ["Taxonomy", liveStatus.taxonomy || summary.taxonomy || "-"],
        ["Done", liveStatus.done === undefined ? (summary.success === undefined ? "-" : String(summary.success)) : String(liveStatus.done)],
        ["Last Action", liveStatus.last_action ? liveStatus.last_action.map((value) => Number(value).toFixed(6)).join(", ") : "-"],
        ["Model", liveStatus.model_reused ? "reused" : "loaded"],
      ];
      return cards
        .map(function(pair) {{
          return '<div class="status-card"><div class="label">' + pair[0] + '</div><div class="value">' + prettyValue(pair[1]) + '</div></div>';
        }})
        .join("");
    }}

    async function updateRunControlMeta() {{
      const selectedTaskInfoEl = document.getElementById("selectedTaskInfo");
      const suiteCountsInfoEl = document.getElementById("suiteCountsInfo");
      const liberoType = document.getElementById("liberoType").value;
      const suiteName = document.getElementById("suiteName").value;
      const rawTaskId = document.getElementById("taskId").value;
      const taskId = Number(rawTaskId);

      selectedTaskInfoEl.textContent = "Loading task metadata...";
      suiteCountsInfoEl.textContent = "Loading suite counts...";

      try {{
        const catalog = await getTaskCatalog(liberoType);
        const suiteCounts = catalog.suite_task_counts || {{}};
        const suiteTaxonomies = catalog.task_taxonomy_by_suite || {{}};
        const countLines = [];
        for (let idx = 0; idx < (lastState && lastState.control && lastState.control.available_suites ? lastState.control.available_suites.length : 0); idx += 1) {{
          const suite = lastState.control.available_suites[idx];
          const count = suiteCounts[suite];
          if (count === undefined || count === null) {{
            countLines.push(suite + ": unavailable");
          }} else {{
            countLines.push(suite + ": " + count + " tasks");
          }}
        }}
        if (!countLines.length) {{
          for (const suite in suiteCounts) {{
            const count = suiteCounts[suite];
            countLines.push(suite + ": " + count + " tasks");
          }}
        }}
        suiteCountsInfoEl.textContent = countLines.join("\\n");

        const suiteOptions = document.getElementById("suiteName").options;
        for (let idx = 0; idx < suiteOptions.length; idx += 1) {{
          const option = suiteOptions[idx];
          const baseValue = option.value;
          const count = suiteCounts[baseValue];
          option.textContent = (count === undefined || count === null)
            ? baseValue
            : (baseValue + " (" + count + ")");
        }}

        const suiteCount = suiteCounts[suiteName];
        if (!Number.isFinite(taskId)) {{
          selectedTaskInfoEl.textContent = "Enter a valid integer Task ID.";
          return;
        }}
        if (suiteCount === undefined || suiteCount === null) {{
          selectedTaskInfoEl.textContent = "Task count is unavailable for " + suiteName + ".";
          return;
        }}
        if (taskId < 0 || taskId >= suiteCount) {{
          selectedTaskInfoEl.textContent =
            "Task ID " + taskId + " is out of range for " + suiteName + ". Valid range: 0.." + String(suiteCount - 1);
          return;
        }}

        const infoLines = [
          "Suite: " + suiteName,
          "Task Count: " + suiteCount,
          "Task ID: " + taskId,
        ];
        if (liberoType === "plus") {{
          const taxonomy = (suiteTaxonomies[suiteName] || {{}})[String(taskId)] || "Unknown";
          infoLines.push("Taxonomy: " + taxonomy);
        }} else {{
          infoLines.push("Taxonomy: n/a for " + liberoType);
        }}
        selectedTaskInfoEl.textContent = infoLines.join("\\n");
      }} catch (err) {{
        selectedTaskInfoEl.textContent = "Task metadata error: " + err.message;
        suiteCountsInfoEl.textContent = "Suite counts error: " + err.message;
      }}
    }}

    function renderActionEditor(state) {{
      const liveStatus = state.live_status || {{}};
      const phase = String(liveStatus.phase || "");
      const metaEl = document.getElementById("actionMeta");
      const hintEl = document.getElementById("actionEditorHint");
      const wrapEl = document.getElementById("actionTableWrap");
      const plan = extractPlan(state);

      syncEditableActions(state, plan);
      updateControlButtons(phase, !!plan);

      if (!plan || !editableActions) {{
        metaEl.textContent = "No action plan available yet";
        hintEl.textContent = "A fresh chunk will become editable here when the policy is ready.";
        wrapEl.innerHTML = "No action plan available yet";
        lastActionEditorRenderKey = null;
        return;
      }}

      const editable = phaseAllowsSimulation(phase);
      const renderKey = editableChunkKey + "|editable=" + String(editable);
      const header = ['<tr><th>Step</th>']
        .concat(plan.labels.map(function(label) {{ return "<th>" + label + "</th>"; }}))
        .concat(["</tr>"])
        .join("");

      const rows = [];
      for (let stepIdx = 0; stepIdx < editableActions.length; stepIdx += 1) {{
        const action = editableActions[stepIdx];
        const classes = [];
        if (typeof plan.lastStep === "number" && stepIdx === plan.lastStep) {{
          classes.push("current-row");
        }}
        if (typeof plan.lastStep === "number" && stepIdx <= plan.lastStep) {{
          classes.push("executed-row");
        }}
        const cells = ['<td>[' + String(stepIdx).padStart(2, "0") + ']</td>'];
        for (let dimIdx = 0; dimIdx < action.length; dimIdx += 1) {{
          const value = Number(action[dimIdx]);
          const disabledAttr = editable ? "" : ' disabled="disabled"';
          cells.push(
            '<td><input class="action-input" type="text" inputmode="decimal" spellcheck="false" data-step="' +
            stepIdx +
            '" data-dim="' +
            dimIdx +
            '" value="' +
            value.toFixed(6) +
            '"' +
            disabledAttr +
            " /></td>"
          );
        }}
        rows.push(
          "<tr" + (classes.length ? ' class=\"' + classes.join(" ") + '\"' : "") + ">" + cells.join("") + "</tr>"
        );
      }}

      const metaParts = [];
      metaParts.push(
        "chunk=" + String(plan.chunkIdx === undefined || plan.chunkIdx === null ? "?" : plan.chunkIdx).padStart(3, "0")
      );
      metaParts.push("source=" + plan.planSource);
      if (typeof plan.lastStep === "number") {{
        metaParts.push("current_step=" + String(plan.lastStep).padStart(2, "0"));
      }}
      if (plan.editHistory.length) {{
        metaParts.push("backend_edits=" + plan.editHistory.join(" | "));
      }}
      if (plan.predictedActions && plan.predictedActions.length) {{
        metaParts.push(
          "pred0=" + plan.predictedActions[0].map(function(value) {{ return Number(value).toFixed(6); }}).join(", ")
        );
      }}
      metaEl.textContent = metaParts.join(" • ");

      if (editable) {{
        hintEl.textContent = "Edit the chunk values below. Simulate Chunk is a dry-run on the same scene, while Run Chunk advances the real environment and then requests the next policy chunk.";
      }} else if (phase === "simulating_chunk") {{
        hintEl.textContent = "Dry-run simulation is running from the current policy input scene. The real environment is not advancing.";
      }} else if (phase === "planning_next_chunk") {{
        hintEl.textContent = "The real environment advanced. A fresh policy chunk is being generated for the next scene.";
      }} else if (phase === "executing_chunk") {{
        hintEl.textContent = "Run Chunk is executing on the real environment. The live video updates as the robot moves.";
      }} else if (phase === "finished") {{
        hintEl.textContent = "This task finished. Use Set Task to move to another task without reloading the model.";
      }} else {{
        hintEl.textContent = "The editor becomes active when a new chunk is waiting for simulation.";
      }}

      const activeEl = document.activeElement;
      const isEditingActionInput =
        !!activeEl &&
        activeEl.classList &&
        activeEl.classList.contains("action-input") &&
        wrapEl.contains(activeEl);
      if (renderKey === lastActionEditorRenderKey && wrapEl.querySelector(".action-table")) {{
        if (isEditingActionInput) {{
          return;
        }}
        return;
      }}

      wrapEl.innerHTML = '<table class="action-table"><thead>' + header + "</thead><tbody>" + rows.join("") + "</tbody></table>";
      lastActionEditorRenderKey = renderKey;

      const inputEls = wrapEl.querySelectorAll(".action-input");
      for (let idx = 0; idx < inputEls.length; idx += 1) {{
        inputEls[idx].addEventListener("input", function(event) {{
          const input = event.target;
          const stepIdx = Number(input.getAttribute("data-step"));
          const dimIdx = Number(input.getAttribute("data-dim"));
          const numericValue = Number(input.value);
          if (Number.isFinite(numericValue)) {{
            editableActions[stepIdx][dimIdx] = numericValue;
          }}
        }});
        inputEls[idx].addEventListener("change", function(event) {{
          const input = event.target;
          const stepIdx = Number(input.getAttribute("data-step"));
          const dimIdx = Number(input.getAttribute("data-dim"));
          const numericValue = Number(input.value);
          if (!Number.isFinite(numericValue)) {{
            input.value = editableActions[stepIdx][dimIdx].toFixed(6);
            return;
          }}
          editableActions[stepIdx][dimIdx] = numericValue;
          input.value = numericValue.toFixed(6);
        }});
      }}
    }}

    function updateControlButtons(phase, hasPlan) {{
      const simulateBtn = document.getElementById("simulateBtn");
      const runChunkBtn = document.getElementById("runChunkBtn");
      const resetBtn = document.getElementById("resetEditsBtn");
      const setTaskBtn = document.getElementById("setTaskBtn");
      const sessionAlive = !!(lastState && lastState.control && lastState.control.session_alive);

      simulateBtn.disabled = !sessionAlive || !hasPlan || !phaseAllowsSimulation(phase);
      runChunkBtn.disabled = !sessionAlive || !hasPlan || !phaseAllowsRun(phase);
      resetBtn.disabled = !sessionAlive || !hasPlan || !phaseAllowsSimulation(phase);
      setTaskBtn.disabled = !sessionAlive || !phaseAllowsSetTask(phase);
    }}

    function readEditableActions() {{
      if (!editableActions) return null;
      const nextActions = deepCopyActions(editableActions);
      const inputEls = document.querySelectorAll("#actionTableWrap .action-input");
      for (let idx = 0; idx < inputEls.length; idx += 1) {{
        const input = inputEls[idx];
        const stepIdx = Number(input.getAttribute("data-step"));
        const dimIdx = Number(input.getAttribute("data-dim"));
        const numericValue = Number(input.value);
        if (!Number.isFinite(numericValue)) {{
          throw new Error("One or more action cells contain invalid numbers.");
        }}
        nextActions[stepIdx][dimIdx] = numericValue;
      }}
      editableActions = deepCopyActions(nextActions);
      return nextActions;
    }}

    function parseChunkNumber(relPath, fallbackIndex) {{
      const match = /chunk(\\d+)/i.exec(relPath || "");
      if (match) return Number(match[1]);
      return fallbackIndex;
    }}

    function buildChunkItems(state) {{
      const liveStatus = state.live_status || {{}};
      const summary = state.summary || {{}};
      const suiteName = liveStatus.suite_name || summary.suite_name || "";
      const taskId = liveStatus.task_id === undefined ? summary.task_id : liveStatus.task_id;
      const taskPrefix = suiteName && taskId !== undefined && taskId !== null
        ? suiteName + "_task" + String(taskId).padStart(4, "0") + "_"
        : "";
      const gifList = (state.chunk_gifs || []).filter(function(path) {{
        return !taskPrefix || path.indexOf(taskPrefix) !== -1;
      }});
      const videoList = (state.chunk_videos || []).filter(function(path) {{
        return !taskPrefix || path.indexOf(taskPrefix) !== -1;
      }});
      const total = Math.max(gifList.length, videoList.length);
      const items = [];
      for (let idx = 0; idx < total; idx += 1) {{
        const gif = gifList[idx] || null;
        const video = videoList[idx] || null;
        const anyPath = gif || video || "";
        items.push({{
          index: idx,
          chunkNumber: parseChunkNumber(anyPath, idx),
          gif: gif,
          video: video,
          label: "Chunk " + String(parseChunkNumber(anyPath, idx)).padStart(3, "0"),
        }});
      }}
      return items;
    }}

    function updateChunkLinks(item) {{
      document.getElementById("gifLink").href = item && item.gif ? artifactUrl(item.gif) : "#";
      document.getElementById("mp4Link").href = item && item.video ? artifactUrl(item.video) : "#";
    }}

    function renderChunkPreview(item, options) {{
      const autoplayVideo = options && options.autoplayVideo;
      const gifEl = document.getElementById("gif");
      const videoEl = document.getElementById("video");
      const selectedLabel = document.getElementById("selectedChunkLabel");
      const pathLabel = document.getElementById("videoPath");

      if (!item) {{
        selectedLabel.textContent = "No completed chunk yet";
        pathLabel.textContent = "No chunk media yet";
        gifEl.style.display = "none";
        videoEl.style.display = "none";
        previewVideoVersion = null;
        previewGifVersion = null;
        updateChunkLinks(null);
        return;
      }}

      selectedLabel.textContent = sequencePlaying ? (item.label + " • sequence playback") : (item.label + " • selected");
      pathLabel.textContent = item.video || item.gif || "No chunk media yet";
      updateChunkLinks(item);

      if (autoplayVideo && item.video) {{
        previewVideoVersion = item.video;
        videoEl.loop = false;
        videoEl.style.display = "block";
        gifEl.style.display = "none";
        videoEl.src = artifactUrl(item.video);
        videoEl.load();
        videoEl.play().catch(function() {{}});
        return;
      }}

      if (item.gif) {{
        previewGifVersion = item.gif;
        gifEl.src = artifactUrl(item.gif);
        gifEl.style.display = "block";
        videoEl.style.display = "none";
        return;
      }}

      if (item.video) {{
        previewVideoVersion = item.video;
        videoEl.loop = true;
        videoEl.style.display = "block";
        gifEl.style.display = "none";
        videoEl.src = artifactUrl(item.video);
        videoEl.load();
        videoEl.play().catch(function() {{}});
        return;
      }}

      gifEl.style.display = "none";
      videoEl.style.display = "none";
    }}

    function renderChunkList() {{
      const listEl = document.getElementById("chunkList");
      listEl.innerHTML = "";
      for (let idx = currentChunkItems.length - 1; idx >= 0; idx -= 1) {{
        const item = currentChunkItems[idx];
        const button = document.createElement("button");
        button.textContent = item.label + (item.video ? " • mp4" : "") + (item.gif ? " • gif" : "");
        if (item.index === selectedChunkIndex) {{
          button.classList.add("active");
        }}
        button.onclick = function() {{
          followLatestChunk = false;
          sequencePlaying = false;
          selectedChunkIndex = item.index;
          renderChunkPreview(item);
          renderChunkList();
        }};
        listEl.appendChild(button);
      }}
    }}

    function syncSelectedChunk() {{
      if (!currentChunkItems.length) {{
        selectedChunkIndex = null;
        renderChunkPreview(null);
        renderChunkList();
        return;
      }}
      if (followLatestChunk || selectedChunkIndex === null || selectedChunkIndex >= currentChunkItems.length) {{
        selectedChunkIndex = currentChunkItems.length - 1;
      }}
      if (!sequencePlaying) {{
        renderChunkPreview(currentChunkItems[selectedChunkIndex]);
      }}
      renderChunkList();
    }}

    function playSequenceAt(queueIndex) {{
      if (!sequencePlaying || queueIndex < 0 || queueIndex >= sequenceQueue.length) {{
        sequencePlaying = false;
        sequenceQueue = [];
        sequenceQueueIndex = -1;
        renderChunkPreview(selectedChunkIndex !== null ? currentChunkItems[selectedChunkIndex] : null);
        renderChunkList();
        return;
      }}

      sequenceQueueIndex = queueIndex;
      selectedChunkIndex = sequenceQueue[queueIndex];
      renderChunkPreview(currentChunkItems[selectedChunkIndex], {{ autoplayVideo: true }});
      renderChunkList();
    }}

    function startSequencePlayback() {{
      const playable = currentChunkItems.filter(function(item) {{ return !!item.video; }}).map(function(item) {{ return item.index; }});
      if (!playable.length) {{
        document.getElementById("launchStatus").textContent = "No stable chunk MP4s available yet.";
        return;
      }}
      followLatestChunk = false;
      sequencePlaying = true;
      sequenceQueue = playable;
      playSequenceAt(0);
    }}

    function stopSequencePlayback() {{
      sequencePlaying = false;
      sequenceQueue = [];
      sequenceQueueIndex = -1;
      renderChunkPreview(selectedChunkIndex !== null ? currentChunkItems[selectedChunkIndex] : null);
      renderChunkList();
    }}

    function setOptions(selectEl, values) {{
      if (selectEl.options.length > 0) return;
      for (let idx = 0; idx < values.length; idx += 1) {{
        const option = document.createElement("option");
        option.value = values[idx];
        option.textContent = values[idx];
        selectEl.appendChild(option);
      }}
    }}

    function populateControls(control) {{
      setOptions(document.getElementById("liberoType"), control.available_libero_types || []);
      setOptions(document.getElementById("suiteName"), control.available_suites || []);
      const cfg = control.last_launch_config || control.defaults || {{}};
      document.getElementById("liberoType").value = cfg.libero_type || "plus";
      document.getElementById("suiteName").value = cfg.task_suite_name || "libero_goal";
      document.getElementById("trialIdx").value = cfg.trial_idx === undefined || cfg.trial_idx === null ? 0 : cfg.trial_idx;
      controlsInitialized = true;
    }}

    function updateLiveVideo(state) {{
      const videoEl = document.getElementById("liveVideo");
      const pathEl = document.getElementById("liveVideoPath");
      const linkEl = document.getElementById("liveVideoLink");
      const liveVideo = state.live_chunk_video || null;
      const version = state.live_chunk_video_mtime_ns || null;

      pathEl.textContent = liveVideo || "No live chunk video yet";
      linkEl.href = liveVideo ? artifactUrl(liveVideo) : "#";

      if (!liveVideo) {{
        videoEl.style.display = "none";
        liveVideoVersion = null;
        liveVideoPathValue = null;
        return;
      }}

      videoEl.style.display = "block";
      videoEl.loop = true;
      if (liveVideo !== liveVideoPathValue || version !== liveVideoVersion) {{
        liveVideoPathValue = liveVideo;
        liveVideoVersion = version;
        videoEl.src = artifactUrl(liveVideo);
        videoEl.load();
        videoEl.play().catch(function() {{}});
      }}
    }}

    async function postJson(path, payload) {{
      const resp = await fetch(path, {{
        method: "POST",
        headers: {{ "Content-Type": "application/json" }},
        body: JSON.stringify(payload || {{}}),
      }});
      const data = await resp.json();
      if (!resp.ok) {{
        throw new Error(data.error || ("request failed: " + resp.status));
      }}
      return data;
    }}

    async function refresh() {{
      const resp = await fetch("/api/state?ts=" + Date.now());
      const state = await resp.json();
      const control = state.control || {{}};
      const phase = String((state.live_status || {{}}).phase || "");
      lastState = state;

      if (!controlsInitialized) {{
        populateControls(control);
      }}

      document.getElementById("sessionInfo").innerHTML =
        "Session: <span class='" + (control.session_alive ? "status-ok" : "status-warn") + "'>" +
        (control.session_name || "n/a") + "</span> | " +
        (control.session_alive ? "alive" : "not running");
      document.getElementById("buildInfo").textContent = "Build: " + (control.build_token || "{build_token}");
      const progress = state.progress || {{ percent: 0, label: "Idle" }};
      document.getElementById("launchProgressLabel").textContent = progress.label + " (" + progress.percent + "%)";
      document.getElementById("launchProgressFill").style.width = String(progress.percent) + "%";
      document.getElementById("runDir").textContent = state.run_dir || control.current_run_dir || "No active run directory yet";
      document.getElementById("summary").textContent = JSON.stringify(state.summary, null, 2);
      document.getElementById("trace").textContent = JSON.stringify(state.trace_tail, null, 2);
      document.getElementById("liveStatus").textContent = JSON.stringify(state.live_status, null, 2);
      document.getElementById("statusGrid").innerHTML = buildStatusCards(state);
      document.getElementById("tmuxTail").textContent = control.tmux_tail || "";

      renderActionEditor(state);
      updateLiveVideo(state);
      updateRunControlMeta().catch(function(err) {{
        document.getElementById("selectedTaskInfo").textContent = "Task metadata error: " + err.message;
      }});

      currentChunkItems = buildChunkItems(state);
      syncSelectedChunk();
      updateControlButtons(phase, !!extractPlan(state));
    }}

    document.getElementById("launchForm").addEventListener("submit", function(event) {{
      event.preventDefault();
    }});

    document.getElementById("setTaskBtn").addEventListener("click", async function() {{
      const statusEl = document.getElementById("launchStatus");
      const liberoType = document.getElementById("liberoType").value;
      const suiteName = document.getElementById("suiteName").value;
      const taskId = Number(document.getElementById("taskId").value);
      const trialIdx = Number(document.getElementById("trialIdx").value);
      statusEl.textContent = "Updating task selection...";
      try {{
        await postJson("/api/set_task", {{
          libero_type: liberoType,
          suite_name: suiteName,
          task_id: taskId,
          trial_idx: trialIdx,
        }});
        followLatestChunk = true;
        selectedChunkIndex = null;
        sequencePlaying = false;
        sequenceQueue = [];
        sequenceQueueIndex = -1;
        editableActions = null;
        editableSourceActions = null;
        editableChunkKey = null;
        statusEl.textContent = "Task update sent. Waiting for the selected scene...";
      }} catch (err) {{
        statusEl.textContent = "Set Task failed: " + err.message;
      }}
    }});

    document.getElementById("simulateBtn").addEventListener("click", async function() {{
      const statusEl = document.getElementById("launchStatus");
      try {{
        const actions = readEditableActions();
        const chunkIdx = lastRenderedPlan ? lastRenderedPlan.chunkIdx : null;
        await postJson("/api/simulate", {{
          actions: actions,
          chunk_idx: chunkIdx,
        }});
        statusEl.textContent = "Dry-run simulation started for the edited chunk.";
      }} catch (err) {{
        statusEl.textContent = "Simulation failed: " + err.message;
      }}
    }});

    document.getElementById("runChunkBtn").addEventListener("click", async function() {{
      const statusEl = document.getElementById("launchStatus");
      try {{
        const actions = readEditableActions();
        const chunkIdx = lastRenderedPlan ? lastRenderedPlan.chunkIdx : null;
        await postJson("/api/run_chunk", {{
          actions: actions,
          chunk_idx: chunkIdx,
        }});
        statusEl.textContent = "Run Chunk started on the real environment.";
      }} catch (err) {{
        statusEl.textContent = "Run Chunk failed: " + err.message;
      }}
    }});

    document.getElementById("resetEditsBtn").addEventListener("click", function() {{
      const statusEl = document.getElementById("launchStatus");
      if (!editableSourceActions || !lastState) {{
        statusEl.textContent = "There is no editable chunk to reset yet.";
        return;
      }}
      editableActions = deepCopyActions(editableSourceActions);
      lastActionEditorRenderKey = null;
      renderActionEditor(lastState);
      statusEl.textContent = "Action editor reset to the current policy chunk.";
    }});

    document.getElementById("liberoType").addEventListener("change", function() {{
      updateRunControlMeta().catch(function(err) {{
        document.getElementById("selectedTaskInfo").textContent = "Task metadata error: " + err.message;
      }});
    }});

    document.getElementById("suiteName").addEventListener("change", function() {{
      updateRunControlMeta().catch(function(err) {{
        document.getElementById("selectedTaskInfo").textContent = "Task metadata error: " + err.message;
      }});
    }});

    document.getElementById("taskId").addEventListener("input", function() {{
      updateRunControlMeta().catch(function(err) {{
        document.getElementById("selectedTaskInfo").textContent = "Task metadata error: " + err.message;
      }});
    }});

    document.getElementById("prevChunkBtn").addEventListener("click", function() {{
      if (!currentChunkItems.length) return;
      followLatestChunk = false;
      sequencePlaying = false;
      selectedChunkIndex = selectedChunkIndex === null ? 0 : Math.max(0, selectedChunkIndex - 1);
      renderChunkPreview(currentChunkItems[selectedChunkIndex]);
      renderChunkList();
    }});

    document.getElementById("nextChunkBtn").addEventListener("click", function() {{
      if (!currentChunkItems.length) return;
      followLatestChunk = false;
      sequencePlaying = false;
      selectedChunkIndex = selectedChunkIndex === null
        ? currentChunkItems.length - 1
        : Math.min(currentChunkItems.length - 1, selectedChunkIndex + 1);
      renderChunkPreview(currentChunkItems[selectedChunkIndex]);
      renderChunkList();
    }});

    document.getElementById("latestChunkBtn").addEventListener("click", function() {{
      if (!currentChunkItems.length) return;
      followLatestChunk = true;
      sequencePlaying = false;
      selectedChunkIndex = currentChunkItems.length - 1;
      renderChunkPreview(currentChunkItems[selectedChunkIndex]);
      renderChunkList();
    }});

    document.getElementById("playSequenceBtn").addEventListener("click", function() {{
      startSequencePlayback();
    }});

    document.getElementById("stopSequenceBtn").addEventListener("click", function() {{
      stopSequencePlayback();
    }});

    document.getElementById("video").addEventListener("ended", function() {{
      if (!sequencePlaying) return;
      playSequenceAt(sequenceQueueIndex + 1);
    }});

    refresh();
    setInterval(refresh, refreshMs);
  </script>
</body>
</html>
"""


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
                _index_html(
                    self.server.refresh_ms,
                    BUILD_TOKEN,
                    self._controller().available_libero_types,
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
                requested_type = str(payload.get("libero_type") or self._controller().current_libero_type())
                current_type = self._controller().current_libero_type()
                if requested_type == current_type and phase not in {"awaiting_chunk_execution", "chunk_complete", "finished"}:
                    raise RuntimeError(f"Set Task is only available when paused at a prompt. current_phase={phase!r}")
                result = self._controller().set_task(
                    libero_type=requested_type,
                    suite_name=str(payload.get("suite_name")),
                    task_id=int(payload.get("task_id")),
                    trial_idx=int(payload.get("trial_idx", 0)),
                )
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
        default="/workspace/RLinf_deprecated/.venv-openpi-liberoplus/bin/python",
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
    parser.add_argument("--default_task_id", type=int, default=1716)
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

#!/usr/bin/env python3

import datetime as dt
import json
import os
import pathlib
import posixpath
import re
import shlex
import subprocess
from typing import Any
from urllib.parse import unquote

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
        "live_observation_image": None,
        "live_observation_image_mtime_ns": None,
        "live_observation_wrist_image": None,
        "live_observation_wrist_image_mtime_ns": None,
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
    live_observation_image_path = live_dir / "observation_image.jpg"
    live_observation_wrist_image_path = live_dir / "observation_wrist_image.jpg"
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
        "live_observation_image": (
            str(live_observation_image_path.relative_to(run_dir))
            if live_observation_image_path.exists()
            else None
        ),
        "live_observation_image_mtime_ns": (
            live_observation_image_path.stat().st_mtime_ns
            if live_observation_image_path.exists()
            else None
        ),
        "live_observation_wrist_image": (
            str(live_observation_wrist_image_path.relative_to(run_dir))
            if live_observation_wrist_image_path.exists()
            else None
        ),
        "live_observation_wrist_image_mtime_ns": (
            live_observation_wrist_image_path.stat().st_mtime_ns
            if live_observation_wrist_image_path.exists()
            else None
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


def _normalize_model_path(value: str | pathlib.Path) -> pathlib.Path:
    path = pathlib.Path(str(value)).expanduser().resolve()
    if not path.exists():
        raise ValueError(f"Model checkpoint path does not exist: {path}")
    return path


def _normalize_optional_task_id(value) -> int | None:
    if value in (None, "", "null"):
        return None
    return int(value)


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
        self.openpi_venv_python = (
            openpi_venv_python if openpi_venv_python.is_absolute() else openpi_venv_python.resolve()
        )
        self.model_path = _normalize_model_path(model_path)
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
        last_launch_config = payload.get("last_launch_config")
        if isinstance(last_launch_config, dict):
            restored = last_launch_config.copy()
            restored["task_id"] = self.default_config.get("task_id")
            restored["trial_idx"] = self.default_config.get("trial_idx")
            restored["exp_name"] = ""
            self.last_launch_config.update(restored)

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

    def current_model_path(self) -> str:
        remembered_path = self.last_launch_config.get("model_path")
        if remembered_path:
            return str(remembered_path)
        return str(self.model_path)

    def ensure_session_started(self) -> dict | None:
        if self.session_alive():
            return None
        return None

    def reset_session(self) -> dict:
        subprocess.run(
            ["tmux", "kill-session", "-t", self.session_name],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        self.current_run_dir = None
        reset_config = self.default_config.copy()
        reset_config["model_path"] = self.current_model_path()
        reset_config["exp_name"] = ""
        self.last_launch_config = reset_config
        self._save_state_file()
        return {
            "ok": True,
            "session_name": self.session_name,
            "current_run_dir": None,
            "last_launch_config": self.last_launch_config,
        }

    def task_catalog(self, libero_type: str) -> dict:
        normalized_type = str(libero_type or self.current_libero_type())
        if normalized_type not in self.available_libero_types:
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

    def set_task(
        self,
        *,
        libero_type: str,
        suite_name: str,
        task_id: int,
        trial_idx: int,
        model_path: str,
    ) -> dict:
        target_type = str(libero_type or self.current_libero_type())
        if target_type not in self.available_libero_types:
            raise ValueError(f"Unsupported libero_type={target_type}")
        requested_model_path = str(_normalize_model_path(model_path))
        if (
            (not self.session_alive())
            or target_type != self.current_libero_type()
            or requested_model_path != self.current_model_path()
        ):
            launch_config = self.last_launch_config.copy()
            launch_config.update(
                {
                    "libero_type": target_type,
                    "task_suite_name": suite_name,
                    "task_id": task_id,
                    "trial_idx": trial_idx,
                    "model_path": requested_model_path,
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
            "task_id": _normalize_optional_task_id(config.get("task_id", self.default_config["task_id"])),
            "trial_idx": int(config.get("trial_idx", self.default_config["trial_idx"])),
            "model_path": str(
                _normalize_model_path(config.get("model_path", self.default_config["model_path"]))
            ),
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

        if normalized["task_id"] is None:
            raise ValueError("task_id is required. Choose a task before launching the worker.")

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
            "PI05_MODEL_PATH": config["model_path"],
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

        self.model_path = pathlib.Path(config["model_path"])
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
        tmux_tail = self.sync_current_run_dir() if self.session_alive() else ""
        return {
            "session_name": self.session_name,
            "session_alive": self.session_alive(),
            "current_run_dir": str(self.current_run_dir) if self.current_run_dir else None,
            "defaults": self.default_config,
            "last_launch_config": self.last_launch_config,
            "current_model_path": self.current_model_path(),
            "available_libero_types": self.available_libero_types,
            "available_suites": AVAILABLE_SUITES,
            "tmux_tail": tmux_tail,
            "build_token": BUILD_TOKEN,
        }


__all__ = [
    "AVAILABLE_LIBERO_TYPES",
    "AVAILABLE_SUITES",
    "BUILD_TOKEN",
    "DebugSessionController",
    "_build_state",
    "_normalize_model_path",
    "_safe_relative_path",
]

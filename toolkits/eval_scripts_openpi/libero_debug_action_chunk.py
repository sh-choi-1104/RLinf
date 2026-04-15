#!/usr/bin/env python3

# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import collections
import contextlib
import io
import json
import pathlib
import shlex
import sys
from typing import Any

import imageio
import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from toolkits.eval_scripts_openpi import setup_logger, setup_policy
from toolkits.eval_scripts_openpi.libero_eval import (
    LIBERO_DUMMY_ACTION,
    LIBERO_ENV_RESOLUTION,
    SUITE_MAX_STEPS,
    SUPPORTED_LIBERO_TYPES,
    _get_libero_env,
    _import_libero_stack,
    _load_taxonomy_lookup,
    _quat2axisangle,
    _sanitize_filename,
)

ACTION_DIM_LABELS = ["eef_x", "eef_y", "eef_z", "rot_x", "rot_y", "rot_z", "gripper"]
DEBUG_SUITE_CHOICES = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"]


def _format_float_list(values: np.ndarray) -> str:
    return " ".join(f"{float(value):+0.6f}" for value in values)


def _format_action_chunk(actions: np.ndarray) -> str:
    lines = []
    num_dims = actions.shape[1] if actions.ndim == 2 else 0
    header_labels = ACTION_DIM_LABELS[:num_dims] + [
        f"a{dim_idx}" for dim_idx in range(len(ACTION_DIM_LABELS), num_dims)
    ]
    if header_labels:
        lines.append("dims: " + ", ".join(f"{idx}={label}" for idx, label in enumerate(header_labels)))
    for step_idx, action in enumerate(actions):
        pieces = []
        for dim_idx, value in enumerate(action):
            if dim_idx < len(ACTION_DIM_LABELS):
                label = ACTION_DIM_LABELS[dim_idx]
            else:
                label = f"a{dim_idx}"
            pieces.append(f"{label}={float(value):+0.6f}")
        lines.append(f"[{step_idx:02d}] " + " ".join(pieces))
    return "\n".join(lines)


def _chunk_preview(actions: np.ndarray) -> dict[str, Any]:
    return {
        "shape": list(actions.shape),
        "min": float(np.min(actions)),
        "max": float(np.max(actions)),
        "mean_abs": float(np.mean(np.abs(actions))),
        "first_action": [float(value) for value in actions[0].tolist()],
    }


def _parse_dim_token(token: str, num_dims: int) -> int:
    if token.isdigit():
        dim_idx = int(token)
    else:
        lowered = token.lower()
        label_lookup = {label.lower(): idx for idx, label in enumerate(ACTION_DIM_LABELS)}
        if lowered not in label_lookup:
            raise ValueError(
                f"Unknown action dim '{token}'. Use an integer index or one of {ACTION_DIM_LABELS[:num_dims]}."
            )
        dim_idx = label_lookup[lowered]

    if dim_idx < 0 or dim_idx >= num_dims:
        raise ValueError(f"Action dim index {dim_idx} is out of range for {num_dims} dims.")
    return dim_idx


def _load_action_chunk_json(path: pathlib.Path) -> np.ndarray:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "actions" in payload:
        payload = payload["actions"]
    array = np.asarray(payload, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array in {path}, got shape={array.shape}")
    return array


def _parse_switch_request(
    parts: list[str],
    current_suite_name: str,
    current_trial_idx: int,
) -> dict[str, Any]:
    if len(parts) == 2:
        suite_name = current_suite_name
        task_id = int(parts[1])
        trial_idx = current_trial_idx
    elif len(parts) == 3:
        suite_name = parts[1]
        task_id = int(parts[2])
        trial_idx = current_trial_idx
    elif len(parts) == 4:
        suite_name = parts[1]
        task_id = int(parts[2])
        trial_idx = int(parts[3])
    else:
        raise ValueError("switch expects: <task_id>, <suite> <task_id>, or <suite> <task_id> <trial_idx>.")

    if suite_name not in DEBUG_SUITE_CHOICES:
        raise ValueError(f"Unsupported suite '{suite_name}'. Expected one of {DEBUG_SUITE_CHOICES}.")

    return {
        "type": "switch_task",
        "suite_name": suite_name,
        "task_id": task_id,
        "trial_idx": trial_idx,
    }


def _save_action_chunk_json(path: pathlib.Path, actions: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"actions": [[float(value) for value in row] for row in actions.tolist()]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _append_jsonl(record: dict[str, Any], out_path: pathlib.Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _prepare_media_frames(
    replay_images: list[np.ndarray],
    video_temp_subsample: int,
) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for frame in replay_images[:: max(1, video_temp_subsample)]:
        frames.append(np.asarray(frame))
    return frames


def _write_live_status(run_dir: pathlib.Path, payload: dict[str, Any]):
    live_dir = run_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    (live_dir / "status.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_latest_frame(run_dir: pathlib.Path, frame: np.ndarray):
    live_dir = run_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(live_dir / "latest_frame.jpg", np.asarray(frame))


def _write_live_chunk_video(
    run_dir: pathlib.Path,
    replay_images: list[np.ndarray],
    video_temp_subsample: int,
) -> pathlib.Path | None:
    if not replay_images:
        return None

    live_dir = run_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    out_path = live_dir / "current_chunk.mp4"
    fps = max(1, 30 // max(1, video_temp_subsample))
    frames = _prepare_media_frames(replay_images, video_temp_subsample)
    imageio.mimwrite(
        out_path,
        frames,
        fps=fps,
    )
    return out_path


def _write_live_chunk_gif(
    run_dir: pathlib.Path,
    replay_images: list[np.ndarray],
    video_temp_subsample: int,
) -> pathlib.Path | None:
    if not replay_images:
        return None

    live_dir = run_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)
    out_path = live_dir / "current_chunk.gif"
    frames = _prepare_media_frames(replay_images, video_temp_subsample)
    imageio.mimsave(out_path, frames, format="GIF", duration=0.15)
    return out_path


def _write_live_scene_preview(
    run_dir: pathlib.Path,
    frame: np.ndarray,
    video_temp_subsample: int,
) -> dict[str, str | None]:
    repeated_frames = [np.asarray(frame) for _ in range(8)]
    _write_latest_frame(run_dir, frame)
    live_video_path = _write_live_chunk_video(
        run_dir=run_dir,
        replay_images=repeated_frames,
        video_temp_subsample=video_temp_subsample,
    )
    live_gif_path = _write_live_chunk_gif(
        run_dir=run_dir,
        replay_images=repeated_frames,
        video_temp_subsample=video_temp_subsample,
    )
    return {
        "live_chunk_video_path": str(live_video_path) if live_video_path else None,
        "live_chunk_gif_path": str(live_gif_path) if live_gif_path else None,
    }


def _save_chunk_video(
    run_dir: pathlib.Path,
    suite_name: str,
    task_id: int,
    task_name: str,
    chunk_idx: int,
    replay_images: list[np.ndarray],
    video_temp_subsample: int,
) -> pathlib.Path | None:
    if not replay_images:
        return None

    out_dir = run_dir / "chunk_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"{suite_name}_task{task_id:04d}_chunk{chunk_idx:03d}_{_sanitize_filename(task_name)}.mp4"
    )
    fps = max(1, 30 // max(1, video_temp_subsample))
    frames = _prepare_media_frames(replay_images, video_temp_subsample)
    imageio.mimwrite(
        out_path,
        frames,
        fps=fps,
    )
    return out_path


def _save_chunk_gif(
    run_dir: pathlib.Path,
    suite_name: str,
    task_id: int,
    task_name: str,
    chunk_idx: int,
    replay_images: list[np.ndarray],
    video_temp_subsample: int,
) -> pathlib.Path | None:
    if not replay_images:
        return None

    out_dir = run_dir / "chunk_videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        f"{suite_name}_task{task_id:04d}_chunk{chunk_idx:03d}_{_sanitize_filename(task_name)}.gif"
    )
    frames = _prepare_media_frames(replay_images, video_temp_subsample)
    imageio.mimsave(out_path, frames, format="GIF", duration=0.15)
    return out_path


def _write_live_completed_chunk_media(
    run_dir: pathlib.Path,
    replay_images: list[np.ndarray],
    video_temp_subsample: int,
) -> dict[str, str | None]:
    live_dir = run_dir / "live"
    live_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = live_dir / "last_completed_chunk.mp4"
    gif_path = live_dir / "last_completed_chunk.gif"

    if replay_images:
        fps = max(1, 30 // max(1, video_temp_subsample))
        frames = _prepare_media_frames(replay_images, video_temp_subsample)
        imageio.mimwrite(mp4_path, frames, fps=fps)
        imageio.mimsave(gif_path, frames, format="GIF", duration=0.15)
        return {
            "last_completed_chunk_mp4": str(mp4_path),
            "last_completed_chunk_gif": str(gif_path),
        }

    return {
        "last_completed_chunk_mp4": None,
        "last_completed_chunk_gif": None,
    }


def _prompt_edit_action_chunk(
    predicted_actions: np.ndarray,
    run_dir: pathlib.Path,
    chunk_idx: int,
    current_suite_name: str,
    current_task_id: int,
    current_trial_idx: int,
    on_simulate=None,
) -> tuple[np.ndarray, list[str], dict[str, Any] | None]:
    current = predicted_actions.copy()
    original = predicted_actions.copy()
    edit_history: list[str] = []
    overrides_dir = run_dir / "chunk_overrides"
    default_save_path = overrides_dir / f"chunk_{chunk_idx:03d}.json"

    help_text = (
        "Commands:\n"
        "  run|enter               execute the current action chunk\n"
        "  print                   show the current action chunk again\n"
        "  reset                   restore the policy output for this chunk\n"
        "  set <step> <dim> <v>    set one scalar value\n"
        "  add <step> <dim> <dv>   add delta to one scalar value\n"
        "  scale <step> <dim> <f>  scale one scalar value by factor\n"
        "  row <step> <v0..vN>     replace one full action row\n"
        "  bias_dim <dim> <dv>     add delta to one dim across all steps\n"
        "  scale_dim <dim> <f>     scale one dim across all steps\n"
        "  simulate                preview this chunk without advancing the real env\n"
        "  switch <task_id>                switch task within current suite and keep trial\n"
        "  switch <suite> <task_id>        switch suite/task and keep current trial\n"
        "  switch <suite> <task_id> <trial_idx>  switch suite/task/trial without reloading model\n"
        "  save [path]             write current chunk to JSON\n"
        "  load <path>             load replacement chunk from JSON\n"
        "  help                    show this help text\n"
        "  quit                    stop the debug run without executing this chunk"
    )

    print(help_text)
    print(_format_action_chunk(current))

    while True:
        try:
            raw = input(f"chunk[{chunk_idx:03d}]> ").strip()
        except EOFError:
            return current, edit_history, None

        if raw == "" or raw.lower() in {"run", "r", "continue", "c"}:
            return current, edit_history, {"type": "run"}

        parts = shlex.split(raw)
        command = parts[0].lower()

        try:
            if command == "help":
                print(help_text)
                continue
            if command == "print":
                print(_format_action_chunk(current))
                continue
            if command == "reset":
                current = original.copy()
                edit_history.append("reset")
                print(_format_action_chunk(current))
                continue
            if command == "set" and len(parts) == 4:
                step_idx = int(parts[1])
                dim_idx = _parse_dim_token(parts[2], current.shape[1])
                current[step_idx, dim_idx] = float(parts[3])
                edit_history.append(raw)
                print(_format_action_chunk(current))
                continue
            if command == "add" and len(parts) == 4:
                step_idx = int(parts[1])
                dim_idx = _parse_dim_token(parts[2], current.shape[1])
                current[step_idx, dim_idx] += float(parts[3])
                edit_history.append(raw)
                print(_format_action_chunk(current))
                continue
            if command == "scale" and len(parts) == 4:
                step_idx = int(parts[1])
                dim_idx = _parse_dim_token(parts[2], current.shape[1])
                current[step_idx, dim_idx] *= float(parts[3])
                edit_history.append(raw)
                print(_format_action_chunk(current))
                continue
            if command == "row" and len(parts) == current.shape[1] + 2:
                step_idx = int(parts[1])
                current[step_idx] = np.asarray([float(value) for value in parts[2:]], dtype=np.float32)
                edit_history.append(raw)
                print(_format_action_chunk(current))
                continue
            if command == "bias_dim" and len(parts) == 3:
                dim_idx = _parse_dim_token(parts[1], current.shape[1])
                current[:, dim_idx] += float(parts[2])
                edit_history.append(raw)
                print(_format_action_chunk(current))
                continue
            if command == "scale_dim" and len(parts) == 3:
                dim_idx = _parse_dim_token(parts[1], current.shape[1])
                current[:, dim_idx] *= float(parts[2])
                edit_history.append(raw)
                print(_format_action_chunk(current))
                continue
            if command in {"simulate", "sim", "preview", "p"} and len(parts) == 1:
                if on_simulate is None:
                    print("simulate preview is not available in this mode")
                    continue
                on_simulate(current.copy(), edit_history.copy())
                continue
            if command == "save" and len(parts) in {1, 2}:
                save_path = pathlib.Path(parts[1]) if len(parts) == 2 else default_save_path
                _save_action_chunk_json(save_path, current)
                print(f"saved {save_path}")
                continue
            if command == "switch" and len(parts) in {2, 3, 4}:
                return current, edit_history, _parse_switch_request(
                    parts,
                    current_suite_name=current_suite_name,
                    current_trial_idx=current_trial_idx,
                )
            if command == "load" and len(parts) == 2:
                load_path = pathlib.Path(parts[1])
                loaded = _load_action_chunk_json(load_path)
                if loaded.shape != current.shape:
                    raise ValueError(
                        f"Loaded chunk shape {loaded.shape} does not match expected {current.shape}."
                    )
                current = loaded.astype(np.float32, copy=False)
                edit_history.append(f"load {load_path}")
                print(_format_action_chunk(current))
                continue
            if command in {"quit", "q", "exit"}:
                return current, edit_history, {"type": "quit"}
        except IndexError as exc:
            print(f"index error: {exc}")
            continue
        except ValueError as exc:
            print(f"value error: {exc}")
            continue

        print("unknown command; type 'help' to see available commands")


def _prompt_after_episode(
    current_suite_name: str,
    current_task_id: int,
    current_trial_idx: int,
) -> dict[str, Any]:
    help_text = (
        "Commands:\n"
        "  rerun|enter              rerun the current task without reloading model\n"
        "  switch <task_id>                switch task within current suite and keep trial\n"
        "  switch <suite> <task_id>        switch suite/task and keep current trial\n"
        "  switch <suite> <task_id> <trial_idx>  switch suite/task/trial without reloading model\n"
        "  help                    show this help text\n"
        "  quit                    stop the debug session"
    )

    print(help_text)
    while True:
        try:
            raw = input(
                f"episode[{current_suite_name} task={current_task_id} trial={current_trial_idx}]> "
            ).strip()
        except EOFError:
            return {"type": "quit"}

        if raw == "" or raw.lower() in {"rerun", "r", "again"}:
            return {"type": "rerun"}

        parts = shlex.split(raw)
        command = parts[0].lower()
        try:
            if command == "help":
                print(help_text)
                continue
            if command == "switch" and len(parts) in {2, 3, 4}:
                return _parse_switch_request(
                    parts,
                    current_suite_name=current_suite_name,
                    current_trial_idx=current_trial_idx,
                )
            if command in {"quit", "q", "exit"}:
                return {"type": "quit"}
        except ValueError as exc:
            print(f"value error: {exc}")
            continue

        print("unknown command; type 'help' to see available commands")


def _capture_robot_state(obs: dict[str, Any]) -> dict[str, list[float]]:
    eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
    eef_axisangle = _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32))
    gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
    return {
        "robot0_eef_pos": [float(value) for value in eef_pos.tolist()],
        "robot0_eef_axisangle": [float(value) for value in eef_axisangle.tolist()],
        "robot0_gripper_qpos": [float(value) for value in gripper_qpos.tolist()],
    }


def _simulate_action_chunk_preview(
    *,
    run_dir: pathlib.Path,
    task,
    initial_state,
    committed_action_history: list[np.ndarray],
    preview_actions: np.ndarray,
    args,
    get_libero_path_fn,
    env_cls,
    logger,
) -> tuple[list[np.ndarray], bool]:
    preview_env, _task_description = _get_libero_env(
        task,
        LIBERO_ENV_RESOLUTION,
        args.seed,
        get_libero_path_fn,
        env_cls,
    )

    try:
        preview_env.reset()
        preview_obs = preview_env.set_init_state(initial_state)
        for _ in range(args.num_steps_wait):
            preview_obs, _reward, _done, _info = preview_env.step(LIBERO_DUMMY_ACTION)

        for committed_action in committed_action_history:
            preview_obs, _reward, _done, _info = preview_env.step(committed_action.tolist())

        preview_frames = [np.ascontiguousarray(preview_obs["agentview_image"][::-1, ::-1])]
        preview_done = False

        for preview_step_idx, preview_action in enumerate(preview_actions):
            logger.info(
                "simulate chunk_step=%s action=%s",
                preview_step_idx,
                _format_float_list(preview_action),
            )
            preview_obs, _reward, preview_done, _info = preview_env.step(preview_action.tolist())
            preview_frame = np.ascontiguousarray(preview_obs["agentview_image"][::-1, ::-1])
            preview_frames.append(preview_frame)
            if preview_done:
                break

        if preview_frames:
            _write_latest_frame(run_dir, preview_frames[-1])
            _write_live_chunk_video(
                run_dir=run_dir,
                replay_images=preview_frames,
                video_temp_subsample=args.video_temp_subsample,
            )
            _write_live_chunk_gif(
                run_dir=run_dir,
                replay_images=preview_frames,
                video_temp_subsample=args.video_temp_subsample,
            )

        return preview_frames, bool(preview_done)
    finally:
        preview_env.close()


def _resolve_task_bundle(
    *,
    suite_name: str,
    task_id: int,
    trial_idx: int,
    benchmark_dict,
    task_suites_cache: dict[str, Any],
    taxonomy_lookup: dict[str, dict[str, str]],
    get_libero_path_fn,
    env_cls,
    seed: int,
):
    if suite_name not in DEBUG_SUITE_CHOICES:
        raise ValueError(f"Unsupported suite_name={suite_name}. Expected one of {DEBUG_SUITE_CHOICES}.")

    if suite_name not in task_suites_cache:
        with contextlib.redirect_stdout(io.StringIO()):
            task_suites_cache[suite_name] = benchmark_dict[suite_name]()
    task_suite = task_suites_cache[suite_name]

    if task_id < 0 or task_id >= task_suite.n_tasks:
        raise ValueError(
            f"task_id={task_id} is out of range for suite={suite_name} with {task_suite.n_tasks} tasks."
        )

    task = task_suite.get_task(task_id)
    task_name = pathlib.Path(task.bddl_file).stem
    taxonomy_name = taxonomy_lookup.get(suite_name, {}).get(task_name, "Unknown")
    initial_states = task_suite.get_task_init_states(task_id)
    if trial_idx < 0 or trial_idx >= len(initial_states):
        raise ValueError(
            f"trial_idx={trial_idx} is out of range for suite={suite_name} task_id={task_id}; "
            f"available_trials={len(initial_states)}."
        )

    env, task_description = _get_libero_env(
        task, LIBERO_ENV_RESOLUTION, seed, get_libero_path_fn, env_cls
    )

    return {
        "suite_name": suite_name,
        "task_id": task_id,
        "trial_idx": trial_idx,
        "task_suite": task_suite,
        "task": task,
        "task_name": task_name,
        "taxonomy_name": taxonomy_name,
        "initial_states": initial_states,
        "env": env,
        "task_description": task_description,
        "max_steps": SUITE_MAX_STEPS[suite_name],
    }


def _save_debug_rollout_video(
    run_dir: pathlib.Path,
    suite_name: str,
    task_id: int,
    task_name: str,
    episode_idx: int,
    success: bool,
    replay_images: list[np.ndarray],
    video_temp_subsample: int,
) -> pathlib.Path | None:
    if not replay_images:
        return None

    suffix = "success" if success else "failure"
    video_dir = run_dir / "videos" / suffix
    video_dir.mkdir(parents=True, exist_ok=True)
    out_path = video_dir / (
        f"{suite_name}_task{task_id:04d}_trial{episode_idx:03d}_"
        f"{_sanitize_filename(task_name)}_{suffix}.mp4"
    )
    fps = max(1, 30 // max(1, video_temp_subsample))
    frames = _prepare_media_frames(replay_images, video_temp_subsample)
    imageio.mimwrite(out_path, frames, fps=fps)
    return out_path


def main(args):
    if args.libero_type not in SUPPORTED_LIBERO_TYPES:
        raise ValueError(
            f"Unsupported libero_type={args.libero_type}. Expected one of {sorted(SUPPORTED_LIBERO_TYPES)}."
        )
    if args.task_suite_name == "all":
        raise ValueError("Debug mode requires a concrete suite, not task_suite_name=all.")

    logger = setup_logger(args.exp_name, args.log_dir)
    run_dir = pathlib.Path(args.log_dir) / args.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    trace_path = run_dir / "debug_trace.jsonl"
    summary_path = run_dir / "debug_summary.json"

    benchmark_module, get_libero_path_fn, env_cls, package_root = _import_libero_stack(
        args.libero_type, args.asset_root
    )
    taxonomy_lookup = (
        _load_taxonomy_lookup(package_root, DEBUG_SUITE_CHOICES)
        if args.libero_type == "plus"
        else {}
    )
    benchmark_dict = benchmark_module.get_benchmark_dict()

    logger.info("policy setup start")
    policy = setup_policy(args)
    logger.info("policy setup done")

    logger.info("libero_type=%s", args.libero_type)
    logger.info("run_dir=%s", run_dir)
    logger.info("action dims labels=%s", ACTION_DIM_LABELS)
    if args.interactive_chunk_edit:
        logger.info("interactive chunk edit is enabled")
    task_suites_cache: dict[str, Any] = {}
    active_spec = {
        "suite_name": args.task_suite_name,
        "task_id": args.task_id,
        "trial_idx": args.trial_idx,
    }

    while True:
        task_bundle = _resolve_task_bundle(
            suite_name=active_spec["suite_name"],
            task_id=active_spec["task_id"],
            trial_idx=active_spec["trial_idx"],
            benchmark_dict=benchmark_dict,
            task_suites_cache=task_suites_cache,
            taxonomy_lookup=taxonomy_lookup,
            get_libero_path_fn=get_libero_path_fn,
            env_cls=env_cls,
            seed=args.seed,
        )
        env = task_bundle["env"]
        task_name = task_bundle["task_name"]
        taxonomy_name = task_bundle["taxonomy_name"]
        task_description = task_bundle["task_description"]
        initial_states = task_bundle["initial_states"]
        max_steps = args.max_steps or task_bundle["max_steps"]

        logger.info(
            "suite=%s task_id=%s taxonomy=%s trial_idx=%s",
            active_spec["suite_name"],
            active_spec["task_id"],
            taxonomy_name,
            active_spec["trial_idx"],
        )
        logger.info("task_name=%s", task_name)
        logger.info("task_description=%s", task_description)
        logger.info(
            "action_chunk=%s num_steps=%s num_steps_wait=%s",
            args.action_chunk,
            args.num_steps,
            args.num_steps_wait,
        )

        _write_live_status(
            run_dir,
            {
                "phase": "setup",
                "libero_type": args.libero_type,
                "suite_name": active_spec["suite_name"],
                "task_id": active_spec["task_id"],
                "task_name": task_name,
                "task_description": task_description,
                "taxonomy": taxonomy_name,
                "trial_idx": active_spec["trial_idx"],
                "model_reused": True,
            },
        )

        policy.reset()
        env.reset()
        obs = env.set_init_state(initial_states[active_spec["trial_idx"]])
        action_plan: collections.deque[np.ndarray] = collections.deque()
        chunk_step_plan: list[np.ndarray] = []
        rollout_frames: list[np.ndarray] = []
        current_chunk_frames: list[np.ndarray] = []
        current_plan_status: dict[str, Any] = {
            "action_dim_labels": ACTION_DIM_LABELS,
            "predicted_actions": None,
            "current_plan_actions": None,
            "edit_history": [],
        }
        chunk_idx = -1
        episode_success = False
        executed_actions = 0
        committed_action_history: list[np.ndarray] = []
        stop_reason = "max_steps_reached"
        switch_request: dict[str, Any] | None = None

        for t in range(max_steps + args.num_steps_wait):
            if t < args.num_steps_wait:
                obs, _reward, _done, _info = env.step(LIBERO_DUMMY_ACTION)
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            rollout_frames.append(img)
            if not current_chunk_frames:
                current_chunk_frames.append(img)

            state = np.concatenate(
                (
                    np.asarray(obs["robot0_eef_pos"], dtype=np.float32),
                    _quat2axisangle(np.asarray(obs["robot0_eef_quat"], dtype=np.float32)),
                    np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32),
                )
            )

            if not action_plan:
                if args.stop_after_chunks is not None and chunk_idx + 1 >= args.stop_after_chunks:
                    stop_reason = "stop_after_chunks_reached"
                    break

                chunk_idx += 1
                observation = {
                    "observation/image": img,
                    "observation/wrist_image": wrist_img,
                    "observation/state": state,
                    "prompt": str(task_description),
                }
                policy_output = np.asarray(policy.infer(observation)["actions"], dtype=np.float32)
                if len(policy_output) < args.action_chunk:
                    raise AssertionError(
                        f"We want to replan every {args.action_chunk} steps, "
                        f"but policy only predicts {len(policy_output)} steps."
                    )
                predicted_actions = policy_output[: args.action_chunk].copy()
                edited_actions = predicted_actions.copy()
                edit_history: list[str] = []
                current_plan_status = {
                    "action_dim_labels": ACTION_DIM_LABELS,
                    "observation_state": [float(value) for value in state.tolist()],
                    "predicted_actions": [[float(value) for value in row] for row in predicted_actions.tolist()],
                    "current_plan_actions": [[float(value) for value in row] for row in edited_actions.tolist()],
                    "edit_history": edit_history,
                }
                scene_preview_media: dict[str, str | None] = {}
                if args.save_live_preview:
                    scene_preview_media = _write_live_scene_preview(
                        run_dir=run_dir,
                        frame=img,
                        video_temp_subsample=args.video_temp_subsample,
                    )

                logger.info(
                    "chunk=%s policy_chunk_preview=%s",
                    chunk_idx,
                    json.dumps(_chunk_preview(predicted_actions), ensure_ascii=True),
                )
                if args.print_observation_state:
                    logger.info("chunk=%s observation_state=%s", chunk_idx, _format_float_list(state))
                if args.print_policy_chunks:
                    logger.info(
                        "chunk=%s predicted_action_chunk:\n%s",
                        chunk_idx,
                        _format_action_chunk(predicted_actions),
                    )

                _write_live_status(
                    run_dir,
                    {
                        "phase": "awaiting_chunk_execution" if args.interactive_chunk_edit else "executing_chunk",
                        "libero_type": args.libero_type,
                        "chunk_idx": chunk_idx,
                        "suite_name": active_spec["suite_name"],
                        "task_id": active_spec["task_id"],
                        "task_name": task_name,
                        "taxonomy": taxonomy_name,
                        "trial_idx": active_spec["trial_idx"],
                        "model_reused": True,
                        **current_plan_status,
                        **scene_preview_media,
                    },
                )

                prompt_result = {"type": "run"}
                if args.interactive_chunk_edit:
                    def _simulate_preview_callback(simulated_actions, simulated_edit_history):
                        preview_plan_status = {
                            "action_dim_labels": ACTION_DIM_LABELS,
                            "observation_state": [float(value) for value in state.tolist()],
                            "predicted_actions": [
                                [float(value) for value in row] for row in predicted_actions.tolist()
                            ],
                            "current_plan_actions": [
                                [float(value) for value in row] for row in simulated_actions.tolist()
                            ],
                            "edit_history": simulated_edit_history,
                        }
                        if args.save_live_preview:
                            _write_live_status(
                                run_dir,
                                {
                                    "phase": "simulating_chunk",
                                    "libero_type": args.libero_type,
                                    "chunk_idx": chunk_idx,
                                    "suite_name": active_spec["suite_name"],
                                    "task_id": active_spec["task_id"],
                                    "task_name": task_name,
                                    "taxonomy": taxonomy_name,
                                    "trial_idx": active_spec["trial_idx"],
                                    "model_reused": True,
                                    **preview_plan_status,
                                },
                            )
                        try:
                            preview_frames, preview_done = _simulate_action_chunk_preview(
                                run_dir=run_dir,
                                task=task_bundle["task"],
                                initial_state=initial_states[active_spec["trial_idx"]],
                                committed_action_history=committed_action_history,
                                preview_actions=simulated_actions,
                                args=args,
                                get_libero_path_fn=get_libero_path_fn,
                                env_cls=env_cls,
                                logger=logger,
                            )
                        except Exception as exc:
                            logger.exception("simulate preview failed for chunk=%s", chunk_idx)
                            if args.save_live_preview:
                                fallback_scene_media = _write_live_scene_preview(
                                    run_dir=run_dir,
                                    frame=img,
                                    video_temp_subsample=args.video_temp_subsample,
                                )
                                _write_live_status(
                                    run_dir,
                                    {
                                        "phase": "awaiting_chunk_execution",
                                        "libero_type": args.libero_type,
                                        "chunk_idx": chunk_idx,
                                        "suite_name": active_spec["suite_name"],
                                        "task_id": active_spec["task_id"],
                                        "task_name": task_name,
                                        "taxonomy": taxonomy_name,
                                        "trial_idx": active_spec["trial_idx"],
                                        "model_reused": True,
                                        "simulation_error": str(exc),
                                        **preview_plan_status,
                                        **fallback_scene_media,
                                    },
                                )
                            print(f"simulate preview failed: {exc}")
                            return

                        _append_jsonl(
                            {
                                "event": "simulate_preview",
                                "chunk_idx": chunk_idx,
                                "suite_name": active_spec["suite_name"],
                                "task_id": active_spec["task_id"],
                                "trial_idx": active_spec["trial_idx"],
                                "simulation_done": preview_done,
                                "simulation_frame_count": len(preview_frames),
                                "simulated_actions": [
                                    [float(value) for value in row] for row in simulated_actions.tolist()
                                ],
                                "edit_history": simulated_edit_history,
                            },
                            trace_path,
                        )
                        if args.save_live_preview:
                            _write_live_status(
                                run_dir,
                                {
                                    "phase": "awaiting_chunk_execution",
                                    "libero_type": args.libero_type,
                                    "chunk_idx": chunk_idx,
                                    "suite_name": active_spec["suite_name"],
                                    "task_id": active_spec["task_id"],
                                    "task_name": task_name,
                                    "taxonomy": taxonomy_name,
                                    "trial_idx": active_spec["trial_idx"],
                                    "model_reused": True,
                                    "simulation_done": preview_done,
                                    "simulation_frame_count": len(preview_frames),
                                    **preview_plan_status,
                                },
                            )
                        print(
                            f"simulate preview updated for chunk[{chunk_idx:03d}] "
                            f"(frames={len(preview_frames)} done={preview_done})"
                        )

                    edited_actions, edit_history, prompt_result = _prompt_edit_action_chunk(
                        predicted_actions=predicted_actions,
                        run_dir=run_dir,
                        chunk_idx=chunk_idx,
                        current_suite_name=active_spec["suite_name"],
                        current_task_id=active_spec["task_id"],
                        current_trial_idx=active_spec["trial_idx"],
                        on_simulate=_simulate_preview_callback,
                    )
                    if prompt_result and prompt_result.get("type") == "quit":
                        stop_reason = "user_quit_before_chunk_execution"
                        break
                    if prompt_result and prompt_result.get("type") == "switch_task":
                        switch_request = {
                            "suite_name": prompt_result["suite_name"],
                            "task_id": int(prompt_result["task_id"]),
                            "trial_idx": int(prompt_result["trial_idx"]),
                        }
                        stop_reason = "switch_task_requested"
                        logger.info(
                            "Switching to suite=%s task_id=%s trial_idx=%s without reloading policy.",
                            switch_request["suite_name"],
                            switch_request["task_id"],
                            switch_request["trial_idx"],
                        )
                        _write_live_status(
                            run_dir,
                            {
                                "phase": "switching_task",
                                "libero_type": args.libero_type,
                                "from_suite_name": active_spec["suite_name"],
                                "from_task_id": active_spec["task_id"],
                                "from_trial_idx": active_spec["trial_idx"],
                                "to_suite_name": switch_request["suite_name"],
                                "to_task_id": switch_request["task_id"],
                                "to_trial_idx": switch_request["trial_idx"],
                                "model_reused": True,
                            },
                        )
                        break
                    logger.info(
                        "chunk=%s edited_action_chunk:\n%s",
                        chunk_idx,
                        _format_action_chunk(edited_actions),
                    )

                if switch_request is not None:
                    break

                chunk_step_plan = [action.copy() for action in edited_actions]
                action_plan.extend(chunk_step_plan)
                current_plan_status = {
                    "action_dim_labels": ACTION_DIM_LABELS,
                    "observation_state": [float(value) for value in state.tolist()],
                    "predicted_actions": [[float(value) for value in row] for row in predicted_actions.tolist()],
                    "current_plan_actions": [[float(value) for value in row] for row in edited_actions.tolist()],
                    "edit_history": edit_history,
                }
                _write_live_status(
                    run_dir,
                    {
                        "phase": "awaiting_chunk_execution" if args.interactive_chunk_edit else "executing_chunk",
                        "libero_type": args.libero_type,
                        "chunk_idx": chunk_idx,
                        "suite_name": active_spec["suite_name"],
                        "task_id": active_spec["task_id"],
                        "task_name": task_name,
                        "taxonomy": taxonomy_name,
                        "trial_idx": active_spec["trial_idx"],
                        "model_reused": True,
                        **current_plan_status,
                    },
                )
                _append_jsonl(
                    {
                        "event": "plan",
                        "chunk_idx": chunk_idx,
                        "suite_name": active_spec["suite_name"],
                        "task_id": active_spec["task_id"],
                        "task_name": task_name,
                        "task_description": task_description,
                        "taxonomy": taxonomy_name,
                        "trial_idx": active_spec["trial_idx"],
                        "observation_state": [float(value) for value in state.tolist()],
                        "predicted_actions": [[float(value) for value in row] for row in predicted_actions.tolist()],
                        "executed_actions": [[float(value) for value in row] for row in edited_actions.tolist()],
                        "edit_history": edit_history,
                    },
                    trace_path,
                )

            if switch_request is not None:
                break

            action = np.asarray(action_plan.popleft(), dtype=np.float32)
            step_in_chunk = len(chunk_step_plan) - len(action_plan) - 1
            logger.info(
                "env_step=%s chunk=%s chunk_step=%s action=%s",
                executed_actions,
                chunk_idx,
                step_in_chunk,
                _format_float_list(action),
            )

            obs, _reward, done, _info = env.step(action.tolist())
            executed_actions += 1
            committed_action_history.append(action.copy())
            post_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            rollout_frames.append(post_img)
            current_chunk_frames.append(post_img)
            robot_state = _capture_robot_state(obs)
            if args.print_step_state:
                logger.info(
                    "post_step env_step=%s eef_pos=%s eef_axisangle=%s gripper_qpos=%s",
                    executed_actions,
                    _format_float_list(np.asarray(robot_state["robot0_eef_pos"], dtype=np.float32)),
                    _format_float_list(np.asarray(robot_state["robot0_eef_axisangle"], dtype=np.float32)),
                    _format_float_list(np.asarray(robot_state["robot0_gripper_qpos"], dtype=np.float32)),
                )

            if args.save_live_preview:
                _write_latest_frame(run_dir, post_img)
                _write_live_chunk_video(
                    run_dir=run_dir,
                    replay_images=current_chunk_frames,
                    video_temp_subsample=args.video_temp_subsample,
                )
                _write_live_chunk_gif(
                    run_dir=run_dir,
                    replay_images=current_chunk_frames,
                    video_temp_subsample=args.video_temp_subsample,
                )
                _write_live_status(
                    run_dir,
                    {
                        "phase": "paused_after_action" if args.pause_after_action else "executing_chunk",
                        "libero_type": args.libero_type,
                        "chunk_idx": chunk_idx,
                        "chunk_step": step_in_chunk,
                        "env_step": executed_actions,
                        "task_name": task_name,
                        "taxonomy": taxonomy_name,
                        "suite_name": active_spec["suite_name"],
                        "task_id": active_spec["task_id"],
                        "trial_idx": active_spec["trial_idx"],
                        "done": bool(done),
                        "last_action": [float(value) for value in action.tolist()],
                        "model_reused": True,
                        **current_plan_status,
                        **robot_state,
                    },
                )

            _append_jsonl(
                {
                    "event": "step",
                    "chunk_idx": chunk_idx,
                    "chunk_step": step_in_chunk,
                    "env_step": executed_actions,
                    "action": [float(value) for value in action.tolist()],
                    "done": bool(done),
                    **robot_state,
                },
                trace_path,
            )

            if args.pause_after_action:
                try:
                    pause_raw = input("press enter to continue, or type 'quit' to stop: ").strip().lower()
                except EOFError:
                    pause_raw = ""
                if pause_raw in {"quit", "q", "exit"}:
                    stop_reason = "user_quit_after_action"
                    break

            if not action_plan:
                chunk_video_path = None
                chunk_gif_path = None
                live_completed_media = {
                    "last_completed_chunk_mp4": None,
                    "last_completed_chunk_gif": None,
                }

                if args.save_chunk_videos:
                    chunk_video_path = _save_chunk_video(
                        run_dir=run_dir,
                        suite_name=active_spec["suite_name"],
                        task_id=active_spec["task_id"],
                        task_name=task_name,
                        chunk_idx=chunk_idx,
                        replay_images=current_chunk_frames,
                        video_temp_subsample=args.video_temp_subsample,
                    )
                    chunk_gif_path = _save_chunk_gif(
                        run_dir=run_dir,
                        suite_name=active_spec["suite_name"],
                        task_id=active_spec["task_id"],
                        task_name=task_name,
                        chunk_idx=chunk_idx,
                        replay_images=current_chunk_frames,
                        video_temp_subsample=args.video_temp_subsample,
                    )
                    live_completed_media = _write_live_completed_chunk_media(
                        run_dir=run_dir,
                        replay_images=current_chunk_frames,
                        video_temp_subsample=args.video_temp_subsample,
                    )
                    _append_jsonl(
                        {
                            "event": "chunk_video",
                            "chunk_idx": chunk_idx,
                            "video_path": str(chunk_video_path) if chunk_video_path else None,
                            "gif_path": str(chunk_gif_path) if chunk_gif_path else None,
                        },
                        trace_path,
                    )

                if args.save_live_preview:
                    _write_live_status(
                        run_dir,
                        {
                            "phase": "planning_next_chunk" if not done else "chunk_complete",
                            "libero_type": args.libero_type,
                            "chunk_idx": chunk_idx,
                            "env_step": executed_actions,
                            "task_name": task_name,
                            "taxonomy": taxonomy_name,
                            "suite_name": active_spec["suite_name"],
                            "task_id": active_spec["task_id"],
                            "trial_idx": active_spec["trial_idx"],
                            "chunk_video_path": str(chunk_video_path) if chunk_video_path else None,
                            "chunk_gif_path": str(chunk_gif_path) if chunk_gif_path else None,
                            "model_reused": True,
                            **current_plan_status,
                            **live_completed_media,
                        },
                    )
                current_chunk_frames = []

            if done:
                episode_success = True
                stop_reason = "task_success"
                break

        if current_chunk_frames and args.save_chunk_videos and chunk_idx >= 0:
            chunk_video_path = _save_chunk_video(
                run_dir=run_dir,
                suite_name=active_spec["suite_name"],
                task_id=active_spec["task_id"],
                task_name=task_name,
                chunk_idx=chunk_idx,
                replay_images=current_chunk_frames,
                video_temp_subsample=args.video_temp_subsample,
            )
            _append_jsonl(
                {
                    "event": "chunk_video",
                    "chunk_idx": chunk_idx,
                    "video_path": str(chunk_video_path) if chunk_video_path else None,
                },
                trace_path,
            )

        rollout_video_path = None
        if args.save_rollout_video:
            rollout_video_path = _save_debug_rollout_video(
                run_dir=run_dir,
                suite_name=active_spec["suite_name"],
                task_id=active_spec["task_id"],
                task_name=task_name,
                episode_idx=active_spec["trial_idx"],
                success=episode_success,
                replay_images=rollout_frames,
                video_temp_subsample=args.video_temp_subsample,
            )

        summary = {
            "libero_type": args.libero_type,
            "suite_name": active_spec["suite_name"],
            "task_id": active_spec["task_id"],
            "task_name": task_name,
            "task_description": task_description,
            "taxonomy": taxonomy_name,
            "trial_idx": active_spec["trial_idx"],
            "action_chunk": args.action_chunk,
            "num_steps": args.num_steps,
            "num_steps_wait": args.num_steps_wait,
            "max_steps": max_steps,
            "stop_after_chunks": args.stop_after_chunks,
            "executed_actions": executed_actions,
            "success": episode_success,
            "stop_reason": stop_reason,
            "rollout_video_path": str(rollout_video_path) if rollout_video_path else None,
            "trace_path": str(trace_path),
            "model_reused": True,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        logger.info("debug summary: %s", json.dumps(summary, ensure_ascii=True))
        logger.info("trace_path=%s", trace_path)
        logger.info("summary_path=%s", summary_path)
        if rollout_video_path is not None:
            logger.info("rollout_video_path=%s", rollout_video_path)

        env.close()

        if switch_request is not None:
            _append_jsonl(
                {
                    "event": "switch_task",
                    "from_suite_name": active_spec["suite_name"],
                    "from_task_id": active_spec["task_id"],
                    "from_trial_idx": active_spec["trial_idx"],
                    "to_suite_name": switch_request["suite_name"],
                    "to_task_id": switch_request["task_id"],
                    "to_trial_idx": switch_request["trial_idx"],
                },
                trace_path,
            )
            active_spec = switch_request
            continue

        _write_live_status(
            run_dir,
            {
                "phase": "finished",
                "libero_type": args.libero_type,
                "summary": summary,
                "suite_name": active_spec["suite_name"],
                "task_id": active_spec["task_id"],
                "trial_idx": active_spec["trial_idx"],
                "task_name": task_name,
                "taxonomy": taxonomy_name,
                "model_reused": True,
            },
        )

        if args.interactive_chunk_edit:
            episode_result = _prompt_after_episode(
                current_suite_name=active_spec["suite_name"],
                current_task_id=active_spec["task_id"],
                current_trial_idx=active_spec["trial_idx"],
            )
            if episode_result.get("type") == "rerun":
                _append_jsonl(
                    {
                        "event": "rerun_task",
                        "suite_name": active_spec["suite_name"],
                        "task_id": active_spec["task_id"],
                        "trial_idx": active_spec["trial_idx"],
                    },
                    trace_path,
                )
                continue
            if episode_result.get("type") == "switch_task":
                switch_request = {
                    "suite_name": episode_result["suite_name"],
                    "task_id": int(episode_result["task_id"]),
                    "trial_idx": int(episode_result["trial_idx"]),
                }
                _append_jsonl(
                    {
                        "event": "switch_task",
                        "from_suite_name": active_spec["suite_name"],
                        "from_task_id": active_spec["task_id"],
                        "from_trial_idx": active_spec["trial_idx"],
                        "to_suite_name": switch_request["suite_name"],
                        "to_task_id": switch_request["task_id"],
                        "to_trial_idx": switch_request["trial_idx"],
                    },
                    trace_path,
                )
                active_spec = switch_request
                continue

        break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save debug logs and artifacts.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="libero_debug_action_chunk",
        help="Experiment name used for debug outputs.",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default="pi05_libero",
        help="OpenPI config name.",
    )
    parser.add_argument(
        "--pretrained_path",
        type=str,
        required=True,
        help="Path to the pretrained model directory.",
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default=None,
        help="Dataset repo id used to resolve norm stats, e.g. libero_plus_lerobot.",
    )
    parser.add_argument(
        "--norm_stats_dir",
        type=str,
        default=None,
        help="Optional explicit norm stats directory.",
    )
    parser.add_argument(
        "--task_suite_name",
        type=str,
        required=True,
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="Task suite to debug.",
    )
    parser.add_argument(
        "--task_id",
        type=int,
        required=True,
        help="Zero-based task id within the selected suite.",
    )
    parser.add_argument(
        "--trial_idx",
        type=int,
        default=0,
        help="Zero-based initial-state index to run.",
    )
    parser.add_argument(
        "--libero_type",
        type=str,
        default="standard",
        choices=sorted(SUPPORTED_LIBERO_TYPES),
        help="Which LIBERO package to evaluate against.",
    )
    parser.add_argument(
        "--asset_root",
        type=str,
        default=None,
        help="Optional asset root override for LIBERO Pro / Plus.",
    )
    parser.add_argument(
        "--action_chunk",
        type=int,
        default=5,
        help="Number of actions to execute before replanning.",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=5,
        help="Number of action steps sampled from the policy each time.",
    )
    parser.add_argument(
        "--num_steps_wait",
        type=int,
        default=10,
        help="Number of warmup sim steps before acting.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional explicit max number of control steps to allow.",
    )
    parser.add_argument(
        "--stop_after_chunks",
        type=int,
        default=None,
        help="Stop after executing this many replanned chunks.",
    )
    parser.add_argument(
        "--interactive_chunk_edit",
        action="store_true",
        help="Open a small REPL before each chunk is executed so you can edit values.",
    )
    parser.add_argument(
        "--pause_after_action",
        action="store_true",
        help="Pause after each executed action until you press enter.",
    )
    parser.add_argument(
        "--print_policy_chunks",
        action="store_true",
        help="Print the policy-produced action chunk before execution.",
    )
    parser.add_argument(
        "--print_observation_state",
        action="store_true",
        help="Print the state vector given to the policy at each replanning point.",
    )
    parser.add_argument(
        "--print_step_state",
        action="store_true",
        help="Print end-effector and gripper state after each executed action.",
    )
    parser.add_argument(
        "--save_rollout_video",
        action="store_true",
        help="Write a rollout video for the debug run.",
    )
    parser.add_argument(
        "--save_chunk_videos",
        action="store_true",
        help="Write one small video per executed chunk.",
    )
    parser.add_argument(
        "--save_live_preview",
        action="store_true",
        help="Continuously refresh live/current_chunk.mp4 and live/latest_frame.jpg after each action.",
    )
    parser.add_argument(
        "--video_temp_subsample",
        type=int,
        default=1,
        help="Save every Nth frame to each video.",
    )
    main(parser.parse_args())

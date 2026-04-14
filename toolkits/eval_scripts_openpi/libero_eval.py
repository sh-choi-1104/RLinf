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
import math
import os
import pathlib
import re
from collections import defaultdict
from typing import Any

import imageio
import numpy as np
import tqdm

from toolkits.eval_scripts_openpi import setup_logger, setup_policy

os.environ["MUJOCO_GL"] = "egl"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256

SUITE_MAX_STEPS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}

SUPPORTED_LIBERO_TYPES = {"standard", "pro", "plus"}
ALL_LIBERO_SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]

LIBERO_PLUS_TAXONOMY_MAP = {
    "Camera Viewpoints": "Camera",
    "Robot Initial States": "Robot",
    "Language Instructions": "Language",
    "Light Conditions": "Light",
    "Background Textures": "Background",
    "Sensor Noise": "Noise",
    "Objects Layout": "Layout",
}


def _quat2axisangle(quat):
    # Copied from robosuite:
    # https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _sanitize_filename(value: str, max_len: int = 96) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("_")
    return safe[:max_len] or "task"


def _import_libero_stack(libero_type: str, asset_root: str | None):
    if libero_type == "plus":
        import liberoplus.liberoplus as libero_pkg
        from liberoplus.liberoplus import benchmark as benchmark_module, get_libero_path
        from liberoplus.liberoplus.envs import OffScreenRenderEnv
    elif libero_type == "pro":
        import liberopro.liberopro as libero_pkg
        from liberopro.liberopro import benchmark as benchmark_module, get_libero_path
        from liberopro.liberopro.envs import OffScreenRenderEnv
    else:
        import libero.libero as libero_pkg
        from libero.libero import benchmark as benchmark_module, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

    package_root = pathlib.Path(libero_pkg.__file__).resolve().parent
    os.environ["LIBERO_TYPE"] = libero_type
    os.environ["LIBERO_BDDL_PATH"] = str(package_root / "bddl_files")
    os.environ["LIBERO_INIT_STATES_PATH"] = str(package_root / "init_files")

    if asset_root:
        resolved_asset_root = str(pathlib.Path(asset_root).resolve())
        os.environ["LIBERO_ASSET_ROOT"] = resolved_asset_root
        import robosuite.models as robosuite_models

        robosuite_models.assets_root = resolved_asset_root

    return benchmark_module, get_libero_path, OffScreenRenderEnv, package_root


def _build_suite_names(task_suite_name: str) -> list[str]:
    if task_suite_name == "all":
        return list(ALL_LIBERO_SUITES)
    return [task_suite_name]


def _load_taxonomy_lookup(
    package_root: pathlib.Path, suite_names: list[str]
) -> dict[str, dict[str, str]]:
    classification_path = package_root / "benchmark" / "task_classification.json"
    if not classification_path.exists():
        return {}

    raw_data = json.loads(classification_path.read_text())
    taxonomy_lookup: dict[str, dict[str, str]] = {}
    for suite_name in suite_names:
        suite_entries = raw_data.get(suite_name, [])
        taxonomy_lookup[suite_name] = {
            entry["name"]: LIBERO_PLUS_TAXONOMY_MAP.get(
                entry["category"], entry["category"]
            )
            for entry in suite_entries
        }
    return taxonomy_lookup


def _get_libero_env(task, resolution, seed, get_libero_path_fn, env_cls):
    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path_fn("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = env_cls(**env_args)
    env.seed(seed)
    return env, task_description


def _maybe_init_wandb(args, run_dir: pathlib.Path, suite_names: list[str], logger):
    if args.wandb_mode == "disabled":
        return None, None

    try:
        import wandb
    except ImportError:
        logger.warning("wandb is not installed; continuing without wandb logging.")
        return None, None

    wandb_dir = run_dir / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "libero_type": args.libero_type,
        "suite_names": suite_names,
        "num_trials_per_task": args.num_trials_per_task,
        "action_chunk": args.action_chunk,
        "num_steps": args.num_steps,
        "num_steps_wait": args.num_steps_wait,
        "save_only_failures": args.save_only_failures,
        "num_save_videos": args.num_save_videos,
        "repo_id": args.repo_id,
        "pretrained_path": args.pretrained_path,
        "task_offset": args.task_offset,
        "max_tasks": args.max_tasks,
    }

    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            group=args.wandb_group or None,
            name=args.wandb_name or args.exp_name,
            dir=str(wandb_dir),
            config=config,
            mode=args.wandb_mode,
        )
        return run, wandb
    except Exception as exc:
        logger.warning("wandb init failed (%s); continuing without wandb logging.", exc)
        return None, None


def _save_rollout_video(
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
    imageio.mimwrite(
        out_path,
        [np.asarray(x) for x in replay_images[:: max(1, video_temp_subsample)]],
        fps=fps,
    )
    return out_path


def _write_jsonl(records: list[dict[str, Any]], out_path: pathlib.Path):
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _append_jsonl_record(record: dict[str, Any], out_path: pathlib.Path):
    with out_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _load_task_manifest(
    manifest_path: str | None,
) -> tuple[dict[str, list[int]] | None, dict[str, Any] | None]:
    if not manifest_path:
        return None, None

    manifest = json.loads(pathlib.Path(manifest_path).read_text())
    task_ids_by_suite = manifest.get("task_ids_by_suite", manifest)
    normalized: dict[str, list[int]] = {}
    for suite_name, task_ids in task_ids_by_suite.items():
        normalized[suite_name] = [int(task_id) for task_id in task_ids]
    return normalized, manifest


def _resolve_task_indices(
    suite_name: str,
    num_tasks_in_suite: int,
    args,
    task_manifest: dict[str, list[int]] | None,
) -> list[int]:
    if task_manifest and suite_name in task_manifest:
        task_indices = list(task_manifest[suite_name])
    else:
        task_end = num_tasks_in_suite
        if args.max_tasks is not None:
            task_end = min(task_end, args.task_offset + args.max_tasks)
        task_indices = list(range(args.task_offset, task_end))

    for task_id in task_indices:
        if task_id < 0 or task_id >= num_tasks_in_suite:
            raise ValueError(
                f"Task id {task_id} is out of range for suite={suite_name} with {num_tasks_in_suite} tasks."
            )
    return task_indices


def _resolve_trial_indices(
    available_trials: int,
    num_trials: int,
    trial_sampling: str,
) -> list[int]:
    if num_trials <= 0 or available_trials <= 0:
        return []

    if trial_sampling == "sequential" or num_trials >= available_trials:
        return list(range(num_trials))

    if trial_sampling == "evenly_spaced":
        positions = np.linspace(0, available_trials - 1, num=num_trials)
        trial_indices = [int(round(position)) for position in positions]
        deduped = []
        seen = set()
        for trial_idx in trial_indices:
            if trial_idx not in seen:
                deduped.append(trial_idx)
                seen.add(trial_idx)
        for trial_idx in range(available_trials):
            if len(deduped) >= num_trials:
                break
            if trial_idx not in seen:
                deduped.append(trial_idx)
                seen.add(trial_idx)
        return deduped[:num_trials]

    raise ValueError(f"Unsupported trial_sampling={trial_sampling}")


def _build_summary(
    args,
    suite_names: list[str],
    total_episodes: int,
    total_successes: int,
    total_tasks_evaluated: int,
    taxonomy_stats,
    suite_stats,
    manifest_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    total_success_rate = total_successes / total_episodes if total_episodes else 0.0
    taxonomy_success_rates = {
        taxonomy: stats["successes"] / stats["episodes"]
        for taxonomy, stats in sorted(taxonomy_stats.items())
        if stats["episodes"] > 0
    }
    taxonomy_counts = {
        taxonomy: {
            "episodes": stats["episodes"],
            "successes": stats["successes"],
        }
        for taxonomy, stats in sorted(taxonomy_stats.items())
        if stats["episodes"] > 0
    }
    suite_success_rates = {
        suite_name: stats["successes"] / stats["episodes"]
        for suite_name, stats in sorted(suite_stats.items())
        if stats["episodes"] > 0
    }
    suite_counts = {
        suite_name: {
            "episodes": stats["episodes"],
            "successes": stats["successes"],
            "tasks": stats["tasks"],
        }
        for suite_name, stats in sorted(suite_stats.items())
        if stats["episodes"] > 0
    }

    return {
        "libero_type": args.libero_type,
        "suite_names": suite_names,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "total_success_rate": total_success_rate,
        "total_tasks_evaluated": total_tasks_evaluated,
        "taxonomy_success_rates": taxonomy_success_rates,
        "taxonomy_counts": taxonomy_counts,
        "suite_success_rates": suite_success_rates,
        "suite_counts": suite_counts,
        "task_offset": args.task_offset,
        "max_tasks": args.max_tasks,
        "num_trials_per_task": args.num_trials_per_task,
        "trial_sampling": args.trial_sampling,
        "task_manifest": args.task_manifest,
        "task_manifest_metadata": manifest_metadata,
    }


def _write_progress(
    task_results_path: pathlib.Path,
    summary_path: pathlib.Path,
    task_records: list[dict[str, Any]],
    summary: dict[str, Any],
):
    task_results_path.write_text(json.dumps(task_records, indent=2), encoding="utf-8")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main(args):
    if args.libero_type not in SUPPORTED_LIBERO_TYPES:
        raise ValueError(
            f"Unsupported libero_type={args.libero_type}. Expected one of {sorted(SUPPORTED_LIBERO_TYPES)}."
        )

    if args.num_trials_per_task is None:
        args.num_trials_per_task = 1 if args.libero_type == "plus" else 50

    suite_names = _build_suite_names(args.task_suite_name)
    logger = setup_logger(args.exp_name, args.log_dir)
    run_dir = pathlib.Path(args.log_dir) / args.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)
    task_results_path = run_dir / "task_results.json"
    episode_results_path = run_dir / "episode_results.jsonl"
    summary_path = run_dir / "summary.json"
    episode_results_path.write_text("", encoding="utf-8")

    np.random.seed(args.seed)

    benchmark_module, get_libero_path_fn, env_cls, package_root = _import_libero_stack(
        args.libero_type, args.asset_root
    )
    taxonomy_lookup = (
        _load_taxonomy_lookup(package_root, suite_names)
        if args.libero_type == "plus"
        else {}
    )
    task_manifest, manifest_metadata = _load_task_manifest(args.task_manifest)

    wandb_run, wandb_module = _maybe_init_wandb(args, run_dir, suite_names, logger)

    logger.info("policy setup start")
    policy = setup_policy(args)
    logger.info("policy setup done")

    total_episodes = 0
    total_successes = 0
    total_tasks_evaluated = 0
    saved_video_count = 0

    task_records = []
    taxonomy_stats = defaultdict(lambda: {"episodes": 0, "successes": 0})
    suite_stats = defaultdict(lambda: {"episodes": 0, "successes": 0, "tasks": 0})

    benchmark_dict = benchmark_module.get_benchmark_dict()

    for suite_name in suite_names:
        logger.info("Task suite: %s", suite_name)
        with contextlib.redirect_stdout(io.StringIO()):
            task_suite = benchmark_dict[suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        max_steps = SUITE_MAX_STEPS[suite_name]
        task_indices = _resolve_task_indices(
            suite_name=suite_name,
            num_tasks_in_suite=num_tasks_in_suite,
            args=args,
            task_manifest=task_manifest,
        )

        for task_id in tqdm.tqdm(
            task_indices,
            desc=f"Evaluating {suite_name}",
            leave=False,
        ):
            task = task_suite.get_task(task_id)
            task_name = pathlib.Path(task.bddl_file).stem
            task_description = task.language
            taxonomy_name = taxonomy_lookup.get(suite_name, {}).get(task_name, "Unknown")

            initial_states = task_suite.get_task_init_states(task_id)
            available_trials = len(initial_states)
            num_trials = min(args.num_trials_per_task, available_trials)
            if num_trials == 0:
                logger.warning(
                    "Skipping suite=%s task_id=%s (%s) because no initial states were found.",
                    suite_name,
                    task_id,
                    task_name,
                )
                continue
            if available_trials < args.num_trials_per_task:
                logger.warning(
                    "Requested %s trials for suite=%s task_id=%s but only %s are available.",
                    args.num_trials_per_task,
                    suite_name,
                    task_id,
                    available_trials,
                )
            selected_trial_indices = _resolve_trial_indices(
                available_trials=available_trials,
                num_trials=num_trials,
                trial_sampling=args.trial_sampling,
            )

            env, task_description = _get_libero_env(
                task, LIBERO_ENV_RESOLUTION, args.seed, get_libero_path_fn, env_cls
            )

            task_episodes = 0
            task_successes = 0
            for episode_idx, trial_idx in enumerate(selected_trial_indices):
                logger.info(
                    "Task suite=%s task_id=%s taxonomy=%s",
                    suite_name,
                    task_id,
                    taxonomy_name,
                )
                logger.info("Task description: %s", task_description)
                logger.info("Starting episode %s...", task_episodes + 1)

                policy.reset()
                env.reset()
                action_plan = collections.deque()
                obs = env.set_init_state(initial_states[trial_idx])

                replay_images = []
                success = False

                for t in range(max_steps + args.num_steps_wait):
                    if t < args.num_steps_wait:
                        obs, _reward, _done, _info = env.step(LIBERO_DUMMY_ACTION)
                        continue

                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(
                        obs["robot0_eye_in_hand_image"][::-1, ::-1]
                    )
                    replay_images.append(img)

                    state = np.concatenate(
                        (
                            obs["robot0_eef_pos"],
                            _quat2axisangle(obs["robot0_eef_quat"]),
                            obs["robot0_gripper_qpos"],
                        )
                    )

                    if not action_plan:
                        observation = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": state,
                            "prompt": str(task_description),
                        }
                        action_chunk = policy.infer(observation)["actions"]
                        assert len(action_chunk) >= args.action_chunk, (
                            f"We want to replan every {args.action_chunk} steps, "
                            f"but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.action_chunk])

                    action = action_plan.popleft()
                    obs, _reward, done, _info = env.step(action.tolist())
                    if done:
                        success = True
                        task_successes += 1
                        total_successes += 1
                        break

                task_episodes += 1
                total_episodes += 1

                maybe_save_video = (
                    args.num_save_videos > 0 and saved_video_count < args.num_save_videos
                )
                if args.save_only_failures and success:
                    maybe_save_video = False

                video_path = None
                if maybe_save_video:
                    video_path = _save_rollout_video(
                        run_dir=run_dir,
                        suite_name=suite_name,
                        task_id=task_id,
                        task_name=task_name,
                        episode_idx=episode_idx,
                        success=success,
                        replay_images=replay_images,
                        video_temp_subsample=args.video_temp_subsample,
                    )
                    if video_path is not None:
                        saved_video_count += 1

                episode_record = {
                    "suite_name": suite_name,
                    "task_id": task_id,
                    "task_name": task_name,
                    "task_description": task_description,
                    "taxonomy": taxonomy_name,
                    "episode_idx": episode_idx,
                    "trial_idx": trial_idx,
                    "success": success,
                    "video_path": str(video_path) if video_path else None,
                }
                _append_jsonl_record(episode_record, episode_results_path)

                taxonomy_stats[taxonomy_name]["episodes"] += 1
                taxonomy_stats[taxonomy_name]["successes"] += int(success)
                suite_stats[suite_name]["episodes"] += 1
                suite_stats[suite_name]["successes"] += int(success)

                logger.info("Success: %s", success)
                logger.info("# episodes completed so far: %s", total_episodes)
                logger.info(
                    "# successes: %s (%.1f%%)",
                    total_successes,
                    (total_successes / total_episodes * 100.0) if total_episodes else 0.0,
                )

                if wandb_run is not None:
                    payload = {
                        "eval/episode_success": float(success),
                        "eval/running_success_rate": total_successes / total_episodes,
                        "eval/episode_index": total_episodes,
                    }
                    if taxonomy_name != "Unknown":
                        payload[f"eval/taxonomy_running/{taxonomy_name}"] = (
                            taxonomy_stats[taxonomy_name]["successes"]
                            / taxonomy_stats[taxonomy_name]["episodes"]
                        )
                    if (
                        video_path is not None
                        and args.wandb_log_videos
                        and wandb_module is not None
                    ):
                        media_key = "eval/failure_video" if not success else "eval/success_video"
                        payload[media_key] = wandb_module.Video(str(video_path), format="mp4")
                    wandb_run.log(payload, step=total_episodes)

            env.close()
            total_tasks_evaluated += 1
            suite_stats[suite_name]["tasks"] += 1

            task_success_rate = task_successes / task_episodes if task_episodes else 0.0
            task_summary = {
                "suite_name": suite_name,
                "task_id": task_id,
                "task_name": task_name,
                "task_description": task_description,
                "taxonomy": taxonomy_name,
                "episodes": task_episodes,
                "successes": task_successes,
                "success_rate": task_success_rate,
            }
            task_records.append(task_summary)
            summary = _build_summary(
                args=args,
                suite_names=suite_names,
                total_episodes=total_episodes,
                total_successes=total_successes,
                total_tasks_evaluated=total_tasks_evaluated,
                taxonomy_stats=taxonomy_stats,
                suite_stats=suite_stats,
                manifest_metadata=manifest_metadata,
            )
            _write_progress(task_results_path, summary_path, task_records, summary)

            logger.info(
                "Task: %s, Successes: %s/%s, Success Rate: %.2f%%",
                task_description,
                task_successes,
                task_episodes,
                task_success_rate * 100.0,
            )

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval/task_success_rate": task_success_rate,
                        "eval/tasks_completed": total_tasks_evaluated,
                    },
                    step=total_episodes,
                )

    summary = _build_summary(
        args=args,
        suite_names=suite_names,
        total_episodes=total_episodes,
        total_successes=total_successes,
        total_tasks_evaluated=total_tasks_evaluated,
        taxonomy_stats=taxonomy_stats,
        suite_stats=suite_stats,
        manifest_metadata=manifest_metadata,
    )
    _write_progress(task_results_path, summary_path, task_records, summary)
    total_success_rate = summary["total_success_rate"]
    taxonomy_success_rates = summary["taxonomy_success_rates"]
    suite_success_rates = summary["suite_success_rates"]

    logger.info("===============")
    logger.info("Per-Suite Success Rate:")
    for suite_name, rate in suite_success_rates.items():
        logger.info("%s: %.2f%%", suite_name, rate * 100.0)

    if taxonomy_success_rates:
        logger.info("Per-Taxonomy Success Rate:")
        for taxonomy_name, rate in taxonomy_success_rates.items():
            logger.info("%s: %.2f%%", taxonomy_name, rate * 100.0)

    logger.info(
        "Total Success Rate: %s/%s = %.2f%%",
        total_successes,
        total_episodes,
        total_success_rate * 100.0,
    )
    logger.info("results/total_success_rate: %s", total_success_rate)
    logger.info("results/total_episodes: %s", total_episodes)
    logger.info("results/total_successes: %s", total_successes)
    logger.info("results/task_results_path: %s", task_results_path)
    logger.info("results/episode_results_path: %s", episode_results_path)
    logger.info("results/summary_path: %s", summary_path)

    if wandb_run is not None:
        taxonomy_table = None
        if taxonomy_success_rates and wandb_module is not None:
            taxonomy_table = wandb_module.Table(
                columns=["taxonomy", "episodes", "successes", "success_rate"],
                data=[
                    [
                        taxonomy_name,
                        taxonomy_stats[taxonomy_name]["episodes"],
                        taxonomy_stats[taxonomy_name]["successes"],
                        taxonomy_success_rates[taxonomy_name],
                    ]
                    for taxonomy_name in sorted(taxonomy_success_rates)
                ],
            )

        suite_table = None
        if wandb_module is not None:
            suite_table = wandb_module.Table(
                columns=["suite_name", "tasks", "episodes", "successes", "success_rate"],
                data=[
                    [
                        suite_name,
                        suite_stats[suite_name]["tasks"],
                        suite_stats[suite_name]["episodes"],
                        suite_stats[suite_name]["successes"],
                        suite_success_rates[suite_name],
                    ]
                    for suite_name in sorted(suite_success_rates)
                ],
            )

        final_payload = {
            "results/total_success_rate": total_success_rate,
            "results/total_episodes": total_episodes,
            "results/total_successes": total_successes,
        }
        for taxonomy_name, rate in taxonomy_success_rates.items():
            final_payload[f"results/taxonomy/{taxonomy_name}"] = rate
        for suite_name, rate in suite_success_rates.items():
            final_payload[f"results/suite/{suite_name}"] = rate
        if taxonomy_table is not None:
            final_payload["results/taxonomy_table"] = taxonomy_table
        if suite_table is not None:
            final_payload["results/suite_table"] = suite_table
        wandb_run.log(final_payload, step=total_episodes)
        wandb_run.summary.update(
            {
                "results/total_success_rate": total_success_rate,
                "results/total_episodes": total_episodes,
                "results/total_successes": total_successes,
                **{
                    f"results/taxonomy/{taxonomy_name}": rate
                    for taxonomy_name, rate in taxonomy_success_rates.items()
                },
                **{
                    f"results/suite/{suite_name}": rate
                    for suite_name, rate in suite_success_rates.items()
                },
            }
        )
        wandb_run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        default="logs",
        help="Directory to save log files and artifacts.",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default="libero_spatial_pi05",
        help="Experiment name used for log files and artifact directories.",
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
        default=None,
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
        help="Optional explicit norm stats directory. Useful when the checkpoint does not bundle the target dataset stats.",
    )
    parser.add_argument(
        "--task_suite_name",
        type=str,
        default="libero_spatial",
        choices=["all", "libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"],
        help="Task suite to evaluate. Use 'all' to evaluate all 10-task suites sequentially.",
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
        "--num_trials_per_task",
        type=int,
        default=None,
        help="Number of rollouts per task. Defaults to 1 for LIBERO-plus and 50 otherwise.",
    )
    parser.add_argument(
        "--trial_sampling",
        type=str,
        default="sequential",
        choices=["sequential", "evenly_spaced"],
        help="How to choose trial indices when evaluating fewer trials than are available.",
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
        "--num_save_videos",
        type=int,
        default=0,
        help="Maximum number of rollout videos to save.",
    )
    parser.add_argument(
        "--save_only_failures",
        action="store_true",
        help="Only save failure trajectories when writing videos.",
    )
    parser.add_argument(
        "--wandb_log_videos",
        action="store_true",
        help="Upload saved videos to wandb in addition to writing them locally.",
    )
    parser.add_argument(
        "--video_temp_subsample",
        type=int,
        default=10,
        help="Save every Nth frame to each video.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed.",
    )
    parser.add_argument(
        "--task_offset",
        type=int,
        default=0,
        help="Start evaluating from this task index within each suite.",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Optional number of tasks to evaluate per suite.",
    )
    parser.add_argument(
        "--task_manifest",
        type=str,
        default=None,
        help="Optional JSON file with explicit zero-based task ids to evaluate per suite.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="libero-plus-eval",
        help="wandb project name.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="wandb entity.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="wandb group.",
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="wandb run name.",
    )
    parser.add_argument(
        "--wandb_mode",
        type=str,
        default="online",
        choices=["online", "offline", "disabled"],
        help="wandb logging mode.",
    )
    main(parser.parse_args())

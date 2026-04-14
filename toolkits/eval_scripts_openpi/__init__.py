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

import contextlib
import dataclasses
import json
import logging
import os
import pathlib
from typing import Any

import openpi.policies.policy as _policy
import openpi.shared.download as download
import openpi.shared.normalize as _normalize
import openpi.transforms as transforms
import safetensors
from openpi.models_pytorch import pi0_pytorch
from openpi.training import checkpoints as _checkpoints
from openpi.training import config as _config

from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config


def setup_logger(exp_name, log_dir):
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{exp_name}.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[logging.FileHandler(log_file, mode="w"), logging.StreamHandler()],
        force=True,
    )
    logger = logging.getLogger(__name__)
    return logger


@contextlib.contextmanager
def _maybe_disable_torch_compile():
    disable_compile = os.environ.get("RLINF_OPENPI_DISABLE_TORCH_COMPILE", "1") == "1"
    if not disable_compile:
        yield
        return

    import torch

    original_compile = getattr(torch, "compile", None)
    if original_compile is None:
        yield
        return

    logging.info("Disabling torch.compile while constructing the OpenPI PyTorch model.")

    def _identity_compile(model_or_fn, *args, **kwargs):
        return model_or_fn

    torch.compile = _identity_compile
    try:
        yield
    finally:
        torch.compile = original_compile


def load_pytorch(train_config, weight_path: str):
    with _maybe_disable_torch_compile():
        model = pi0_pytorch.PI0Pytorch(config=train_config.model)
    if weight_path.endswith(".pt"):
        import torch

        model_state_dict = torch.load(weight_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(model_state_dict, strict=False)
    else:
        model_state_dict = safetensors.torch.load_file(weight_path, device="cpu")
        prefixed_keys = sum(key.startswith("model.") for key in model_state_dict)
        if prefixed_keys > len(model_state_dict) // 2:
            model_state_dict = {
                key[len("model.") :] if key.startswith("model.") else key: value
                for key, value in model_state_dict.items()
            }
            logging.info(
                "Stripped top-level 'model.' prefix from %s checkpoint keys.",
                prefixed_keys,
            )
        missing, unexpected = model.load_state_dict(model_state_dict, strict=False)

    if missing or unexpected:
        logging.info(
            "Checkpoint load finished with %d missing and %d unexpected keys.",
            len(missing),
            len(unexpected),
        )
        if missing:
            logging.info("Missing keys sample: %s", missing[:10])
        if unexpected:
            logging.info("Unexpected keys sample: %s", unexpected[:10])
    return model


def _resolve_pytorch_weight_path(checkpoint_dir: str) -> str | None:
    candidates = [
        os.path.join(checkpoint_dir, "model.safetensors"),
        os.path.join(checkpoint_dir, "model_state_dict", "full_weights.pt"),
        os.path.join(checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def _load_json_if_exists(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _resolve_checkpoint_compat_metadata(
    checkpoint_dir: pathlib.Path | str,
) -> dict[str, Any]:
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    config = _load_json_if_exists(checkpoint_dir / "config.json") or {}
    preprocessor = _load_json_if_exists(checkpoint_dir / "policy_preprocessor.json") or {}

    normalization_mapping = config.get("normalization_mapping", {}) or {}
    state_norm = str(normalization_mapping.get("STATE", "")).upper()
    action_norm = str(normalization_mapping.get("ACTION", "")).upper()

    discrete_state_input = config.get("type") == "pi05"
    for step in preprocessor.get("steps", []):
        if step.get("registry_name") == "pi05_prepare_state_tokenizer_processor_step":
            discrete_state_input = True
            break

    use_quantile_norm = None
    if state_norm or action_norm:
        use_quantile_norm = any(
            mode == "QUANTILES" for mode in (state_norm, action_norm)
        )

    return {
        "checkpoint_type": config.get("type"),
        "chunk_size": config.get("chunk_size"),
        "num_inference_steps": config.get("num_inference_steps"),
        "discrete_state_input": discrete_state_input,
        "use_quantile_norm": use_quantile_norm,
    }


def _find_checkpoint_normalizer_state_file(
    checkpoint_dir: pathlib.Path | str,
) -> pathlib.Path | None:
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    preprocessor = _load_json_if_exists(checkpoint_dir / "policy_preprocessor.json") or {}
    for step in preprocessor.get("steps", []):
        if step.get("registry_name") != "normalizer_processor":
            continue
        state_file = step.get("state_file")
        if state_file:
            candidate = checkpoint_dir / state_file
            if candidate.exists():
                return candidate

    fallback = checkpoint_dir / "policy_preprocessor_step_2_normalizer_processor.safetensors"
    if fallback.exists():
        return fallback
    return None


def _load_lerobot_checkpoint_norm_stats(
    checkpoint_dir: pathlib.Path | str,
) -> dict[str, _normalize.NormStats] | None:
    stats_file = _find_checkpoint_normalizer_state_file(checkpoint_dir)
    if stats_file is None:
        return None

    raw_stats = safetensors.torch.load_file(str(stats_file), device="cpu")
    key_map = {
        "state": "observation.state",
        "actions": "action",
    }

    norm_stats: dict[str, _normalize.NormStats] = {}
    for output_key, checkpoint_prefix in key_map.items():
        mean_key = f"{checkpoint_prefix}.mean"
        std_key = f"{checkpoint_prefix}.std"
        if mean_key not in raw_stats or std_key not in raw_stats:
            continue

        stats_kwargs: dict[str, Any] = {
            "mean": raw_stats[mean_key].cpu().numpy(),
            "std": raw_stats[std_key].cpu().numpy(),
        }
        for quantile_key in ("q01", "q99"):
            full_key = f"{checkpoint_prefix}.{quantile_key}"
            if full_key in raw_stats:
                stats_kwargs[quantile_key] = raw_stats[full_key].cpu().numpy()

        norm_stats[output_key] = _normalize.NormStats(**stats_kwargs)

    return norm_stats or None


def _apply_checkpoint_compat_overrides(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
) -> tuple[_config.TrainConfig, dict[str, Any]]:
    metadata = _resolve_checkpoint_compat_metadata(checkpoint_dir)
    model_config = train_config.model
    override_kwargs: dict[str, Any] = {}

    if metadata.get("discrete_state_input") and hasattr(model_config, "discrete_state_input"):
        override_kwargs["discrete_state_input"] = True

    if metadata.get("chunk_size") and hasattr(model_config, "action_horizon"):
        override_kwargs["action_horizon"] = int(metadata["chunk_size"])

    if override_kwargs:
        model_config = dataclasses.replace(model_config, **override_kwargs)
        train_config = dataclasses.replace(train_config, model=model_config)

    return train_config, metadata


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    norm_stats_dir: pathlib.Path | str | None = None,
    pytorch_device: str | None = None,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint."""
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = pathlib.Path(download.maybe_download(str(checkpoint_dir)))
    train_config, checkpoint_metadata = _apply_checkpoint_compat_overrides(
        train_config, checkpoint_dir
    )

    weight_path = _resolve_pytorch_weight_path(checkpoint_dir)
    is_pytorch = weight_path is not None

    logging.info("Loading model...")
    if is_pytorch:
        model = load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        raise AssertionError("Only PyTorch models are supported for now")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    use_quantile_norm = data_config.use_quantile_norm
    if checkpoint_metadata.get("use_quantile_norm") is not None:
        use_quantile_norm = bool(checkpoint_metadata["use_quantile_norm"])
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        try:
            norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
        except FileNotFoundError:
            last_err: Exception | None = None

            if norm_stats_dir is not None:
                try:
                    logging.info("Falling back to explicit norm stats dir %s", norm_stats_dir)
                    norm_stats = _normalize.load(norm_stats_dir)
                except FileNotFoundError as err:
                    last_err = err
                    try:
                        norm_stats = _checkpoints.load_norm_stats(
                            norm_stats_dir, data_config.asset_id
                        )
                    except FileNotFoundError as err2:
                        last_err = err2

            if norm_stats is None:
                bundled_norm_stats = _load_lerobot_checkpoint_norm_stats(checkpoint_dir)
                if bundled_norm_stats is not None:
                    logging.info(
                        "Using bundled LeRobot processor norm stats from %s",
                        checkpoint_dir,
                    )
                    norm_stats = bundled_norm_stats

            if norm_stats is None:
                hf_lerobot_home = os.environ.get("HF_LEROBOT_HOME")
                if hf_lerobot_home:
                    try:
                        norm_stats = _checkpoints.load_norm_stats(
                            hf_lerobot_home, data_config.asset_id
                        )
                    except FileNotFoundError as err:
                        last_err = err

            if norm_stats is None:
                searched = [f"{checkpoint_dir}/{data_config.asset_id}/norm_stats.json"]
                if norm_stats_dir is not None:
                    searched.append(f"{pathlib.Path(norm_stats_dir)}/norm_stats.json")
                    searched.append(
                        f"{pathlib.Path(norm_stats_dir)}/{data_config.asset_id}/norm_stats.json"
                    )
                hf_lerobot_home = os.environ.get("HF_LEROBOT_HOME")
                if hf_lerobot_home:
                    searched.append(f"{hf_lerobot_home}/{data_config.asset_id}/norm_stats.json")
                raise FileNotFoundError(
                    "Norm stats file not found. Searched: "
                    + ", ".join(str(path) for path in searched)
                ) from last_err

    if is_pytorch and pytorch_device is None:
        import torch

        if torch.cuda.is_available():
            pytorch_device = "cuda"
        elif hasattr(torch, "npu") and torch.npu.is_available():
            pytorch_device = "npu"
        else:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(
                norm_stats, use_quantiles=use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=use_quantile_norm
            ),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )


def setup_policy(args):
    data_kwargs = None
    if getattr(args, "repo_id", None):
        data_kwargs = {"repo_id": args.repo_id}
    config = get_openpi_config(args.config_name, data_kwargs=data_kwargs)
    policy = create_trained_policy(
        config,
        args.pretrained_path,
        sample_kwargs={"num_steps": args.num_steps},
        norm_stats_dir=getattr(args, "norm_stats_dir", None),
    )
    return policy

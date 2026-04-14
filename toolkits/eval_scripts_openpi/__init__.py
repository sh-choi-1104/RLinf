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
        model.load_state_dict(model_state_dict, strict=False)
    else:
        safetensors.torch.load_model(model, weight_path, strict=False)
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
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    weight_path = _resolve_pytorch_weight_path(checkpoint_dir)
    is_pytorch = weight_path is not None

    logging.info("Loading model...")
    if is_pytorch:
        model = load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    else:
        raise AssertionError("Only PyTorch models are supported for now")

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        try:
            norm_stats = _checkpoints.load_norm_stats(checkpoint_dir, data_config.asset_id)
        except FileNotFoundError:
            if norm_stats_dir is None:
                raise
            logging.info("Falling back to norm stats from %s", norm_stats_dir)
            norm_stats = _normalize.load(norm_stats_dir)

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
                norm_stats, use_quantiles=data_config.use_quantile_norm
            ),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(
                norm_stats, use_quantiles=data_config.use_quantile_norm
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

# Copyright 2026 The RLinf Authors.
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

import os
import pathlib
from typing import Any, Callable

import numpy as np


def _get_openpi_modules():
    import openpi.models.model as _model
    import openpi.shared.normalize as normalize
    import openpi.training.data_loader as _data_loader
    import openpi.transforms as transforms

    return _model, normalize, _data_loader, transforms


def _get_config_loader() -> Callable[..., Any]:
    from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

    return get_openpi_config


class RemoveStrings:
    def __call__(self, x: dict) -> dict:
        return {
            k: v
            for k, v in x.items()
            if not np.issubdtype(np.asarray(v).dtype, np.str_)
        }


def create_torch_dataloader(
    data_config: Any,
    action_horizon: int,
    batch_size: int,
    model_config: Any,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[Any, int]:
    _, _, _data_loader, _ = _get_openpi_modules()
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(
        data_config, action_horizon, model_config
    )
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: Any,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[Any, int]:
    _, _, _data_loader, _ = _get_openpi_modules()
    dataset = _data_loader.create_rlds_dataset(
        data_config, action_horizon, batch_size, shuffle=False
    )
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            # Remove strings since they are not supported by JAX and are not needed to compute norm stats.
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        # NOTE: this length is currently hard-coded for DROID.
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(
        dataset,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def build_train_config(
    config_name: str,
    repo_id: str,
    model_path: str | None = None,
) -> Any:
    """Build an OpenPI train config with an optional model checkpoint root."""
    get_openpi_config = _get_config_loader()
    return get_openpi_config(
        config_name,
        model_path=model_path,
        data_kwargs={"repo_id": repo_id},
    )


def resolve_norm_stats_output_path(
    assets_root: str | os.PathLike[str],
    repo_id: str,
) -> pathlib.Path:
    """Resolve the directory that will store the generated normalization stats."""
    return pathlib.Path(assets_root) / repo_id


def main(
    config_name: str,
    repo_id: str,
    model_path: str | None = None,
):
    import tqdm

    if os.path.isabs(repo_id):
        raise ValueError(
            "repo_id must be a LeRobot dataset name such as 'libero_plus' or "
            "'namespace/libero_plus', not an absolute path. Set HF_LEROBOT_HOME "
            "to the dataset root and pass only the dataset directory name here."
        )

    if not os.environ.get("HF_LEROBOT_HOME"):
        raise EnvironmentError(
            "HF_LEROBOT_HOME must be set before running this script. "
            "Export it manually, for example: "
            "export HF_LEROBOT_HOME=/path/to/lerobot_root"
        )
    _, normalize, _, _ = _get_openpi_modules()
    config = build_train_config(config_name, repo_id, model_path=model_path)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config,
            config.model.action_horizon,
            config.batch_size,
            config.model,
            config.num_workers,
        )

    keys = ["state", "actions"]
    stats = {key: normalize.RunningStats() for key in keys}

    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Computing stats"):
        for key in keys:
            stats[key].update(np.asarray(batch[key]))

    norm_stats = {key: stats.get_statistics() for key, stats in stats.items()}

    output_path = resolve_norm_stats_output_path(config.assets_dirs, data_config.repo_id)
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)


if __name__ == "__main__":
    import tyro

    tyro.cli(main)

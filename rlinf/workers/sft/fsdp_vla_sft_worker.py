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
import json
import os
import shutil
from typing import Any

import torch
from omegaconf import DictConfig
from torch.utils import _pytree

from rlinf.config import SupportedModel
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.utils.pytree import register_pytree_dataclasses
from rlinf.workers.sft.fsdp_sft_worker import FSDPSftWorker


class FSDPVlaSftWorker(FSDPSftWorker):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)

    def _use_openpi_weight_only_checkpoint(self) -> bool:
        return (
            SupportedModel(self.cfg.actor.model.model_type) == SupportedModel.OPENPI
            and str(self.cfg.actor.fsdp_config.get("sharding_strategy", "")).lower()
            == "no_shard"
            and bool(self.cfg.actor.fsdp_config.get("use_orig_params", False))
        )

    def _save_data_state(self, save_path: str, step: int) -> None:
        state = {
            "global_step": int(step),
            "data_epoch": int(self._data_epoch),
            "data_iter_offset": int(self._data_iter_offset),
        }
        with open(os.path.join(save_path, "trainer_state.json"), "w") as f:
            json.dump(state, f)

    def _restore_data_state(self, load_path: str) -> None:
        state_path = os.path.join(load_path, "trainer_state.json")
        if not os.path.exists(state_path):
            return
        with open(state_path, "r") as f:
            state = json.load(f)
        self._data_epoch = int(state.get("data_epoch", 0))
        self._data_iter_offset = int(state.get("data_iter_offset", 0))
        if hasattr(self.data_loader, "sampler") and hasattr(
            self.data_loader.sampler, "set_epoch"
        ):
            self.data_loader.sampler.set_epoch(self._data_epoch)
        self.data_iter = iter(self.data_loader)
        for _ in range(self._data_iter_offset):
            try:
                next(self.data_iter)
            except StopIteration:
                self._data_epoch += 1
                if hasattr(self.data_loader, "sampler") and hasattr(
                    self.data_loader.sampler, "set_epoch"
                ):
                    self.data_loader.sampler.set_epoch(self._data_epoch)
                self.data_iter = iter(self.data_loader)

    def save_checkpoint(self, save_path: str, step: int = 0):
        if not self._use_openpi_weight_only_checkpoint():
            return super().save_checkpoint(save_path, step)

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

        if self._rank == 0:
            os.makedirs(save_path, exist_ok=True)
            model_state_dir = os.path.join(save_path, "model_state_dict")
            os.makedirs(model_state_dir, exist_ok=True)

            module = self.model.module if hasattr(self.model, "module") else self.model
            model_state_dict = {
                key: value.detach().cpu().clone()
                if isinstance(value, torch.Tensor)
                else value
                for key, value in module.state_dict().items()
            }
            torch.save(model_state_dict, os.path.join(model_state_dir, "full_weights.pt"))
            self._save_data_state(save_path, step)

            base_model_dir = str(self.cfg.actor.model.model_path)
            for filename in [
                "config.json",
                "policy_preprocessor.json",
                "policy_postprocessor.json",
                "README.md",
            ]:
                src = os.path.join(base_model_dir, filename)
                if os.path.exists(src):
                    shutil.copy2(src, os.path.join(save_path, filename))

            asset_id = getattr(getattr(self, "data_config", None), "asset_id", None)
            if asset_id:
                src = os.path.join(base_model_dir, asset_id, "norm_stats.json")
                if os.path.exists(src):
                    dst_dir = os.path.join(save_path, asset_id)
                    os.makedirs(dst_dir, exist_ok=True)
                    shutil.copy2(src, os.path.join(dst_dir, "norm_stats.json"))

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.barrier()

    def load_checkpoint(self, load_path: str):
        if self._use_openpi_weight_only_checkpoint():
            full_weights_path = os.path.join(load_path, "model_state_dict", "full_weights.pt")
            if os.path.exists(full_weights_path):
                module = self.model.module if hasattr(self.model, "module") else self.model
                model_state_dict = torch.load(full_weights_path, map_location="cpu")
                module.load_state_dict(model_state_dict, strict=False)
                self._restore_data_state(load_path)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.barrier()
                return
        return super().load_checkpoint(load_path)

    def build_dataloader(self, data_paths: list[str], eval_dataset: bool = False):
        if SupportedModel(self.cfg.actor.model.model_type) in [SupportedModel.OPENPI]:
            import openpi.training.data_loader as openpi_data_loader

            from rlinf.models.embodiment.openpi.dataconfig import get_openpi_config

            data_kwargs = getattr(self.cfg.actor.model, "openpi_data", None) or getattr(
                self.cfg.actor, "openpi_data", None
            )
            config = get_openpi_config(
                self.cfg.actor.model.openpi.config_name,
                model_path=self.cfg.actor.model.model_path,
                batch_size=self.cfg.actor.micro_batch_size * self._world_size,
                data_kwargs=data_kwargs,
            )
            data_loader = openpi_data_loader.create_data_loader(
                config, framework="pytorch", shuffle=True
            )
            return data_loader, data_loader.data_config()
        elif SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA
        ]:
            from rlinf.models.embodiment.lingbotvla.sft_builder import (
                build_lingbot_sft_dataloader,
            )

            return build_lingbot_sft_dataloader(
                self.cfg, self._world_size, self._rank, data_paths
            )
        else:
            raise KeyError(
                f"not support such model type {self.cfg.actor.model.model_type} for SFT right now."
            )

    def get_eval_model_output(self, batch: dict[str, Any]):
        # now the eval is not supported for embodied sft
        raise NotImplementedError("eval is not supported for embodied sft right now.")

    def get_train_model_output(self, batch: dict[str, Any]):
        if SupportedModel(self.cfg.actor.model.model_type) in [
            SupportedModel.LINGBOTVLA
        ]:
            batch_data = _pytree.tree_map(
                lambda x: (
                    torch.as_tensor(x, device=self.device).contiguous().clone()
                    if isinstance(x, torch.Tensor)
                    else x
                ),
                batch,
            )
            with self.amp_context:
                losses_dict = self.model(forward_type=ForwardType.SFT, data=batch_data)
            return losses_dict["loss"]
        observation, actions = batch

        register_pytree_dataclasses(observation)
        observation = _pytree.tree_map(
            lambda x: (
                torch.as_tensor(x, device=self.device).contiguous().clone()
                if x is not None
                else x
            ),
            observation,
        )
        actions = actions.to(torch.float32)
        actions = actions.to(self.device)

        with self.amp_context:
            losses = self.model(
                forward_type=ForwardType.SFT,
                data={"observation": observation, "actions": actions},
            )

        # train model return the loss
        return losses

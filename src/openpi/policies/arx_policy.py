import random
import dataclasses
import einops
import numpy as np
from typing import ClassVar

from openpi import transforms
from openpi.models import model as _model


def make_arx_example() -> dict:
    """Creates a random input example for the ARX policy."""
    return {
        "state": np.random.rand(14),
        "image": {
            "left_wrist_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "face_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
            "right_wrist_view": np.random.randint(256, size=(3, 224, 224), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class ArxInputs(transforms.DataTransformFn):
    """Transform inputs for the ARX policy."""

    mode: str = "s2s"  # "s2s", "s2m", "m2m", "sm2m", "sm2sm", "smp2smp"
    action_dim: int = 32
    model_type: _model.ModelType = _model.ModelType.PI0
    state_history_size: int = 0
    state_future_size: int = 0
    slave_state_dim: int = 14
    mask_history_slave_states: bool = False
    random_drop_master: float = 0.
    random_drop_history: float = 0.
    random_drop_future: float = 0.
    random_drop_label: float = 0.
    random_drop_label_global: bool = True
    random_pos_offset: float = 0.
    only_right_obs: bool = False
    mask_left_obs: bool = False
    project_from_sm2sm: bool = False

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("left_wrist_view", "face_view", "right_wrist_view")

    def __call__(self, data: dict) -> dict:
        state = data["state"]
        
        if state.ndim == 2:
            assert state.shape[0] == self.state_history_size + 1 + self.state_future_size
            state, master_mask = self._mask_states(state)
            state = transforms.pad_to_dim(state, self.action_dim)
            state[:, -1] = master_mask
        else:
            if random.random() < self.random_drop_master:
                state[self.slave_state_dim:] = state[:self.slave_state_dim]
                state = transforms.pad_to_dim(state, self.action_dim)
                state[-1] = 1.
            
        state = transforms.pad_to_dim(state, self.action_dim)
            
        def convert_image(img):
            img = np.asarray(img)
            # Convert to uint8 if using float images.
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            # Convert from [channel, height, width] to [height, width, channel].
            if img.shape[-1] != 3:
                output_image = einops.rearrange(img, "c h w -> h w c")
            else:
                output_image = img
            assert output_image.shape[-1] == 3, f"Image must have 3 channels, got {output_image.shape}."
            return output_image

        # Convert images to uint8 and rearrange to (H,W,C) format
        for key in self.EXPECTED_CAMERAS:
            assert key in data['images'].keys(), f"Images must contain {key}."
            data['images'][key] = convert_image(data['images'][key])

        inputs = {
            "image": {
                "base_0_rgb": data['images']['face_view'],
                "left_wrist_0_rgb": data['images']['left_wrist_view'],
                "right_wrist_0_rgb": data['images']['right_wrist_view'],
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": state,
        }

        if "actions" in data:
            inputs["actions"] = transforms.pad_to_dim(data["actions"], self.action_dim)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if "actions_is_pad" in data:
            inputs["actions_is_pad"] = data["actions_is_pad"]

        # random position offset augmentation
        if self.random_pos_offset > 0.:
            pos_offset = (np.random.rand(3) * 2 - 1.) * self.random_pos_offset
            inputs["state"][..., 7:10] += pos_offset
            inputs["actions"][..., 7:10] += pos_offset
            if self.project_from_sm2sm or self.mode in ["sm2m", "sm2sm", "smp2smp"]:
                inputs["state"][..., 21:24] += pos_offset
            if self.project_from_sm2sm or self.mode in ["sm2sm", "smp2smp"]:
                inputs["actions"][..., 21:24] += pos_offset
        
        if self.only_right_obs or self.mask_left_obs:
            inputs["image_mask"]["left_wrist_0_rgb"] = np.False_
            if self.slave_state_dim == 14:  # (left + right) x (pos + rot + gripper)
                inputs["state"][..., :7] = 0.
                if self.project_from_sm2sm or self.mode in ["sm2m", "sm2sm", "smp2smp"]:
                    inputs["state"][..., 14:21] = 0.
                if "actions" in inputs:
                    inputs["actions"][..., :7] = 0.
                    if self.project_from_sm2sm or self.mode in ["sm2sm", "smp2smp"]:
                        inputs["actions"][..., 14:21] = 0.
                        
            if self.only_right_obs:
                inputs["image_mask"]["base_0_rgb"] = np.False_

        if random.random() < self.random_drop_label:
            if self.random_drop_label_global or (inputs["state"][0, 28] != inputs["actions"][10, 28]):  # todo 定义 transition more robustly
                state[:, 28] = 0

        return inputs

    def _mask_states(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Mask future slave states by copying current slave state."""
        state = np.asarray(state).copy()
        current_idx = self.state_history_size
        current_slave = state[current_idx, :self.slave_state_dim]
        current_state = state[current_idx]
        if state.shape[-1] == 32:
            master_mask = state[:, -1]
        else:
            master_mask = np.zeros((state.shape[0],), dtype=state.dtype)

        if self.state_future_size > 0:  # always mask future slave states
            state[current_idx + 1:, :self.slave_state_dim] = current_slave
        if self.mask_history_slave_states and self.state_history_size > 0:
            state[:current_idx, :self.slave_state_dim] = current_slave

        # Data augmentation: randomly drop history/future/master states
        if random.random() < self.random_drop_master:
            state[:, self.slave_state_dim:self.slave_state_dim*2] = current_slave
            master_mask[:] = 1.
        if random.random() < self.random_drop_history and self.state_history_size > 0:
            state[:current_idx] = current_state
            master_mask[:current_idx] = 1.
        if random.random() < self.random_drop_future and self.state_future_size > 0:
            mask_size = random.randint(1, self.state_future_size)
            state[-mask_size:] = state[-mask_size - 1]
            master_mask[-mask_size:] = 1.

        if self.mode == "smp2smp":
            state[:, 28] = state[current_idx, 28]
            
        return state, master_mask


@dataclasses.dataclass(frozen=True)
class ProjectNormalizedSm2sm(transforms.DataTransformFn):
    """Project normalized full SM2SM state/actions into a policy mode."""

    mode: str
    action_dim: int = 32
    state_history_size: int = 0
    state_future_size: int = 0
    slave_state_dim: int = 14

    def __call__(self, data: dict) -> dict:
        state = np.asarray(data["state"])
        self._validate_dim(state, "state")
        state_mask = state[..., -1].copy() if state.shape[-1] == self.action_dim else None
        input_mode, output_mode = self.mode.split("2", maxsplit=1)

        if input_mode == "s":
            projected_state = state[..., :self.slave_state_dim].copy()
            if state.ndim == 2 and self.state_future_size > 0:
                future_start = self.state_history_size + 1
                projected_state[future_start:] = state[
                    future_start:,
                    self.slave_state_dim:self.slave_state_dim * 2,
                ]
            state = projected_state
        elif input_mode == "m":
            state = state[..., self.slave_state_dim:self.slave_state_dim * 2]
        elif input_mode not in {"sm", "smp"}:
            raise ValueError(f"Unsupported ARX input mode: {self.mode}")

        state = transforms.pad_to_dim(state, self.action_dim)
        if state_mask is not None:
            state[..., -1] = state_mask
        data["state"] = state

        if "actions" in data:
            actions = np.asarray(data["actions"])
            self._validate_dim(actions, "actions")
            if output_mode == "s":
                actions = actions[..., :self.slave_state_dim]
            elif output_mode == "m":
                actions = actions[..., self.slave_state_dim:self.slave_state_dim * 2]
            elif output_mode not in {"sm", "smp"}:
                raise ValueError(f"Unsupported ARX output mode: {self.mode}")
            data["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        return data

    def _validate_dim(self, value: np.ndarray, name: str) -> None:
        required_dim = self.slave_state_dim * 2
        if value.shape[-1] < required_dim:
            raise ValueError(
                f"ARX {self.mode} expects normalized {name} from a full SM2SM dataset with at least "
                f"{required_dim} dimensions, got {value.shape[-1]}"
            )


@dataclasses.dataclass(frozen=True)
class RestoreNormalizedSm2smActions(transforms.DataTransformFn):
    """Restore projected model actions to full SM2SM slots before unnormalization."""

    mode: str
    action_dim: int = 32
    slave_state_dim: int = 14

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        output_mode = self.mode.split("2", maxsplit=1)[1]
        if output_mode in {"sm", "smp"}:
            return data

        restored = np.zeros((*actions.shape[:-1], self.action_dim), dtype=actions.dtype)
        if output_mode == "s":
            restored[..., :self.slave_state_dim] = actions[..., :self.slave_state_dim]
        elif output_mode == "m":
            restored[..., self.slave_state_dim:self.slave_state_dim * 2] = actions[
                ..., :self.slave_state_dim
            ]
        else:
            raise ValueError(f"Unsupported ARX output mode: {self.mode}")
        data["actions"] = restored
        return data


@dataclasses.dataclass(frozen=True)
class ArxOutputs(transforms.DataTransformFn):
    """Outputs for the ARX policy."""

    mode: str
    action_dim: int = 14
    project_from_sm2sm: bool = False

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"])
        if not self.project_from_sm2sm:
            return {"actions": actions[:, :self.action_dim]}

        output_mode = self.mode.split("2", maxsplit=1)[1]
        output_start = self.action_dim if output_mode == "m" else 0
        actions = actions[:, output_start:output_start + self.action_dim]
        return {"actions": actions}


@dataclasses.dataclass(frozen=True)
class ArxSm2smInputs(transforms.DataTransformFn):
    """Prepare X1Pro SM2SM state sequences with optional dense memory."""

    representation: str = "full_state"
    state_history_size: int = 3
    state_future_size: int = 3
    slave_state_dim: int = 14
    robot_state_dim: int = 28
    memory_dim: int = 3
    availability_mask_index: int = 31
    random_drop_master: float = 0.0
    random_drop_history: float = 0.0
    random_drop_future: float = 0.0
    random_pos_offset: float = 0.0

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "left_wrist_view",
        "face_view",
        "right_wrist_view",
    )

    def __post_init__(self) -> None:
        if self.representation not in {"full_state", "state_token"}:
            raise ValueError(f"Unsupported ARX memory representation: {self.representation!r}")
        if self.robot_state_dim != self.slave_state_dim * 2:
            raise ValueError("SM2SM robot_state_dim must contain equally-sized slave and master blocks")

    def __call__(self, data: dict) -> dict:
        raw_state = np.asarray(data["state"], dtype=np.float32)
        state, availability = self._prepare_state(raw_state)
        inputs = {
            "image": {
                "base_0_rgb": self._convert_image(data["images"]["face_view"]),
                "left_wrist_0_rgb": self._convert_image(data["images"]["left_wrist_view"]),
                "right_wrist_0_rgb": self._convert_image(data["images"]["right_wrist_view"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
            "state": state,
            "state_inpainting_mask": availability,
        }

        if "actions" in data:
            raw_actions = np.asarray(data["actions"], dtype=np.float32)
            output_dim = self.robot_state_dim + (self.memory_dim if self.representation == "full_state" else 0)
            if raw_actions.shape[-1] < output_dim:
                raise ValueError(f"ARX actions need at least {output_dim} dims, got {raw_actions.shape}")
            inputs["actions"] = np.array(raw_actions[..., :output_dim], copy=True)
            if "memory_action_valid" not in data:
                raise ValueError("Shared-memory ARX dataset is missing memory_action_valid")
            memory_valid = np.asarray(data["memory_action_valid"], dtype=np.bool_)
            if memory_valid.ndim > 0 and memory_valid.shape[-1] == 1:
                memory_valid = memory_valid[..., 0]
            if memory_valid.shape != raw_actions.shape[:-1]:
                raise ValueError(
                    f"memory_action_valid shape {memory_valid.shape} does not match actions {raw_actions.shape}"
                )
            action_loss_mask = np.zeros(
                (*raw_actions.shape[:-1], self.availability_mask_index + 1), dtype=np.bool_
            )
            robot_valid = self._causal_robot_action_valid(memory_valid)
            action_loss_mask[..., : self.robot_state_dim] = robot_valid[..., None]
            if self.representation == "full_state":
                action_loss_mask[..., self.robot_state_dim : output_dim] = memory_valid[..., None]
            inputs["action_loss_mask"] = action_loss_mask

        if self.representation == "state_token":
            input_ids = np.asarray(data["key_state_input_ids"], dtype=np.int32)
            inputs["key_state_input_ids"] = input_ids.reshape(1) if input_ids.ndim == 0 else input_ids
            if "actions" in data:
                for key, dtype in (
                    ("key_state_target_ids", np.int32),
                    ("key_state_target_mask", np.bool_),
                ):
                    if key not in data:
                        raise ValueError(f"State-token ARX dataset is missing {key}")
                    value = np.asarray(data[key], dtype=dtype)
                    inputs[key] = value.reshape(1) if value.ndim == 0 else value

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        if self.random_pos_offset > 0.0 and "actions" in inputs:
            offset = (np.random.rand(3).astype(np.float32) * 2.0 - 1.0) * self.random_pos_offset
            inputs["state"][..., 7:10] += offset
            inputs["state"][..., 21:24] += offset
            inputs["actions"][..., 7:10] += offset
            inputs["actions"][..., 21:24] += offset
        return inputs

    @staticmethod
    def _causal_robot_action_valid(memory_valid: np.ndarray) -> np.ndarray:
        if memory_valid.ndim == 0:
            return np.ones((), dtype=np.bool_)
        if memory_valid.ndim != 1:
            raise ValueError(f"Expected one action-horizon validity axis, got {memory_valid.shape}")
        search_start = 0
        if not memory_valid[0]:
            available = np.flatnonzero(memory_valid)
            if not len(available):
                return np.ones_like(memory_valid)
            search_start = int(available[0])
        future_forced = np.flatnonzero(~memory_valid[search_start:])
        if not len(future_forced):
            return np.ones_like(memory_valid)
        result = np.ones_like(memory_valid)
        result[search_start + int(future_forced[0]) :] = False
        return result

    def _prepare_state(self, raw_state: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if raw_state.ndim not in {1, 2}:
            raise ValueError(f"ARX state must be a vector or sequence, got {raw_state.shape}")
        required_dim = self.robot_state_dim + (self.memory_dim if self.representation == "full_state" else 0)
        if raw_state.shape[-1] < required_dim:
            raise ValueError(f"ARX state needs at least {required_dim} dims, got {raw_state.shape}")
        output = np.array(raw_state[..., :required_dim], copy=True)
        if raw_state.ndim == 1:
            availability = np.asarray(
                raw_state[self.availability_mask_index]
                if raw_state.shape[-1] > self.availability_mask_index
                else 0.0,
                dtype=np.float32,
            )
            return output, availability

        expected_length = self.state_history_size + 1 + self.state_future_size
        if raw_state.shape[0] != expected_length:
            raise ValueError(f"Expected {expected_length} state frames, got {raw_state.shape[0]}")
        current_index = self.state_history_size
        current_state = np.array(output[current_index], copy=True)
        current_slave = current_state[: self.slave_state_dim]
        availability = (
            np.array(raw_state[:, self.availability_mask_index], copy=True)
            if raw_state.shape[-1] > self.availability_mask_index
            else np.zeros(expected_length, dtype=np.float32)
        )
        if self.state_future_size > 0:
            future = slice(current_index + 1, None)
            output[future, : self.slave_state_dim] = current_slave
            if self.representation == "full_state":
                output[future, self.robot_state_dim : required_dim] = current_state[
                    self.robot_state_dim : required_dim
                ]
        if random.random() < self.random_drop_master:
            output[:, self.slave_state_dim : self.robot_state_dim] = current_slave
            availability[:] = 1.0
        if self.state_history_size > 0 and random.random() < self.random_drop_history:
            output[:current_index] = current_state
            availability[:current_index] = 1.0
        if self.state_future_size > 0 and random.random() < self.random_drop_future:
            drop_size = random.randint(1, self.state_future_size)
            output[-drop_size:] = output[-drop_size - 1]
            availability[-drop_size:] = 1.0
        return output, availability

    @staticmethod
    def _convert_image(image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if np.issubdtype(image.dtype, np.floating):
            image = (255.0 * image).astype(np.uint8)
        if image.shape[-1] != 3:
            image = einops.rearrange(image, "c h w -> h w c")
        if image.shape[-1] != 3:
            raise ValueError(f"Expected an RGB image, got {image.shape}")
        return image


@dataclasses.dataclass(frozen=True)
class AddStateInpaintingMask(transforms.DataTransformFn):
    action_dim: int = 32

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(np.asarray(data["state"]), self.action_dim)
        mask = np.asarray(data.pop("state_inpainting_mask"), dtype=state.dtype)
        if mask.shape != state.shape[:-1]:
            raise ValueError(f"Inpainting mask shape {mask.shape} does not match state {state.shape}")
        state[..., -1] = mask
        data["state"] = state
        return data


@dataclasses.dataclass(frozen=True)
class ArxSm2smOutputs(transforms.DataTransformFn):
    representation: str = "full_state"
    robot_state_dim: int = 28
    memory_dim: int = 3

    def __call__(self, data: dict) -> dict:
        output_dim = self.robot_state_dim + (self.memory_dim if self.representation == "full_state" else 0)
        return {"actions": np.asarray(data["actions"])[..., :output_dim]}

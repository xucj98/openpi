import dataclasses
import logging

import numpy as np

from openpi import transforms

_LOG_COUNT = 0


@dataclasses.dataclass(frozen=True)
class AdaptiveAccInputs(transforms.DataTransformFn):
    action_horizon: int

    def __call__(self, data: dict) -> dict:
        global _LOG_COUNT
        if "actions" not in data:
            return data

        factor = float(data.get("_speedup_factor", 1.0))
        actions = np.asarray(data["actions"])
        original_length = actions.shape[0]

        L = max(1, min(int(self.action_horizon * factor), original_length))

        actions = actions[:L]

        if "actions_is_pad" in data:
            old_pad = np.asarray(data["actions_is_pad"])
            old_pad = old_pad[: min(L, len(old_pad))]

        resampled = L != self.action_horizon

        if resampled:
            x_old = np.linspace(0, 1, L)
            x_new = np.linspace(0, 1, self.action_horizon)
            resampled_arr = np.zeros((self.action_horizon, actions.shape[1]), dtype=actions.dtype)
            for dim in range(actions.shape[1]):
                resampled_arr[:, dim] = np.interp(x_new, x_old, actions[:, dim])
            data["actions"] = resampled_arr

            if "actions_is_pad" in data:
                indices = np.linspace(0, min(L - 1, len(old_pad) - 1), self.action_horizon).astype(int)
                data["actions_is_pad"] = old_pad[indices]
        else:
            data["actions"] = actions
            if "actions_is_pad" in data:
                data["actions_is_pad"] = old_pad

        _LOG_COUNT += 1
        if _LOG_COUNT <= 10 or _LOG_COUNT % 5000 == 0:
            print(
                f"[AdaptiveAcc] #{_LOG_COUNT} factor={factor:.2f} orig_len={original_length} L={L} "
                f"actions ({actions.shape[0]},{actions.shape[1]})->({data['actions'].shape[0]},{data['actions'].shape[1]}) "
                f"resampled={resampled}",
                flush=True,
            )

        return data

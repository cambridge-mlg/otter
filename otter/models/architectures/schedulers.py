import logging
from typing import Any, Dict, Self

import numpy as np
import numpy.typing as npt


class RolloutHorizonScheduler:
    @classmethod
    def pretraining_scheduler(
        cls,
        num_steps: int,
        temporal_resolution_hours: int,
    ) -> Self:
        return cls(
            num_steps=num_steps,
            min_rollout_steps=1,
            max_rollout_steps=1,
            temporal_resolution_hours=temporal_resolution_hours,
        )

    def __init__(
        self,
        num_steps: int,
        min_rollout_steps: int,
        max_rollout_steps: int,
        temporal_resolution_hours: int,
    ) -> None:
        """
        Args:
            num_steps: total number of steps in the training process.
            min_rollout_steps: minimum rollout steps.
            max_rollout_steps: maximum rollout steps.
            temporal_resolution_hours: temporal resolution of the data in hours.
        """
        super().__init__()

        assert max_rollout_steps >= min_rollout_steps, (
            f"Max rollout horizon {max_rollout_steps} "
            f"must be greater than or equal to min rollout horizon "
            f"{min_rollout_steps}"
        )

        self.num_steps = num_steps
        self.min_rollout_steps = min_rollout_steps
        self.max_rollout_steps = max_rollout_steps

        self.current_rollout_steps = min_rollout_steps
        self.current_step = 0

        self.increment_every_num_steps = num_steps // (
            max_rollout_steps - min_rollout_steps + 1
        )
        logging.info("Number of training steps: {}".format(num_steps))
        logging.info(
            f"RolloutHorizonScheduler will increment "
            f"the rollout horizon every "
            f"{self.increment_every_num_steps} steps."
        )

        self.temporal_resolution_hours = temporal_resolution_hours

    def step(self) -> None:
        """
        Increments the current step and updates the rollout horizon if it is divisible by `increment_every_num_steps`.
        """
        self.current_step += 1

        if (
            self.current_step % self.increment_every_num_steps == 0
            and self.current_rollout_steps < self.max_rollout_steps
        ):
            logging.info(
                f"Incrementing rollout horizon from "
                f"{self.current_rollout_steps} to "
                f"{self.current_rollout_steps + 1} at step "
                f"{self.current_step}"
            )
            self.current_rollout_steps += 1

    def get_rollout_steps(self) -> int:
        """
        Returns the current rollout horizon in steps.
        """
        return self.current_rollout_steps

    def get_rollout_time_hours(self) -> npt.NDArray[np.int32]:
        """
        Returns the current rollout horizon in hours.
        """
        return self.temporal_resolution_hours * (
            np.arange(self.current_rollout_steps) + 1
        )

    def state_dict(self) -> Dict[str, Any]:
        """Returns the scheduler's state as a dictionary."""
        return dict(self.__dict__)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

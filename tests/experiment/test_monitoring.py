from typing import Any

import pytest

from otter.experiment import monitoring
from otter.experiment.monitoring import WandbLogger


def test_timed_logs_the_total_of_a_steps_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logged: dict[str, Any] = {}

    def fake_log(log_dict: dict[str, Any], step: int) -> None:
        assert step == 7
        logged.update(log_dict)

    durations = iter([1.0, 2.0, 4.0])

    def fake_run_with_timing(f: Any, *args: Any) -> tuple[float, Any]:
        return next(durations), f(*args)

    monkeypatch.setattr(monitoring.wandb, "log", fake_log)
    monkeypatch.setattr(monitoring, "run_with_timing", fake_run_with_timing)

    logger = WandbLogger(step=7)
    assert logger.timed("next_batch", int, "3") == 3
    logger.timed("next_batch", int, "4")
    logger.timed("forward", int, "5")

    assert logged == {"time/next_batch": 3.0, "time/forward": 4.0}

import numpy as np
import pytest

from otter.models.architectures.schedulers import RolloutHorizonScheduler


def test_rollout_horizon_scheduler_init():
    num_steps = 100
    max_horizon = 10
    min_horizon = 1
    temporal_res = 6

    scheduler = RolloutHorizonScheduler(
        num_steps=num_steps,
        min_rollout_steps=min_horizon,
        max_rollout_steps=max_horizon,
        temporal_resolution_hours=temporal_res,
    )

    assert scheduler.num_steps == num_steps
    assert scheduler.max_rollout_steps == max_horizon
    assert scheduler.min_rollout_steps == min_horizon
    assert scheduler.current_rollout_steps == min_horizon
    assert scheduler.current_step == 0
    # increment_every_num_steps = 100 // (10 - 1 + 1) = 100 // 10 = 10
    assert scheduler.increment_every_num_steps == 10
    assert scheduler.temporal_resolution_hours == temporal_res


def test_rollout_horizon_scheduler_assertion():
    with pytest.raises(
        AssertionError,
        match="Max rollout horizon 1 must be greater than or equal to min rollout horizon 5",
    ):
        RolloutHorizonScheduler(
            num_steps=100,
            max_rollout_steps=1,
            min_rollout_steps=5,
            temporal_resolution_hours=6,
        )


def test_rollout_horizon_scheduler_step():
    num_steps = 20
    max_horizon = 3
    min_horizon = 1
    temporal_res = 6

    # increment_every_num_steps = 20 // (3 - 1 + 1) = 20 // 3 = 6
    scheduler = RolloutHorizonScheduler(
        num_steps=num_steps,
        max_rollout_steps=max_horizon,
        min_rollout_steps=min_horizon,
        temporal_resolution_hours=temporal_res,
    )

    assert scheduler.get_rollout_steps() == 1

    # Step 1 to 5: horizon should remain 1
    for i in range(1, 6):
        scheduler.step()
        assert scheduler.current_step == i
        assert scheduler.get_rollout_steps() == 1

    # Step 6: horizon should increment to 2
    scheduler.step()
    assert scheduler.current_step == 6
    assert scheduler.get_rollout_steps() == 2

    # Step 7 to 11: horizon should remain 2
    for i in range(7, 12):
        scheduler.step()
        assert scheduler.get_rollout_steps() == 2

    # Step 12: horizon should increment to 3
    scheduler.step()
    assert scheduler.get_rollout_steps() == 3

    # Step 13 to 20: horizon should remain 3 (max reached)
    for i in range(13, 21):
        scheduler.step()
        assert scheduler.get_rollout_steps() == 3


def test_get_temporal_rollout_horizon_hours():
    scheduler = RolloutHorizonScheduler(
        num_steps=100,
        max_rollout_steps=5,
        min_rollout_steps=2,
        temporal_resolution_hours=6,
    )

    # Current horizon is 2
    expected = np.array([6, 12])
    np.testing.assert_array_equal(scheduler.get_rollout_time_hours(), expected)

    # Increment to 3
    scheduler.current_rollout_steps = 3
    expected = np.array([6, 12, 18])
    np.testing.assert_array_equal(scheduler.get_rollout_time_hours(), expected)


def test_state_dict_serialization():
    scheduler = RolloutHorizonScheduler(
        num_steps=100,
        max_rollout_steps=10,
        min_rollout_steps=1,
        temporal_resolution_hours=6,
    )

    scheduler.step()
    scheduler.step()

    state = scheduler.state_dict()
    assert state["current_step"] == 2
    assert state["num_steps"] == 100

    new_scheduler = RolloutHorizonScheduler(
        num_steps=1,
        max_rollout_steps=1,
        min_rollout_steps=1,
        temporal_resolution_hours=1,
    )

    new_scheduler.load_state_dict(state)
    assert new_scheduler.current_step == 2
    assert new_scheduler.num_steps == 100
    assert new_scheduler.max_rollout_steps == 10
    assert new_scheduler.min_rollout_steps == 1
    assert new_scheduler.temporal_resolution_hours == 6

import torch
import pytest
from otter.experiment.utils import get_sequential_warmup_cosine_lr_scheduler

def test_sequential_warmup_cosine_lr_scheduler():
    # Setup
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    warmup_steps = 5
    total_steps = 15

    scheduler = get_sequential_warmup_cosine_lr_scheduler(
        optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        eta_min=0.01
    )

    lrs = []
    # Step through the scheduler
    for _ in range(total_steps):
        lrs.append(scheduler.get_last_lr()[0])
        optimizer.step()
        scheduler.step()

    # Verification

    # 1. Start small (warmup)
    assert lrs[0] < 0.1

    # 2. Peak at warmup_steps (milestone is at step 5, so peak is at step 4 or 5 depending on implementation details)
    # The scheduler transitions at warmup_steps.
    # Step 0: Warmup (start_factor)
    # Step 4: Warmup (near end_factor=1.0)
    # Step 5: Cosine starts (usually at max LR)

    # Check that we reach ~1.0 around the transition
    assert lrs[warmup_steps] == pytest.approx(1.0, abs=1e-5) or lrs[warmup_steps-1] == pytest.approx(1.0, abs=1e-5)

    # 3. Decay phase
    # After peak, it should decrease
    assert lrs[-1] < lrs[warmup_steps]

    # 4. Check end value (should be equal to eta_min=0.01)
    assert lrs[-1] == pytest.approx(0.01, abs=1e-5) # Final LR should match eta_min

    # Precise check for a few points:
    # Warmup is LinearLR(start=1e-10, end=1.0, iters=5)
    # t=0: 1*1e-10
    # t=1: 0.2
    # t=2: 0.4
    # t=3: 0.6
    # t=4: 0.8
    # t=5: Cosine starts (max) -> 1.0 (Cosine T_max = 10)
    # t=14: Cosine min -> 0.01

    # Re-instantiate to check exact values if needed, but trend check is safer for integration tests.
    assert lrs[1] > lrs[0] # Increasing during warmup
    assert lrs[warmup_steps + 2] < lrs[warmup_steps] # Decreasing during cosine

import torch

from otter.experiment.utils import CombinedOptimizer


def test_combined_optimizer_state_dict_decoupling() -> None:
    """
    Tests that updating the CombinedOptimizer's param_groups (e.g. by a Scheduler)
    correctly updates the underlying child optimizers after a checkpoint load.
    """
    # 1. Setup dummy parameters and optimizers
    param1 = torch.nn.Parameter(torch.randn(10, 10))
    param2 = torch.nn.Parameter(torch.randn(10, 10))

    # Use SGD for simplicity.
    # param_groups[0] corresponds to opt1, param_groups[1] to opt2
    opt1 = torch.optim.SGD([param1], lr=0.1)
    opt2 = torch.optim.SGD([param2], lr=0.01)

    combined_opt = CombinedOptimizer([opt1, opt2])

    # 2. Verify initial linkage (Pre-save)
    # Changing the wrapper should change the child
    combined_opt.param_groups[0]["lr"] = 0.5
    assert opt1.param_groups[0]["lr"] == 0.5, "Initial linkage failed"

    # 3. Save State
    # Current State: opt1 lr=0.5, opt2 lr=0.01
    state_dict = combined_opt.state_dict()

    # 4. Initialize a NEW optimizer instance (Simulate restarting a run)
    opt1_new = torch.optim.SGD([param1], lr=0.1)  # Defaults
    opt2_new = torch.optim.SGD([param2], lr=0.01)  # Defaults
    combined_new = CombinedOptimizer([opt1_new, opt2_new])

    # 5. Load State
    combined_new.load_state_dict(state_dict)

    # Verify load was successful
    assert opt1_new.param_groups[0]["lr"] == 0.5
    assert combined_new.param_groups[0]["lr"] == 0.5

    # 6. Simulate LR Scheduler Step
    # The scheduler modifies combined_new.param_groups
    new_lr = 0.0001
    combined_new.param_groups[0]["lr"] = new_lr

    # 7. Assert that the Child Optimizer actually got the update
    assert opt1_new.param_groups[0]["lr"] == new_lr

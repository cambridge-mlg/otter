from otter.experiment.utils import get_num_training_steps

class FakeDataLoader:
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

def test_get_num_training_steps_no_accumulation():
    dataloader = FakeDataLoader(length=100)
    num_epochs = 10

    # Expected: 10 * 100 = 1000
    steps = get_num_training_steps(num_epochs, dataloader, accumulate_batches=1)
    assert steps == 1000

def test_get_num_training_steps_with_accumulation():
    dataloader = FakeDataLoader(length=100)
    num_epochs = 10
    accumulate_batches = 4

    # Expected: (10 * 100) // 4 = 1000 // 4 = 250
    steps = get_num_training_steps(num_epochs, dataloader, accumulate_batches=accumulate_batches)
    assert steps == 250

def test_get_num_training_steps_with_accumulation_remainder():
    dataloader = FakeDataLoader(length=101) # Indivisible by 4
    num_epochs = 1
    accumulate_batches = 4

    # Expected: 101 // 4 = 25
    steps = get_num_training_steps(num_epochs, dataloader, accumulate_batches=accumulate_batches)
    assert steps == 25

import pytest
import torch

from lerobot.utils.delay_sampling import sample_delay_distribution, validate_delay_distribution

RANGES = ((1, 10), (11, 17), (18, 25), (26, 30))
PROBABILITIES = (0.15, 0.20, 0.50, 0.15)


def test_real_robot_delay_prior_and_uniformity():
    valid = torch.ones(100_000, 31, dtype=torch.bool)
    delays = sample_delay_distribution(valid, RANGES, PROBABILITIES, torch.Generator().manual_seed(42))
    assert delays.min() >= 1 and delays.max() <= 30
    histogram = torch.bincount(delays, minlength=31).float() / len(delays)
    for (low, high), probability in zip(RANGES, PROBABILITIES, strict=True):
        assert abs(histogram[low : high + 1].sum().item() - probability) < 0.006
        expected = probability / (high - low + 1)
        assert (histogram[low : high + 1] - expected).abs().max() < 0.003


def test_tail_never_selects_padding_and_zero_only_without_future():
    valid = torch.zeros(3000, 31, dtype=torch.bool)
    valid[:1000, :1] = True
    valid[1000:2000, :6] = True
    valid[2000:, [0, 19, 27]] = True
    delays = sample_delay_distribution(valid, RANGES, PROBABILITIES, torch.Generator().manual_seed(7))
    assert (delays[:1000] == 0).all()
    assert ((delays[1000:2000] >= 1) & (delays[1000:2000] <= 5)).all()
    assert torch.isin(delays[2000:], torch.tensor([19, 27])).all()
    assert valid[torch.arange(len(delays)), delays].all()


@pytest.mark.parametrize(
    "ranges, probabilities",
    [
        (((1, 10), (12, 30)), (0.5, 0.5)),
        (RANGES, (0.1, 0.1, 0.1, 0.1)),
        (RANGES, (0.15, 0.2, float("nan"), 0.15)),
    ],
)
def test_bad_prior_rejected(ranges, probabilities):
    with pytest.raises(ValueError):
        validate_delay_distribution(ranges, probabilities, 30)

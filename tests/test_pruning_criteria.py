import pytest

torch = pytest.importorskip("torch")
from torch.utils.data import DataLoader, TensorDataset

from project_rank.models import ResNet20
from project_rank.pruning_criteria import (
    collect_l1_weight_scores,
    collect_taylor_scores,
    l2_normalize,
)


def test_pruning_scores_have_one_value_per_internal_channel():
    torch.manual_seed(7)
    model = ResNet20()
    images = torch.randn(3, 3, 32, 32)
    targets = torch.tensor([0, 1, 2])
    loader = DataLoader(TensorDataset(images, targets), batch_size=2)

    l1 = collect_l1_weight_scores(model)
    taylor, samples = collect_taylor_scores(
        model, loader, torch.device("cpu"), max_samples=3
    )

    assert samples == 3
    assert list(l1) == [f"block{i}" for i in range(1, 10)]
    for index, width in enumerate(model.inner_widths, start=1):
        assert len(l1[f"block{index}"]) == width
        assert len(taylor[f"block{index}"]) == width
        assert all(value >= 0.0 for value in taylor[f"block{index}"])


def test_l2_normalize_handles_zero_scores():
    values = l2_normalize([0.0, 0.0, 0.0, 0.0])
    assert sum(value * value for value in values) == pytest.approx(1.0)

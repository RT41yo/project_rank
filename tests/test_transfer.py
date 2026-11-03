import pytest

torch = pytest.importorskip("torch")

from project_rank.models import ResNet20
from project_rank.transfer import (
    activation_subspace_selection,
    transfer_by_l1,
    transfer_by_random,
)


def test_transfer_builds_working_narrow_model():
    torch.manual_seed(5)
    source = ResNet20()
    widths = [8, 12, 8, 16, 24, 20, 32, 48, 40]
    target = ResNet20(widths)
    selected = transfer_by_l1(source, target)
    assert len(selected) == 9
    assert all(len(selected[f"block{i + 1}"]) == widths[i] for i in range(9))
    output = target(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 10)


def test_random_transfer_is_reproducible():
    source = ResNet20()
    widths = [8, 10, 12, 16, 18, 20, 24, 28, 32]
    first = ResNet20(widths)
    second = ResNet20(widths)
    third = ResNet20(widths)

    selected_first = transfer_by_random(source, first, seed=2001)
    selected_second = transfer_by_random(source, second, seed=2001)
    selected_third = transfer_by_random(source, third, seed=2002)

    assert selected_first == selected_second
    assert selected_first != selected_third
    assert torch.equal(first.blocks[0].conv1.weight, second.blocks[0].conv1.weight)


def test_activation_subspace_prefers_leading_covariance_axes():
    covariances = {
        f"block{index}": torch.diag(torch.tensor([9.0, 8.0, 1.0, 0.1]))
        for index in range(1, 10)
    }
    selected, scores = activation_subspace_selection(covariances, [2] * 9)

    assert selected["block1"] == [0, 1]
    assert scores["block1"][0] == pytest.approx(1.0)
    assert scores["block1"][2] == pytest.approx(0.0)


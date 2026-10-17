import pytest

torch = pytest.importorskip("torch")

from project_rank.models import ResNet20
from project_rank.transfer import transfer_by_l1


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


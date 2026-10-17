import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import DataLoader, TensorDataset

from project_rank.effective_rank import collect_activation_ranks
from project_rank.models import ResNet20


def test_activation_collection_returns_all_blocks():
    model = ResNet20()
    images = torch.randn(4, 3, 32, 32)
    targets = torch.zeros(4, dtype=torch.long)
    loader = DataLoader(TensorDataset(images, targets), batch_size=2)
    results = collect_activation_ranks(model, loader, torch.device("cpu"), max_samples=3)
    assert list(results) == [f"block{index}" for index in range(1, 10)]
    for item in results.values():
        assert item["images"] == 3
        assert 0.0 <= item["pooled_effective_rank"] <= item["channels"] + 1e-8


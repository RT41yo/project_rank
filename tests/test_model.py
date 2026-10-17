import pytest

torch = pytest.importorskip("torch")

from project_rank.models import ResNet20, count_resnet20_parameters


def test_model_output_and_analytic_parameter_count():
    widths = [16, 12, 8, 32, 24, 16, 64, 48, 32]
    model = ResNet20(widths)
    output = model(torch.randn(2, 3, 32, 32))
    assert output.shape == (2, 10)
    assert sum(parameter.numel() for parameter in model.parameters()) == (
        count_resnet20_parameters(widths)
    )


def test_model_requires_nine_widths():
    with pytest.raises(ValueError):
        ResNet20([16, 16])


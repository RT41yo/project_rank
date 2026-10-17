import csv

from project_rank.build_architectures import read_rank_columns


def test_read_rank_columns(tmp_path):
    path = tmp_path / "ranks.csv"
    fieldnames = [
        "block",
        "mean_per_image_effective_rank",
        "pooled_effective_rank",
        "weight_effective_rank",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(1, 10):
            writer.writerow(
                {
                    "block": f"block{index}",
                    "mean_per_image_effective_rank": 5 + index,
                    "pooled_effective_rank": 6 + index,
                    "weight_effective_rank": 7 + index,
                }
            )
    result = read_rank_columns(path)
    assert result["mean_per_image_effective_rank"][0] == 6.0
    assert result["pooled_effective_rank"][-1] == 15.0


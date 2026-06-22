from __future__ import annotations

from pathlib import Path
import csv

from . import INK_CHANNELS
from .cmm import convert_7clr_to_lab
from .model import load_or_train_xgamut_model, predict_7clr_from_lab
from .txt_chart import parse_7clr_txt

OUTPUT_COLUMNS = [
    "SampleID",
    "SAMPLE_NAME",
    *INK_CHANNELS,
    "Lab_L",
    "Lab_a",
    "Lab_b",
    *[f"Pred_{channel}" for channel in INK_CHANNELS],
]


def write_validation_csv(rows: list[dict[str, object]], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    return path


def run_sample_validation(
    txt_path: str | Path,
    icc_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    sample_size: int = 50,
    cmm_backend: str = "lcms",
    transicc_path: str | Path = "transicc",
    xicclu_path: str = "xicclu",
    std_path: str | Path = "Pantone_Coated_CS1_Extract-2.xlsx",
    ink_path: str | Path = "New_V3_7色-2-real_Cleaned_Sorted.xlsx",
    active_learning_path: str | Path | None = "2390-1.xlsx",
    retrained_model_path: str | Path = "xgamut_model_current.pkl",
) -> Path:
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")

    rows = parse_7clr_txt(txt_path)
    sample = rows[:sample_size]
    converted = convert_7clr_to_lab(
        sample,
        icc_path,
        cmm_backend=cmm_backend,
        transicc_path=transicc_path,
        xicclu_path=xicclu_path,
    )
    model_package = load_or_train_xgamut_model(
        model_path=model_path,
        std_path=std_path,
        ink_path=ink_path,
        active_learning_path=active_learning_path,
        retrained_model_path=retrained_model_path,
    )
    predicted = predict_7clr_from_lab(model_package, converted)
    return write_validation_csv(predicted, output_path)

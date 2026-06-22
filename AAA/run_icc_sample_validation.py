from __future__ import annotations

import argparse
from pathlib import Path

from icc_validation.cmm import CmmError
from icc_validation.runner import run_sample_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a sample 7CLR ICC -> Lab -> model validation chain.",
    )
    parser.add_argument("--txt", default="CMYKOGV_i1 Pro3 iO_XGAMUNT.txt")
    parser.add_argument("--icc", default="CMYKOGV_i1 Pro3 iO_XGAMUNT_Real.icc")
    parser.add_argument("--model", default="xgamut_model.pkl")
    parser.add_argument("--retrained-model", default="xgamut_model_current.pkl")
    parser.add_argument("--std", default="Pantone_Coated_CS1_Extract-2.xlsx")
    parser.add_argument("--ink", default="New_V3_7色-2-real_Cleaned_Sorted.xlsx")
    parser.add_argument("--active-learning", default="2390-1.xlsx")
    parser.add_argument("--output", default="outputs/icc_sample_validation.csv")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--xicclu", default="xicclu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        output = run_sample_validation(
            txt_path=Path(args.txt),
            icc_path=Path(args.icc),
            model_path=Path(args.model),
            output_path=Path(args.output),
            sample_size=args.sample_size,
            xicclu_path=args.xicclu,
            std_path=Path(args.std),
            ink_path=Path(args.ink),
            active_learning_path=Path(args.active_learning) if args.active_learning else None,
            retrained_model_path=Path(args.retrained_model),
        )
    except CmmError as exc:
        print(f"ICC conversion failed: {exc}")
        return 2
    print(f"Wrote validation CSV: {output}")
    print("Prediction columns are observational only; they are not a pass/fail accuracy metric.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

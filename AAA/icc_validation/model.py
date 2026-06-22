from __future__ import annotations

from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from . import INK_CHANNELS

EXPECTED_FEATURE_COLS = ["L", "a", "b", "C", "h"]
EXPECTED_TARGET_COLS = list(INK_CHANNELS)


def load_xgamut_model(model_path: str | Path) -> dict[str, object]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    package = joblib.load(path)
    if not isinstance(package, dict):
        raise ValueError("Model package must be a dict")

    missing = [key for key in ("model", "feature_cols", "target_cols") if key not in package]
    if missing:
        raise ValueError(f"Model package missing keys: {', '.join(missing)}")

    _validate_model_package(package)
    return package


def load_or_train_xgamut_model(
    model_path: str | Path,
    std_path: str | Path,
    ink_path: str | Path,
    active_learning_path: str | Path | None = None,
    retrained_model_path: str | Path | None = None,
    force_retrain: bool = False,
) -> dict[str, object]:
    path = Path(model_path)
    output_path = Path(retrained_model_path) if retrained_model_path else path.with_name(f"{path.stem}_current.pkl")

    if output_path.exists() and output_path != path and not force_retrain:
        try:
            return load_xgamut_model(output_path)
        except Exception as exc:
            print(f"Cached retrained model could not be loaded ({exc}); rebuilding it.")

    if path.exists() and not force_retrain:
        try:
            return load_xgamut_model(path)
        except Exception as exc:
            print(f"Existing model could not be loaded ({exc}); retraining with current Python environment.")

    return train_xgamut_model(
        std_path=std_path,
        ink_path=ink_path,
        active_learning_path=active_learning_path,
        model_output_path=output_path,
    )


def train_xgamut_model(
    std_path: str | Path,
    ink_path: str | Path,
    active_learning_path: str | Path | None = None,
    model_output_path: str | Path | None = None,
) -> dict[str, object]:
    db = build_training_database(std_path, ink_path, active_learning_path)
    if db.empty:
        raise ValueError("Training dataset is empty")

    db["C"] = np.sqrt(db["a"] ** 2 + db["b"] ** 2)
    db["h"] = np.degrees(np.arctan2(db["b"], db["a"])) % 360

    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(db[EXPECTED_FEATURE_COLS], db[EXPECTED_TARGET_COLS])

    package: dict[str, object] = {
        "model": model,
        "feature_cols": EXPECTED_FEATURE_COLS,
        "target_cols": EXPECTED_TARGET_COLS,
    }

    if model_output_path:
        output = Path(model_output_path)
        joblib.dump(package, output)
        print(f"Saved retrained model: {output}")

    return package


def build_training_database(
    std_path: str | Path,
    ink_path: str | Path,
    active_learning_path: str | Path | None = None,
) -> pd.DataFrame:
    df_std = _load_clean_training_frame(std_path)
    df_7c = _load_clean_training_frame(ink_path)
    df_new = _load_clean_training_frame(active_learning_path) if active_learning_path else None

    train_a = pd.DataFrame()
    if df_std is not None and df_7c is not None and "Key" in df_std.columns and "Key" in df_7c.columns:
        merged = pd.merge(df_std, df_7c, on="Key", how="inner")
        req_cols = ["L", "a", "b", *EXPECTED_TARGET_COLS]
        if all(c in merged.columns for c in req_cols):
            train_a = merged[req_cols].copy()

    train_b = pd.DataFrame()
    if df_new is not None:
        req_cols = ["L", "a", "b", *EXPECTED_TARGET_COLS]
        if all(c in df_new.columns for c in req_cols):
            train_b = df_new[req_cols].copy()
        elif "Key" in df_new.columns and df_std is not None and "Key" in df_std.columns:
            temp_std = df_std[["Key", "L", "a", "b"]]
            merged_new = pd.merge(temp_std, df_new, on="Key", how="inner")
            if all(c in merged_new.columns for c in req_cols):
                train_b = merged_new[req_cols].copy()

    full_db = pd.concat([train_a, train_b], ignore_index=True)
    full_db.dropna(inplace=True)
    return full_db


def _load_clean_training_frame(path: str | Path | None) -> pd.DataFrame | None:
    if not path:
        return None
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(f"Training data file not found: {data_path}")

    if data_path.suffix.lower() == ".csv":
        df = pd.read_csv(data_path)
    else:
        df = pd.read_excel(data_path, engine="openpyxl")

    df.columns = [str(column).strip() for column in df.columns]

    possible_keys = ["SAMPLE_NAME", "SAMPLEID", "NAME", "SAMPLE", "ID", "KEY", "Name", "Sample"]
    possible_key_names = {key.upper() for key in possible_keys}
    key_col = next((c for c in df.columns if str(c).upper() in possible_key_names), None)
    if key_col:
        df["Key"] = df[key_col].astype(str).str.replace('"', "").str.strip().str.upper()
        df["Key"] = df["Key"].replace(r"\s+", " ", regex=True)

    renames: dict[str, str] = {}
    for column in df.columns:
        col_upper = str(column).upper()
        if col_upper in ["L", "L*", "CIE_L", "LAB_L"]:
            renames[column] = "L"
        elif col_upper in ["A", "A*", "CIE_A", "LAB_A"]:
            renames[column] = "a"
        elif col_upper in ["B", "B*", "CIE_B", "LAB_B"]:
            renames[column] = "b"

    ink_map_pattern = {
        "7CLR_1": "Cyan",
        "CYAN": "Cyan",
        "7CLR_2": "Magenta",
        "MAGENTA": "Magenta",
        "7CLR_3": "Yellow",
        "YELLOW": "Yellow",
        "7CLR_4": "Black",
        "BLACK": "Black",
        "7CLR_5": "Orange",
        "ORANGE": "Orange",
        "7CLR_6": "Green",
        "GREEN": "Green",
        "7CLR_7": "Violet",
        "VIOLET": "Violet",
    }
    for column in df.columns:
        col_upper = str(column).upper()
        for pattern, target in ink_map_pattern.items():
            if pattern in col_upper:
                renames[column] = target
                break

    if renames:
        df.rename(columns=renames, inplace=True)

    for column in [c for c in ["L", "a", "b", *EXPECTED_TARGET_COLS] if c in df.columns]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def _validate_model_package(model_package: dict[str, object]) -> None:
    feature_cols = list(model_package["feature_cols"])
    target_cols = list(model_package["target_cols"])

    if feature_cols != EXPECTED_FEATURE_COLS:
        raise ValueError(f"Unexpected feature columns: {feature_cols}")
    if target_cols != EXPECTED_TARGET_COLS:
        raise ValueError(f"Unexpected target columns: {target_cols}")


def _build_feature_frame(model_package: dict[str, object], lab_rows: list[dict[str, object]]) -> pd.DataFrame:
    _validate_model_package(model_package)

    records: list[dict[str, float]] = []
    for row in lab_rows:
        lab_l = float(row["Lab_L"])
        lab_a = float(row["Lab_a"])
        lab_b = float(row["Lab_b"])
        chroma = math.sqrt(lab_a**2 + lab_b**2)
        hue = math.degrees(math.atan2(lab_b, lab_a)) % 360
        records.append({"L": lab_l, "a": lab_a, "b": lab_b, "C": chroma, "h": hue})

    return pd.DataFrame(records, columns=list(model_package["feature_cols"]))


def predict_7clr_from_lab(
    model_package: dict[str, object],
    lab_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    if not lab_rows:
        return []

    frame = _build_feature_frame(model_package, lab_rows)
    predictions = model_package["model"].predict(frame)
    predictions = np.clip(predictions, 0, 100)

    output: list[dict[str, object]] = []
    target_cols = list(model_package["target_cols"])
    for row, prediction in zip(lab_rows, predictions):
        merged = dict(row)
        for channel, value in zip(target_cols, prediction):
            merged[f"Pred_{channel}"] = round(float(value), 4)
        output.append(merged)
    return output

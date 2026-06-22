# ICC 7CLR Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python validation chain that samples 7CLR recipes from the CGS/XGamut txt chart, converts them to Lab through the 7CLR ICC profile, predicts 7CLR with the existing random forest model, and writes a CSV for inspection.

**Architecture:** Add a small `icc_validation` package under `AAA` with separate modules for txt parsing, ICC/CMM conversion, model prediction, and orchestration. Use ArgyllCMS `xicclu` as the first CMM backend because the profile device space is `7CLR`, which ordinary RGB/CMYK-only Python ICC paths do not cover. Keep all public 7CLR channel values in the `0-100` range.

**Tech Stack:** Python 3, stdlib `csv`/`pathlib`/`subprocess`, `pytest`, `joblib`, `numpy`, `pandas`, `scikit-learn`, ArgyllCMS `xicclu`.

## Global Constraints

- The main flow uses fixed local inputs: `AAA/CMYKOGV_i1 Pro3 iO_XGAMUNT.txt`, `AAA/CMYKOGV_i1 Pro3 iO_XGAMUNT_Real.icc`, and `AAA/xgamut_model.pkl`.
- The txt channel mapping is `7CLR_1=Cyan`, `7CLR_2=Magenta`, `7CLR_3=Yellow`, `7CLR_4=Black`, `7CLR_5=Orange`, `7CLR_6=Green`, `7CLR_7=Violet`.
- 7CLR values are percentages in the `0-100` range; do not expose `0-1` values in parser, runner, CSV, or model rows.
- ICC conversion must use a path that supports a 7-channel printer profile with PCS Lab.
- Prediction differences from txt recipes are observation data only, not a pass/fail accuracy metric.
- First run targets 20-50 sampled rows and writes `AAA/outputs/icc_sample_validation.csv`.
- Current environment checks showed `git` and `xicclu` are not on PATH. Commit steps are included for normal development; if `git` is unavailable, record the command failure and continue.

---

## File Structure

- Create `AAA/requirements.txt`: Python package dependencies for tests and model loading.
- Create `AAA/icc_validation/__init__.py`: package marker and exported constants.
- Create `AAA/icc_validation/txt_chart.py`: parse CGS/XGamut txt chart into structured 7CLR rows.
- Create `AAA/icc_validation/cmm.py`: call `xicclu` to convert 7CLR device values to Lab.
- Create `AAA/icc_validation/model.py`: load `xgamut_model.pkl` and predict 7CLR from Lab rows.
- Create `AAA/icc_validation/runner.py`: orchestrate parsing, sampling, conversion, prediction, and CSV writing.
- Create `AAA/run_icc_sample_validation.py`: command-line entry point.
- Create `AAA/tests/test_txt_chart.py`: parser tests.
- Create `AAA/tests/test_cmm.py`: CMM wrapper unit tests with a fake subprocess runner.
- Create `AAA/tests/test_model.py`: model feature and prediction tests with a fake model.
- Create `AAA/tests/test_runner.py`: CSV and orchestration tests with fake dependency functions.
- Create `AAA/README-icc-validation.md`: short usage notes and required external `xicclu` dependency.

## External Reference

- ArgyllCMS `xicclu` documentation: `https://www.argyllcms.com/doc/xicclu.html`

---

### Task 1: Add Package Skeleton and Txt Parser

**Files:**
- Create: `AAA/requirements.txt`
- Create: `AAA/icc_validation/__init__.py`
- Create: `AAA/icc_validation/txt_chart.py`
- Create: `AAA/tests/test_txt_chart.py`

**Interfaces:**
- Produces: `INK_CHANNELS: tuple[str, ...]`
- Produces: `TXT_TO_CHANNEL: dict[str, str]`
- Produces: `parse_7clr_txt(path: str | pathlib.Path) -> list[dict[str, object]]`
- Consumes: no earlier task interfaces

- [ ] **Step 1: Create Python dependency file**

Create `AAA/requirements.txt`:

```text
joblib
numpy
pandas
pytest
scikit-learn
```

- [ ] **Step 2: Write failing parser tests**

Create `AAA/tests/test_txt_chart.py`:

```python
from pathlib import Path

import pytest

from icc_validation.txt_chart import INK_CHANNELS, parse_7clr_txt


def write_chart(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "chart.txt"
    path.write_text(body, encoding="utf-8")
    return path


def test_parse_7clr_txt_maps_fields_to_named_channels(tmp_path):
    chart = write_chart(
        tmp_path,
        "\n".join(
            [
                'ORIGINATOR\t"X GAMUT"',
                "NUMBER_OF_FIELDS 9",
                "BEGIN_DATA_FORMAT",
                "SampleID\tSAMPLE_NAME\t7CLR_1\t7CLR_2\t7CLR_3\t7CLR_4\t7CLR_5\t7CLR_6\t7CLR_7",
                "END_DATA_FORMAT",
                "NUMBER_OF_SETS\t2",
                "BEGIN_DATA",
                "1\tA1\t20.00\t0.00\t0.00\t100.00\t0.00\t0.00\t0.00",
                "2\tA2\t0.00\t70.00\t0.00\t100.00\t0.00\t0.00\t0.00",
                "END_DATA",
            ]
        ),
    )

    rows = parse_7clr_txt(chart)

    assert len(rows) == 2
    assert rows[0]["SampleID"] == "1"
    assert rows[0]["SAMPLE_NAME"] == "A1"
    assert rows[0]["Cyan"] == 20.0
    assert rows[0]["Magenta"] == 0.0
    assert rows[0]["Yellow"] == 0.0
    assert rows[0]["Black"] == 100.0
    assert rows[0]["Orange"] == 0.0
    assert rows[0]["Green"] == 0.0
    assert rows[0]["Violet"] == 0.0
    assert tuple(INK_CHANNELS) == (
        "Cyan",
        "Magenta",
        "Yellow",
        "Black",
        "Orange",
        "Green",
        "Violet",
    )


def test_parse_7clr_txt_rejects_missing_channel(tmp_path):
    chart = write_chart(
        tmp_path,
        "\n".join(
            [
                "BEGIN_DATA_FORMAT",
                "SampleID\tSAMPLE_NAME\t7CLR_1\t7CLR_2",
                "END_DATA_FORMAT",
                "BEGIN_DATA",
                "1\tA1\t20.00\t0.00",
                "END_DATA",
            ]
        ),
    )

    with pytest.raises(ValueError, match="Missing required fields"):
        parse_7clr_txt(chart)


def test_parse_7clr_txt_rejects_out_of_range_channel(tmp_path):
    chart = write_chart(
        tmp_path,
        "\n".join(
            [
                "BEGIN_DATA_FORMAT",
                "SampleID\tSAMPLE_NAME\t7CLR_1\t7CLR_2\t7CLR_3\t7CLR_4\t7CLR_5\t7CLR_6\t7CLR_7",
                "END_DATA_FORMAT",
                "BEGIN_DATA",
                "1\tA1\t101.00\t0.00\t0.00\t100.00\t0.00\t0.00\t0.00",
                "END_DATA",
            ]
        ),
    )

    with pytest.raises(ValueError, match="outside 0-100"):
        parse_7clr_txt(chart)
```

- [ ] **Step 3: Run parser tests and verify they fail**

Run from `AAA`:

```powershell
python -m pytest tests/test_txt_chart.py -v
```

Expected: FAIL because `icc_validation.txt_chart` does not exist yet.

- [ ] **Step 4: Implement package constants**

Create `AAA/icc_validation/__init__.py`:

```python
INK_CHANNELS = (
    "Cyan",
    "Magenta",
    "Yellow",
    "Black",
    "Orange",
    "Green",
    "Violet",
)

TXT_TO_CHANNEL = {
    "7CLR_1": "Cyan",
    "7CLR_2": "Magenta",
    "7CLR_3": "Yellow",
    "7CLR_4": "Black",
    "7CLR_5": "Orange",
    "7CLR_6": "Green",
    "7CLR_7": "Violet",
}
```

- [ ] **Step 5: Implement txt parser**

Create `AAA/icc_validation/txt_chart.py`:

```python
from pathlib import Path

from . import INK_CHANNELS, TXT_TO_CHANNEL


def _find_line(lines: list[str], marker: str) -> int:
    for index, line in enumerate(lines):
        if line.strip() == marker:
            return index
    raise ValueError(f"Missing marker: {marker}")


def _split_fields(line: str) -> list[str]:
    return line.strip().split()


def parse_7clr_txt(path: str | Path) -> list[dict[str, object]]:
    txt_path = Path(path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Txt chart not found: {txt_path}")

    lines = txt_path.read_text(encoding="utf-8").splitlines()
    format_start = _find_line(lines, "BEGIN_DATA_FORMAT")
    format_end = _find_line(lines, "END_DATA_FORMAT")
    data_start = _find_line(lines, "BEGIN_DATA")
    data_end = _find_line(lines, "END_DATA")

    if format_end <= format_start + 1:
        raise ValueError("No data format row found")
    if data_end <= data_start + 1:
        raise ValueError("No data rows found")

    fields = _split_fields(lines[format_start + 1])
    required = ["SampleID", "SAMPLE_NAME", *TXT_TO_CHANNEL.keys()]
    missing = [name for name in required if name not in fields]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(lines[data_start + 1 : data_end], start=data_start + 2):
        if not line.strip():
            continue

        values = _split_fields(line)
        if len(values) != len(fields):
            raise ValueError(
                f"Malformed data row at line {line_number}: expected {len(fields)} fields, got {len(values)}"
            )

        raw = dict(zip(fields, values))
        row: dict[str, object] = {
            "SampleID": raw["SampleID"],
            "SAMPLE_NAME": raw["SAMPLE_NAME"],
        }

        for txt_field, channel in TXT_TO_CHANNEL.items():
            value = float(raw[txt_field])
            if value < 0 or value > 100:
                raise ValueError(
                    f"Channel {txt_field} for sample {row['SAMPLE_NAME']} is outside 0-100: {value}"
                )
            row[channel] = value

        rows.append(row)

    return rows
```

- [ ] **Step 6: Run parser tests and verify they pass**

Run from `AAA`:

```powershell
python -m pytest tests/test_txt_chart.py -v
```

Expected: PASS for all parser tests.

- [ ] **Step 7: Run parser against real chart**

Run from `AAA`:

```powershell
python -c "from icc_validation.txt_chart import parse_7clr_txt; rows=parse_7clr_txt('CMYKOGV_i1 Pro3 iO_XGAMUNT.txt'); print(len(rows)); print(rows[0])"
```

Expected: first line prints `3024`; second line includes `SampleID: '1'`, `SAMPLE_NAME: 'A1'`, `Cyan: 20.0`, and `Black: 100.0`.

- [ ] **Step 8: Commit parser task**

```powershell
git add requirements.txt icc_validation/__init__.py icc_validation/txt_chart.py tests/test_txt_chart.py
git commit -m "feat: parse 7clr chart txt"
```

Expected if `git` is installed: commit succeeds. Expected in the current environment: `git` command is not found; record that commit was skipped.

---

### Task 2: Add 7CLR ICC Conversion Wrapper

**Files:**
- Create: `AAA/icc_validation/cmm.py`
- Create: `AAA/tests/test_cmm.py`

**Interfaces:**
- Consumes: `INK_CHANNELS`
- Produces: `CmmError(Exception)`
- Produces: `convert_7clr_to_lab(rows: list[dict[str, object]], icc_path: str | pathlib.Path, xicclu_path: str = "xicclu") -> list[dict[str, object]]`

- [ ] **Step 1: Write failing CMM wrapper tests**

Create `AAA/tests/test_cmm.py`:

```python
from pathlib import Path
from types import SimpleNamespace

import pytest

from icc_validation.cmm import CmmError, _build_xicclu_input, _parse_lab_output, convert_7clr_to_lab


def sample_rows():
    return [
        {
            "SampleID": "1",
            "SAMPLE_NAME": "A1",
            "Cyan": 20.0,
            "Magenta": 0.0,
            "Yellow": 0.0,
            "Black": 100.0,
            "Orange": 0.0,
            "Green": 0.0,
            "Violet": 0.0,
        }
    ]


def test_build_xicclu_input_uses_0_to_100_values():
    text = _build_xicclu_input(sample_rows())

    assert text == "20.000000 0.000000 0.000000 100.000000 0.000000 0.000000 0.000000\n"


def test_parse_lab_output_returns_lab_rows():
    labs = _parse_lab_output("45.1 2.2 -3.3\n", expected_count=1)

    assert labs == [{"Lab_L": 45.1, "Lab_a": 2.2, "Lab_b": -3.3}]


def test_parse_lab_output_rejects_wrong_count():
    with pytest.raises(CmmError, match="Expected 2 Lab rows"):
        _parse_lab_output("45.1 2.2 -3.3\n", expected_count=2)


def test_convert_7clr_to_lab_calls_xicclu_with_expected_options(tmp_path, monkeypatch):
    icc_path = tmp_path / "device.icc"
    icc_path.write_bytes(b"fake")
    calls = []

    def fake_run(command, input, text, capture_output, check):
        calls.append(
            {
                "command": command,
                "input": input,
                "text": text,
                "capture_output": capture_output,
                "check": check,
            }
        )
        return SimpleNamespace(stdout="45.1 2.2 -3.3\n", stderr="", returncode=0)

    monkeypatch.setattr("icc_validation.cmm.subprocess.run", fake_run)

    converted = convert_7clr_to_lab(sample_rows(), icc_path, xicclu_path="xicclu.exe")

    assert calls[0]["command"] == [
        "xicclu.exe",
        "-v0",
        "-ff",
        "-ir",
        "-pl",
        "-s",
        "100",
        str(icc_path),
    ]
    assert calls[0]["input"].startswith("20.000000 0.000000")
    assert converted[0]["Lab_L"] == 45.1
    assert converted[0]["Lab_a"] == 2.2
    assert converted[0]["Lab_b"] == -3.3


def test_convert_7clr_to_lab_reports_missing_xicclu(tmp_path, monkeypatch):
    icc_path = tmp_path / "device.icc"
    icc_path.write_bytes(b"fake")

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("xicclu")

    monkeypatch.setattr("icc_validation.cmm.subprocess.run", fake_run)

    with pytest.raises(CmmError, match="xicclu executable not found"):
        convert_7clr_to_lab(sample_rows(), icc_path, xicclu_path="xicclu")
```

- [ ] **Step 2: Run CMM tests and verify they fail**

Run from `AAA`:

```powershell
python -m pytest tests/test_cmm.py -v
```

Expected: FAIL because `icc_validation.cmm` does not exist yet.

- [ ] **Step 3: Implement CMM wrapper**

Create `AAA/icc_validation/cmm.py`:

```python
from __future__ import annotations

from pathlib import Path
import subprocess

from . import INK_CHANNELS


class CmmError(RuntimeError):
    pass


def _build_xicclu_input(rows: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for row in rows:
        values = []
        for channel in INK_CHANNELS:
            value = float(row[channel])
            if value < 0 or value > 100:
                raise CmmError(f"{channel} is outside 0-100 for sample {row.get('SAMPLE_NAME')}: {value}")
            values.append(f"{value:.6f}")
        lines.append(" ".join(values))
    return "\n".join(lines) + "\n"


def _parse_lab_output(output: str, expected_count: int) -> list[dict[str, float]]:
    labs: list[dict[str, float]] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) < 3:
            raise CmmError(f"Could not parse Lab output line: {line}")
        try:
            lab_l, lab_a, lab_b = (float(parts[0]), float(parts[1]), float(parts[2]))
        except ValueError as exc:
            raise CmmError(f"Could not parse Lab output line: {line}") from exc
        labs.append({"Lab_L": lab_l, "Lab_a": lab_a, "Lab_b": lab_b})

    if len(labs) != expected_count:
        raise CmmError(f"Expected {expected_count} Lab rows from xicclu, got {len(labs)}")
    return labs


def convert_7clr_to_lab(
    rows: list[dict[str, object]],
    icc_path: str | Path,
    xicclu_path: str = "xicclu",
) -> list[dict[str, object]]:
    profile = Path(icc_path)
    if not profile.exists():
        raise FileNotFoundError(f"ICC profile not found: {profile}")
    if not rows:
        return []

    command = [
        xicclu_path,
        "-v0",
        "-ff",
        "-ir",
        "-pl",
        "-s",
        "100",
        str(profile),
    ]
    input_text = _build_xicclu_input(rows)

    try:
        result = subprocess.run(
            command,
            input=input_text,
            text=True,
            capture_output=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise CmmError(
            "xicclu executable not found. Install ArgyllCMS and ensure xicclu is on PATH, "
            "or pass --xicclu with the executable path."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else str(exc)
        raise CmmError(f"xicclu conversion failed: {stderr}") from exc

    labs = _parse_lab_output(result.stdout, expected_count=len(rows))
    converted: list[dict[str, object]] = []
    for row, lab in zip(rows, labs):
        merged = dict(row)
        merged.update(lab)
        converted.append(merged)
    return converted
```

- [ ] **Step 4: Run CMM tests and verify they pass**

Run from `AAA`:

```powershell
python -m pytest tests/test_cmm.py -v
```

Expected: PASS for all CMM unit tests.

- [ ] **Step 5: Run real one-row ICC smoke test when `xicclu` is installed**

Run from `AAA`:

```powershell
python -c "from icc_validation.txt_chart import parse_7clr_txt; from icc_validation.cmm import convert_7clr_to_lab; rows=parse_7clr_txt('CMYKOGV_i1 Pro3 iO_XGAMUNT.txt')[:1]; print(convert_7clr_to_lab(rows, 'CMYKOGV_i1 Pro3 iO_XGAMUNT_Real.icc')[0])"
```

Expected if `xicclu` is installed: prints one row with `Lab_L`, `Lab_a`, and `Lab_b`. Expected in the current environment before installing ArgyllCMS: clear `CmmError` saying `xicclu executable not found`.

- [ ] **Step 6: Commit CMM task**

```powershell
git add icc_validation/cmm.py tests/test_cmm.py
git commit -m "feat: convert 7clr recipes through icc"
```

Expected if `git` is installed: commit succeeds. Expected in the current environment: `git` command is not found; record that commit was skipped.

---

### Task 3: Add Model Loading and Lab-to-7CLR Prediction

**Files:**
- Create: `AAA/icc_validation/model.py`
- Create: `AAA/tests/test_model.py`

**Interfaces:**
- Consumes: rows containing `Lab_L`, `Lab_a`, `Lab_b`
- Produces: `load_xgamut_model(model_path: str | pathlib.Path) -> dict[str, object]`
- Produces: `predict_7clr_from_lab(model_package: dict[str, object], lab_rows: list[dict[str, object]]) -> list[dict[str, object]]`

- [ ] **Step 1: Write failing model tests**

Create `AAA/tests/test_model.py`:

```python
import math

import numpy as np
import pytest

from icc_validation.model import _build_feature_frame, predict_7clr_from_lab


class FakeModel:
    def predict(self, frame):
        assert list(frame.columns) == ["L", "a", "b", "C", "h"]
        assert frame.iloc[0]["L"] == 50.0
        assert frame.iloc[0]["a"] == 3.0
        assert frame.iloc[0]["b"] == 4.0
        assert frame.iloc[0]["C"] == 5.0
        assert math.isclose(frame.iloc[0]["h"], 53.13010235415598)
        return np.array([[10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]])


def package():
    return {
        "model": FakeModel(),
        "feature_cols": ["L", "a", "b", "C", "h"],
        "target_cols": ["Cyan", "Magenta", "Yellow", "Black", "Orange", "Green", "Violet"],
    }


def test_build_feature_frame_computes_chroma_and_hue():
    frame = _build_feature_frame(package(), [{"Lab_L": 50.0, "Lab_a": 3.0, "Lab_b": 4.0}])

    assert list(frame.columns) == ["L", "a", "b", "C", "h"]
    assert frame.iloc[0]["C"] == 5.0
    assert math.isclose(frame.iloc[0]["h"], 53.13010235415598)


def test_predict_7clr_from_lab_appends_prediction_columns():
    rows = [{"SampleID": "1", "SAMPLE_NAME": "A1", "Lab_L": 50.0, "Lab_a": 3.0, "Lab_b": 4.0}]

    predicted = predict_7clr_from_lab(package(), rows)

    assert predicted[0]["Pred_Cyan"] == 10.0
    assert predicted[0]["Pred_Magenta"] == 20.0
    assert predicted[0]["Pred_Violet"] == 70.0


def test_predict_7clr_from_lab_clips_predictions_to_0_100():
    class ClippingModel:
        def predict(self, frame):
            return np.array([[-1.0, 0.0, 50.0, 101.0, 120.0, 88.8, 1.2]])

    model_package = package()
    model_package["model"] = ClippingModel()

    predicted = predict_7clr_from_lab(
        model_package,
        [{"SampleID": "1", "SAMPLE_NAME": "A1", "Lab_L": 50.0, "Lab_a": 3.0, "Lab_b": 4.0}],
    )

    assert predicted[0]["Pred_Cyan"] == 0.0
    assert predicted[0]["Pred_Black"] == 100.0
    assert predicted[0]["Pred_Orange"] == 100.0


def test_predict_7clr_from_lab_rejects_wrong_feature_columns():
    bad_package = package()
    bad_package["feature_cols"] = ["L", "a", "b"]

    with pytest.raises(ValueError, match="Unexpected feature columns"):
        predict_7clr_from_lab(bad_package, [{"Lab_L": 50.0, "Lab_a": 3.0, "Lab_b": 4.0}])
```

- [ ] **Step 2: Run model tests and verify they fail**

Run from `AAA`:

```powershell
python -m pytest tests/test_model.py -v
```

Expected: FAIL because `icc_validation.model` does not exist yet.

- [ ] **Step 3: Implement model wrapper**

Create `AAA/icc_validation/model.py`:

```python
from __future__ import annotations

from pathlib import Path
import math

import joblib
import numpy as np
import pandas as pd

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
```

- [ ] **Step 4: Run model tests and verify they pass**

Run from `AAA`:

```powershell
python -m pytest tests/test_model.py -v
```

Expected: PASS for all model tests.

- [ ] **Step 5: Run real model load smoke test**

Run from `AAA`:

```powershell
python -c "from icc_validation.model import load_xgamut_model; package=load_xgamut_model('xgamut_model.pkl'); print(package['feature_cols']); print(package['target_cols'])"
```

Expected after Python dependencies are installed: first line prints `['L', 'a', 'b', 'C', 'h']`; second line prints `['Cyan', 'Magenta', 'Yellow', 'Black', 'Orange', 'Green', 'Violet']`.

- [ ] **Step 6: Commit model task**

```powershell
git add icc_validation/model.py tests/test_model.py
git commit -m "feat: predict 7clr from lab"
```

Expected if `git` is installed: commit succeeds. Expected in the current environment: `git` command is not found; record that commit was skipped.

---

### Task 4: Add Runner, CLI, and CSV Output

**Files:**
- Create: `AAA/icc_validation/runner.py`
- Create: `AAA/run_icc_sample_validation.py`
- Create: `AAA/tests/test_runner.py`

**Interfaces:**
- Consumes: `parse_7clr_txt(path)`
- Consumes: `convert_7clr_to_lab(rows, icc_path, xicclu_path="xicclu")`
- Consumes: `load_xgamut_model(model_path)`
- Consumes: `predict_7clr_from_lab(model_package, lab_rows)`
- Produces: `write_validation_csv(rows: list[dict[str, object]], output_path: str | pathlib.Path) -> pathlib.Path`
- Produces: `run_sample_validation(txt_path, icc_path, model_path, output_path, sample_size=50, xicclu_path="xicclu") -> pathlib.Path`

- [ ] **Step 1: Write failing runner tests**

Create `AAA/tests/test_runner.py`:

```python
import csv
from pathlib import Path

from icc_validation.runner import OUTPUT_COLUMNS, write_validation_csv


def test_write_validation_csv_writes_expected_columns(tmp_path):
    output_path = tmp_path / "out.csv"
    rows = [
        {
            "SampleID": "1",
            "SAMPLE_NAME": "A1",
            "Cyan": 20.0,
            "Magenta": 0.0,
            "Yellow": 0.0,
            "Black": 100.0,
            "Orange": 0.0,
            "Green": 0.0,
            "Violet": 0.0,
            "Lab_L": 45.1,
            "Lab_a": 2.2,
            "Lab_b": -3.3,
            "Pred_Cyan": 10.0,
            "Pred_Magenta": 20.0,
            "Pred_Yellow": 30.0,
            "Pred_Black": 40.0,
            "Pred_Orange": 50.0,
            "Pred_Green": 60.0,
            "Pred_Violet": 70.0,
        }
    ]

    written = write_validation_csv(rows, output_path)

    assert written == output_path
    with output_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == OUTPUT_COLUMNS
        data = list(reader)
    assert data[0]["SampleID"] == "1"
    assert data[0]["Lab_L"] == "45.1"
    assert data[0]["Pred_Violet"] == "70.0"
```

- [ ] **Step 2: Run runner tests and verify they fail**

Run from `AAA`:

```powershell
python -m pytest tests/test_runner.py -v
```

Expected: FAIL because `icc_validation.runner` does not exist yet.

- [ ] **Step 3: Implement runner**

Create `AAA/icc_validation/runner.py`:

```python
from __future__ import annotations

from pathlib import Path
import csv

from . import INK_CHANNELS
from .cmm import convert_7clr_to_lab
from .model import load_xgamut_model, predict_7clr_from_lab
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
    xicclu_path: str = "xicclu",
) -> Path:
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than 0")

    rows = parse_7clr_txt(txt_path)
    sample = rows[:sample_size]
    converted = convert_7clr_to_lab(sample, icc_path, xicclu_path=xicclu_path)
    model_package = load_xgamut_model(model_path)
    predicted = predict_7clr_from_lab(model_package, converted)
    return write_validation_csv(predicted, output_path)
```

- [ ] **Step 4: Implement CLI entry point**

Create `AAA/run_icc_sample_validation.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path

from icc_validation.runner import run_sample_validation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a sample 7CLR ICC -> Lab -> model validation chain.",
    )
    parser.add_argument("--txt", default="CMYKOGV_i1 Pro3 iO_XGAMUNT.txt")
    parser.add_argument("--icc", default="CMYKOGV_i1 Pro3 iO_XGAMUNT_Real.icc")
    parser.add_argument("--model", default="xgamut_model.pkl")
    parser.add_argument("--output", default="outputs/icc_sample_validation.csv")
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--xicclu", default="xicclu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output = run_sample_validation(
        txt_path=Path(args.txt),
        icc_path=Path(args.icc),
        model_path=Path(args.model),
        output_path=Path(args.output),
        sample_size=args.sample_size,
        xicclu_path=args.xicclu,
    )
    print(f"Wrote validation CSV: {output}")
    print("Prediction columns are observational only; they are not a pass/fail accuracy metric.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run runner tests and verify they pass**

Run from `AAA`:

```powershell
python -m pytest tests/test_runner.py -v
```

Expected: PASS for runner tests.

- [ ] **Step 6: Run all unit tests**

Run from `AAA`:

```powershell
python -m pytest tests -v
```

Expected: PASS for parser, CMM wrapper, model wrapper, and runner unit tests.

- [ ] **Step 7: Run CLI with real files after dependencies and `xicclu` are installed**

Run from `AAA`:

```powershell
python run_icc_sample_validation.py --sample-size 50
```

Expected if dependencies and `xicclu` are installed: prints `Wrote validation CSV: outputs\icc_sample_validation.csv` and creates `AAA/outputs/icc_sample_validation.csv`.

- [ ] **Step 8: Commit runner task**

```powershell
git add icc_validation/runner.py run_icc_sample_validation.py tests/test_runner.py
git commit -m "feat: add icc validation runner"
```

Expected if `git` is installed: commit succeeds. Expected in the current environment: `git` command is not found; record that commit was skipped.

---

### Task 5: Add Usage Notes and Run the Sample Validation

**Files:**
- Create: `AAA/README-icc-validation.md`
- Generate at runtime: `AAA/outputs/icc_sample_validation.csv`

**Interfaces:**
- Consumes: `run_icc_sample_validation.py`
- Produces: documented command sequence for dependency installation, `xicclu` verification, sample run, and full-chart run

- [ ] **Step 1: Create usage notes**

Create `AAA/README-icc-validation.md`:

```markdown
# ICC 7CLR Validation

This folder contains a first validation chain for the 7-color CGS/XGamut profile:

```text
CMYKOGV_i1 Pro3 iO_XGAMUNT.txt -> CMYKOGV_i1 Pro3 iO_XGAMUNT_Real.icc -> Lab -> xgamut_model.pkl -> predicted 7CLR
```

The txt values are 7CLR device percentages in the `0-100` range. The predicted 7CLR values are observational only and are not a pass/fail accuracy metric.

## Python Dependencies

Install Python dependencies from this directory:

```powershell
python -m pip install -r requirements.txt
```

## External CMM Dependency

The ICC profile device space is `7CLR`, so the validation script uses ArgyllCMS `xicclu` for ICC conversion.

Check that `xicclu` is available:

```powershell
where.exe xicclu
```

If `xicclu` is not on PATH, pass its full path:

```powershell
python run_icc_sample_validation.py --xicclu "C:\Path\To\xicclu.exe"
```

## Run a 50-Row Sample

```powershell
python run_icc_sample_validation.py --sample-size 50
```

Expected output:

```text
Wrote validation CSV: outputs\icc_sample_validation.csv
Prediction columns are observational only; they are not a pass/fail accuracy metric.
```

## Run the Full 3024-Row Chart

```powershell
python run_icc_sample_validation.py --sample-size 3024 --output outputs/icc_full_validation.csv
```

## Output Columns

- `SampleID`
- `SAMPLE_NAME`
- `Cyan`
- `Magenta`
- `Yellow`
- `Black`
- `Orange`
- `Green`
- `Violet`
- `Lab_L`
- `Lab_a`
- `Lab_b`
- `Pred_Cyan`
- `Pred_Magenta`
- `Pred_Yellow`
- `Pred_Black`
- `Pred_Orange`
- `Pred_Green`
- `Pred_Violet`
```

- [ ] **Step 2: Install Python dependencies**

Run from `AAA`:

```powershell
python -m pip install -r requirements.txt
```

Expected: `joblib`, `numpy`, `pandas`, `pytest`, and `scikit-learn` install successfully or are already satisfied.

- [ ] **Step 3: Verify `xicclu` availability**

Run from `AAA`:

```powershell
where.exe xicclu
```

Expected after ArgyllCMS is installed and on PATH: prints the path to `xicclu.exe`. Expected before installing ArgyllCMS: `INFO: Could not find files for the given pattern(s).`

- [ ] **Step 4: Run all unit tests**

Run from `AAA`:

```powershell
python -m pytest tests -v
```

Expected: all tests pass.

- [ ] **Step 5: Run one-row real chain**

Run from `AAA`:

```powershell
python run_icc_sample_validation.py --sample-size 1 --output outputs/icc_one_row_validation.csv
```

Expected with `xicclu` installed: writes `outputs/icc_one_row_validation.csv` with one data row.

- [ ] **Step 6: Run 50-row sample chain**

Run from `AAA`:

```powershell
python run_icc_sample_validation.py --sample-size 50
```

Expected with `xicclu` installed: writes `outputs/icc_sample_validation.csv` with 50 data rows.

- [ ] **Step 7: Inspect CSV header and row count**

Run from `AAA`:

```powershell
python -c "import csv; p='outputs/icc_sample_validation.csv'; rows=list(csv.DictReader(open(p, encoding='utf-8-sig'))); print(len(rows)); print(rows[0].keys())"
```

Expected: first line prints `50`; second line includes `SampleID`, `SAMPLE_NAME`, `Lab_L`, `Lab_a`, `Lab_b`, and `Pred_Violet`.

- [ ] **Step 8: Commit docs and generated sample output decision**

Do not commit generated CSV files unless the user explicitly wants sample output versioned. Commit the usage notes:

```powershell
git add README-icc-validation.md
git commit -m "docs: explain icc validation run"
```

Expected if `git` is installed: commit succeeds. Expected in the current environment: `git` command is not found; record that commit was skipped.

---

## Self-Review

**Spec coverage:** The plan covers txt parsing, 7CLR-to-Lab ICC conversion, model loading, Lab-to-7CLR prediction, CSV output, 20-50 row sampling, and explicit handling that prediction differences are observational only. PDF `DeviceN` parsing, web UI, arbitrary upload, multiple ICC files, CMYK input PDFs, and retraining are excluded from implementation tasks as required.

**Placeholder scan:** The plan contains concrete file paths, function names, test code, implementation code, commands, and expected results. It avoids open-ended implementation instructions.

**Type consistency:** `parse_7clr_txt` returns `list[dict[str, object]]`; `convert_7clr_to_lab` consumes and returns the same row shape with `Lab_L/Lab_a/Lab_b`; `predict_7clr_from_lab` consumes Lab rows and appends `Pred_*`; `run_sample_validation` returns the written `Path`. These signatures are consistent across tasks.

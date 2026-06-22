# ICC 7CLR Validation

This folder contains a first validation chain for the 7-color CGS/XGamut profile:

```text
CMYKOGV_i1 Pro3 iO_XGAMUNT.txt -> CMYKOGV_i1 Pro3 iO_XGAMUNT_Real.icc -> Lab -> xgamut model -> predicted 7CLR
```

The txt values are 7CLR device percentages in the `0-100` range. The predicted 7CLR values are observational only and are not a pass/fail accuracy metric.

## Python Dependencies

Install Python dependencies from this directory:

```powershell
python -m pip install -r requirements.txt
```

The original `xgamut_model.pkl` was saved by an older scikit-learn version. If it cannot be loaded, the runner retrains the same random-forest model logic from the existing Excel files and saves a current-environment cache:

```text
xgamut_model_current.pkl
```

The original `xgamut_model.pkl` is not overwritten.

## External CMM Dependency

The ICC profile device space is `7CLR`, so the validation script uses LittleCMS `transicc` for ICC conversion. RGB-only or CMYK-only Python ICC APIs are not enough for this profile.

The repository does not commit third-party binaries. In this workspace, LittleCMS was built locally at:

```text
tools-src/Little-CMS/bin/transicc.exe
```

The CLI uses that local path automatically when it exists. If you build or install LittleCMS somewhere else, pass the path explicitly:

```powershell
python run_icc_sample_validation.py --transicc "C:\Path\To\transicc.exe"
```

The old ArgyllCMS `xicclu` backend is still available for ICC v2 profiles, but this CGS/XGamut ICC is v4.3 and did not work with Argyll:

```powershell
python run_icc_sample_validation.py --cmm xicclu --xicclu "C:\Path\To\xicclu.exe"
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

If `transicc` is missing, expected output is:

```text
ICC conversion failed: transicc executable not found. Build LittleCMS transicc, ensure it is on PATH, or pass --transicc with the executable path.
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

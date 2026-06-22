# ICC 7CLR Validation Design

## Context

The current `AAA` project predicts 7-color separation recipes with a saved random forest model. The existing model takes `L`, `a`, `b`, derived `C`, and derived `h` as features, then predicts seven ink channels:

- Cyan
- Magenta
- Yellow
- Black
- Orange
- Green
- Violet

The project also contains a CGS/XGamut 7-color printer ICC profile and the measurement chart files used for that profile:

- `AAA/CMYKOGV_i1 Pro3 iO_XGAMUNT_Real.icc`
- `AAA/CMYKOGV_i1 Pro3 iO_XGAMUNT.txt`
- `AAA/CMYKOGV_i1 Pro3 iO_XGAMUNT.pdf`

The ICC profile declares:

- profile class: printer
- device color space: `7CLR`
- PCS: `Lab`
- creator: `CGS`

The text chart contains 3024 rows of 7-color input recipes. These recipe values are test-chart device inputs, not independent proof that the model is correct. The first validation stage should therefore use them to verify the ICC conversion and model wiring, not to make an accuracy claim.

## Goal

Build a small Python validation script that proves this chain can run:

```text
7CLR recipe from txt -> device ICC -> Lab -> existing random forest model -> predicted 7CLR recipe
```

The first run should sample 20-50 rows from the txt chart and write a CSV for inspection.

## Non-Goals

This stage will not:

- build a web UI
- upload arbitrary files
- parse arbitrary customer PDFs
- parse the `DeviceN` PDF chart yet
- support multiple ICC files through a picker
- claim final color accuracy
- use `txt 7CLR` versus predicted `7CLR` as a pass/fail metric
- train a new model
- use CMYK input PDFs

PDF `DeviceN` parsing is intentionally preserved as a later phase.

## Inputs

The validation script uses fixed local inputs:

- `AAA/CMYKOGV_i1 Pro3 iO_XGAMUNT.txt`
- `AAA/CMYKOGV_i1 Pro3 iO_XGAMUNT_Real.icc`
- `AAA/xgamut_model.pkl`

The txt channel order maps directly to model target names:

| TXT Field | Model Channel |
| --- | --- |
| `7CLR_1` | `Cyan` |
| `7CLR_2` | `Magenta` |
| `7CLR_3` | `Yellow` |
| `7CLR_4` | `Black` |
| `7CLR_5` | `Orange` |
| `7CLR_6` | `Green` |
| `7CLR_7` | `Violet` |

The 7CLR values are percentages in the `0-100` range. The main path should not treat them as `0-1` values.

## Architecture

The implementation should keep the validation chain split into small units.

### `parse_7clr_txt(path)`

Responsibilities:

- read the CGS/XGamut text file
- locate `BEGIN_DATA_FORMAT`, `END_DATA_FORMAT`, `BEGIN_DATA`, and `END_DATA`
- parse the field names and data rows
- return structured rows with `SampleID`, `SAMPLE_NAME`, and seven named ink channels
- validate that all seven channels are present
- coerce channel values to numeric percentages

Expected validation:

- parsed row count is 3024 for the current chart
- values outside `0-100` are reported
- malformed rows are reported with row number and sample name when available

### `convert_7clr_to_lab(rows, icc_path)`

Responsibilities:

- load the 7CLR printer ICC profile
- convert each row's seven device channels to PCS Lab
- append `Lab_L`, `Lab_a`, and `Lab_b` to each row

The conversion layer must use a color management path that supports multichannel printer ICC profiles. RGB-only or CMYK-only conversion APIs are not enough for this profile because the device color space is `7CLR`.

The script should keep the public row values as `0-100` percentages. If the chosen CMM API requires normalized values internally, that normalization must be isolated inside this function and documented in code comments.

Expected validation:

- ICC profile can be opened
- profile device space is compatible with 7 channels
- output Lab values are numeric
- conversion failures include sample ID and input channel values

### `load_xgamut_model(model_path)`

Responsibilities:

- load `xgamut_model.pkl`
- verify the package contains the saved model, feature columns, and target columns
- verify expected feature columns are `L`, `a`, `b`, `C`, and `h`
- verify expected target columns match the seven channel names

### `predict_7clr_from_lab(model_package, lab_rows)`

Responsibilities:

- compute `C = sqrt(a^2 + b^2)`
- compute hue angle `h = atan2(b, a)` in degrees, normalized to `0-360`
- call the loaded random forest model
- clip predicted channel values to `0-100`
- append `Pred_Cyan`, `Pred_Magenta`, `Pred_Yellow`, `Pred_Black`, `Pred_Orange`, `Pred_Green`, and `Pred_Violet`

The prediction is observation data for this stage. It should not be treated as proof that the model is accurate for the profile.

### `run_sample_validation(sample_size=50)`

Responsibilities:

- parse all txt rows
- choose the first `sample_size` rows by default
- optionally allow deterministic random sampling later
- run ICC conversion
- run model prediction
- write a CSV output

## Output

The first script output should be:

```text
AAA/outputs/icc_sample_validation.csv
```

The CSV should include:

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

No error columns are required in the first version. If a row cannot be converted, the script should fail loudly with enough context instead of silently writing partial data.

## Success Criteria

The first stage is successful when:

1. The txt chart is parsed into 3024 structured 7CLR rows.
2. A 20-50 row sample can be converted through the 7CLR ICC into Lab.
3. The existing saved model can predict 7CLR values from those Lab values.
4. `AAA/outputs/icc_sample_validation.csv` is written with the expected columns.
5. The script output makes it clear that prediction differences from txt recipes are informational only.

## Error Handling

The script should fail with clear messages for:

- missing txt, ICC, or model files
- unrecognized txt data format
- missing 7CLR columns
- channel values outside `0-100`
- ICC profile load failure
- ICC conversion failure
- model package missing expected keys
- model feature or target column mismatch

## Testing Strategy

Initial checks:

- parse the txt file and assert the row count is 3024
- assert row 1 maps to `SampleID=1`, `SAMPLE_NAME=A1`, and the expected seven channel values
- run ICC conversion for 1 row, then 20-50 rows
- run model prediction for the converted Lab rows
- verify the CSV exists and has the expected column names

The implementation should keep these checks lightweight because the goal is to validate the chain before building a full application.

## Later Phases

After the sample chain works:

1. Run all 3024 txt rows and write a full CSV.
2. Add summary statistics for predicted channels, Lab ranges, and conversion failures.
3. Parse the PDF chart's `DeviceN` 7-channel fill values and compare them with the txt rows to validate PDF extraction.
4. Add support for multiple built-in equipment ICC files.
5. Add measured Lab data when available and use it for real accuracy evaluation.
6. Revisit target-color workflows such as `Lab/RGB PDF -> device/profile-conditioned separation`.

## Key Assumptions

- The current ICC profile should be used as a 7-channel printer device profile whose PCS is Lab.
- The txt chart values are 7CLR percentage inputs in the `0-100` range.
- The saved random forest model remains the first model used for validation.
- This design is a validation step, not the final production separation workflow.

import csv

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

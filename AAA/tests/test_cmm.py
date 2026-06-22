from types import SimpleNamespace

import pytest

from icc_validation.cmm import (
    CmmError,
    _build_xicclu_input,
    _parse_lab_output,
    convert_7clr_to_lab,
    convert_7clr_to_lab_xicclu,
)


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


def test_convert_7clr_to_lab_calls_transicc_with_expected_options(tmp_path, monkeypatch):
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

    converted = convert_7clr_to_lab(sample_rows(), icc_path, transicc_path="transicc.exe")

    assert calls[0]["command"] == [
        "transicc.exe",
        f"-i{icc_path}",
        "-o*Lab",
        "-t1",
        "-n",
    ]
    assert calls[0]["input"].startswith("20.000000 0.000000")
    assert converted[0]["Lab_L"] == 45.1
    assert converted[0]["Lab_a"] == 2.2
    assert converted[0]["Lab_b"] == -3.3


def test_convert_7clr_to_lab_reports_missing_transicc(tmp_path, monkeypatch):
    icc_path = tmp_path / "device.icc"
    icc_path.write_bytes(b"fake")

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("xicclu")

    monkeypatch.setattr("icc_validation.cmm.subprocess.run", fake_run)

    with pytest.raises(CmmError, match="transicc executable not found"):
        convert_7clr_to_lab(sample_rows(), icc_path, transicc_path="transicc")


def test_convert_7clr_to_lab_xicclu_backend_is_still_available(tmp_path, monkeypatch):
    icc_path = tmp_path / "device.icc"
    icc_path.write_bytes(b"fake")
    calls = []

    def fake_run(command, input, text, capture_output, check):
        calls.append(command)
        return SimpleNamespace(stdout="45.1 2.2 -3.3\n", stderr="", returncode=0)

    monkeypatch.setattr("icc_validation.cmm.subprocess.run", fake_run)

    converted = convert_7clr_to_lab_xicclu(sample_rows(), icc_path, xicclu_path="xicclu.exe")

    assert calls[0] == [
        "xicclu.exe",
        "-v0",
        "-ff",
        "-ir",
        "-pl",
        "-s",
        "100",
        str(icc_path),
    ]
    assert converted[0]["Lab_L"] == 45.1

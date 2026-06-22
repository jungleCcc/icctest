import math

import numpy as np
import pytest

from icc_validation import model as model_module
from icc_validation.model import _build_feature_frame, load_or_train_xgamut_model, predict_7clr_from_lab


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


def test_load_or_train_xgamut_model_retrains_when_loading_fails(tmp_path, monkeypatch):
    model_path = tmp_path / "old.pkl"
    retrained_path = tmp_path / "retrained.pkl"
    model_path.write_bytes(b"not a compatible model")
    trained_package = package()
    calls = {}

    def fake_load(path):
        calls["load_path"] = path
        raise ValueError("incompatible pickle")

    def fake_train(std_path, ink_path, active_learning_path=None, model_output_path=None):
        calls["train_args"] = (std_path, ink_path, active_learning_path, model_output_path)
        return trained_package

    monkeypatch.setattr(model_module, "load_xgamut_model", fake_load)
    monkeypatch.setattr(model_module, "train_xgamut_model", fake_train)

    loaded = load_or_train_xgamut_model(
        model_path,
        std_path="std.xlsx",
        ink_path="ink.xlsx",
        active_learning_path="new.xlsx",
        retrained_model_path=retrained_path,
    )

    assert loaded is trained_package
    assert calls["load_path"] == model_path
    assert calls["train_args"] == ("std.xlsx", "ink.xlsx", "new.xlsx", retrained_path)

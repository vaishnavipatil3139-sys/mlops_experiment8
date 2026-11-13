import pandas as pd
import pytest

from ml_pipeline import clean_data, train_model, validate_predictions


def test_clean_data_removes_duplicates():
    data = pd.DataFrame({
        "Age": [20, 20, 21],
        "Marks": [70, 70, 80]
    })
    result = clean_data(data)
    assert len(result) == 2


def test_clean_data_fills_missing_values():
    data = pd.DataFrame({
        "Age": [20, None, 22],
        "Marks": [70, 80, 90]
    })
    result = clean_data(data)
    assert result.isnull().sum().sum() == 0


def test_training_pipeline():
    model, accuracy, X_test = train_model()
    assert 0 <= accuracy <= 1
    assert len(model.predict(X_test)) == len(X_test)


def test_model_output_validation_passes():
    assert validate_predictions([0, 1, 2], [0, 1, 2]) is True


def test_model_output_validation_rejects_invalid_label():
    with pytest.raises(ValueError):
        validate_predictions([0, 3], [0, 1, 2])


def test_model_output_validation_rejects_empty_output():
    with pytest.raises(ValueError):
        validate_predictions([], [0, 1, 2])
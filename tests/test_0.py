import pytest
from definition_e549142d6c804b95a664b3222b956435 import load_sample_data

@pytest.mark.parametrize("dataset_name, expected_type, expected_shape", [
    ("Boston", type, (n, m)),  # n and m are placeholders; use actual values if known
    ("Diabetes", type, (n, m)),
    ("Synthetic", type, (n, m)),
    ("UnknownDataset", type, (n, m)),  # Edge case: unknown dataset name
    ("", type, (n, m)),  # Edge case: empty string as dataset name
    (None, type, None),  # Edge case: None as dataset name
])

def test_load_sample_data(dataset_name, expected_type, expected_shape):
    try:
        df = load_sample_data(dataset_name)
        assert isinstance(df, expected_type)
        if expected_shape is not None:
            assert df.shape == expected_shape
    except Exception as e:
        # For unknown datasets or invalid inputs, exceptions are acceptable
        assert isinstance(e, (ValueError, TypeError, FileNotFoundError))

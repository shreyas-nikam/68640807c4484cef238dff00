import pytest
import pandas as pd
from definition_f17d94a4d5f244f4a7b612aad722e142 import load_sample_data # DO NOT REPLACE or REMOVE this block

@pytest.mark.parametrize("dataset_name, expected", [
    # Valid dataset names - expecting a non-empty pandas DataFrame
    ("Boston", lambda df: isinstance(df, pd.DataFrame) and not df.empty),
    ("Diabetes", lambda df: isinstance(df, pd.DataFrame) and not df.empty),
    ("Wine", lambda df: isinstance(df, pd.DataFrame) and not df.empty),
    ("Synthetic", lambda df: isinstance(df, pd.DataFrame) and not df.empty),

    # Invalid dataset names (string inputs that are not recognized) - expecting ValueError
    ("InvalidDatasetName", ValueError),
    ("boston", ValueError), # Case sensitivity
    ("unknown_dataset", ValueError),
    ("", ValueError), # Empty string

    # Invalid input types for dataset_name - expecting TypeError as the argument is type-hinted as 'str'
    (None, TypeError),
    (123, TypeError),
    (1.0, TypeError),
    (True, TypeError), # Boolean
    ([], TypeError), # Empty list
    (["Boston"], TypeError), # List with a string
    ({}, TypeError), # Empty dictionary
    ({"name": "Boston"}, TypeError), # Dictionary
    ((1, 2, 3), TypeError), # Tuple
])
def test_load_sample_data(dataset_name, expected):
    if isinstance(expected, type) and issubclass(expected, Exception):
        # If 'expected' is an exception type, we assert that the function raises that exception
        with pytest.raises(expected) as excinfo:
            load_sample_data(dataset_name)
        # Optional: Further checks on the error message can be added here if specific messages are part of the spec
        # For instance: if expected == ValueError: assert "Unknown dataset_name" in str(excinfo.value)
    else:
        # If 'expected' is a callable (like a lambda function), it means we expect a successful return
        # and 'expected' itself contains the assertion logic for the returned value.
        result = load_sample_data(dataset_name)
        assert expected(result), f"Test failed for dataset_name='{dataset_name}': " \
                                 f"Expected a non-empty pandas.DataFrame, but got {type(result)} or an empty DataFrame."

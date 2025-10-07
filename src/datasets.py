"""
datasets.py

Wrapper functions to load training and validation datasets, with optional debug
subset selection.
"""

from preproc_utils import get_combined_dataset

def load_train_dataset(modes, debug=False):
    """
    Load and return the training dataset.

    Args:
        modes (list[str]): List of dataset names to combine.
        debug (bool): If True, select only a small subset for debugging.

    Returns:
        Dataset: Combined training dataset.
    """
    dataset = get_combined_dataset(modes, "train")
    if debug:
        dataset = dataset.select(range(20))
    return dataset

def load_val_dataset(test_set, test=False, debug=False):
    """
    Load and return the validation or test dataset.

    Args:
        test_set (str): Name of the test set.
        test (bool): If True, load the test split instead of validation.
        debug (bool): If True, select only a small subset for debugging.

    Returns:
        Dataset: Validation or test dataset.
    """
    split = "test" if test else "val"
    dataset = get_combined_dataset([test_set], split)
    if debug:
        dataset = dataset.select(range(2))
    return dataset

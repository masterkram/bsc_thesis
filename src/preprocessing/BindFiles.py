from typing import List
from collections.abc import Callable
import numpy as np
from tqdm import tqdm


class BindFiles:
    def __init__(self, filenames: List[str], naming_function: Callable = None):
        self.filenames = filenames
        self.naming_function = naming_function

    def _saveFile(self, result: np.ndarray, filename: str):
        np.save(
            self.naming_function(filename),
            result,
        )

    def bind(self, f: Callable, desc: str = ""):
        for file in tqdm(self.filenames, desc=desc):
            result = f(file)
            self._saveFile(result, file)

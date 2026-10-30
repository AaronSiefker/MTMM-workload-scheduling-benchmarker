from __future__ import annotations

from typing import List

import numpy as np

def _generate_random_numbers() -> np.ndarray:
    rng = np.random.default_rng()
    return rng.random(100_000)

class StaticRng:
    _random_numbers : List[float] = _generate_random_numbers()

    def __init__(self):
        rng = np.random.default_rng()
        self._current_index = int(rng.random() * len(StaticRng._random_numbers))

    def random(self) -> float:
        out = StaticRng._random_numbers[self._current_index]
        if self._current_index == (len(StaticRng._random_numbers) - 1):
            self._current_index = 0
        else:
            self._current_index += 1
        return out

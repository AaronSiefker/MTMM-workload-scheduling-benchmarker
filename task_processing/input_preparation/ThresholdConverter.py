from typing import Dict, Final

import torch

from task_processing.input_preparation.ConfigurationInputConverter import ConfigurationInputConverter, \
    MultidimensionalArray


class ThresholdConverter(ConfigurationInputConverter):

    def __init__(self, threshold_key : str, threshold_value : float):
        self._threshold_key : Final[str] = threshold_key
        self._threshold_value : Final[torch.Tensor] = torch.Tensor([threshold_value])

    def convert_input(self, model_input: Dict[str, MultidimensionalArray]) -> Dict[str, MultidimensionalArray]:
        out = model_input.copy()
        out[self._threshold_key] = self._threshold_value
        return out
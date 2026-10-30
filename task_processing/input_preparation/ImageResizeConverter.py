from typing import Dict

import torch
from torchvision import transforms

from task_processing.input_preparation.ConfigurationInputConverter import ConfigurationInputConverter, \
    MultidimensionalArray, convert_hwc_to_bchw_if_applicable


class ImageResizeConverter(ConfigurationInputConverter):

    def __init__(self, image_key : str, target_size : int):
        self._image_key : str = image_key
        self._target_size : int = target_size
        self._resize = transforms.Resize((target_size, target_size))

    def convert_input(self, model_input: Dict[str, MultidimensionalArray]) -> Dict[str, MultidimensionalArray]:
        out = model_input.copy()
        image = torch.as_tensor(out[self._image_key])
        image = convert_hwc_to_bchw_if_applicable(image)
        if image.shape[-2:] != (self._target_size, self._target_size):
            image = self._resize(image)
        out[self._image_key] = image
        return out

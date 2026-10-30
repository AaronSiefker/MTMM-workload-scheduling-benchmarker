from typing import Dict

import torch
from torchvision import transforms

from task_processing.input_preparation.ConfigurationInputConverter import ConfigurationInputConverter, \
    MultidimensionalArray, convert_hwc_to_bchw_if_applicable


class ImageNetConverter(ConfigurationInputConverter):
    def __init__(self, image_key):
        self._image_key : str = image_key
        self._normalize = transforms.Normalize(
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225],
        )

    def convert_input(self, model_input: Dict[str, MultidimensionalArray]) -> Dict[str, MultidimensionalArray]:
        out = model_input.copy()
        image = torch.as_tensor(out[self._image_key])
        if image.dtype != torch.float32:
            image = image.to(torch.float32)
        image = convert_hwc_to_bchw_if_applicable(image)
        out[self._image_key] = self._normalize(image)
        return out
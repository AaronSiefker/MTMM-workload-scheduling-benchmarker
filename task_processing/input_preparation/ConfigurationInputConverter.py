from abc import ABC, abstractmethod
from typing import Dict, Protocol

import torch


class MultidimensionalArray(Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        ...

class ConfigurationInputConverter(ABC):
    @abstractmethod
    def convert_input(self, model_input : Dict[str, MultidimensionalArray]) -> Dict[str, MultidimensionalArray]:
        """
        converts the model input according to the conversion associated with the class this is called on.
        Note that the arrays in the returned dictionary may share memory with the original dictionary. As such,
        the changes may affect both the input and output dictionary
        :param model_input: the model input that needs to be converted
        :return: A new dictionary containing the converted input. Note that the arrays may share memory with the
        original dict, meaning changes are reflected in both
        """
        pass

def convert_hwc_to_bchw_if_applicable(image : torch.Tensor) -> torch.Tensor:
    """
    Converts an image loaded via PIL to BCHW. If the image is already in BCHW form, nothing happens
    :param image: the image to convert
    :return: the input image in form BCHW
    """
    if image.ndim == 3:
        # ensure batch dimension is present
        image = image.unsqueeze(0)
    # loaded in a way where channels in last dimension
    if image.shape[-1] == 3:
        image = image.permute(0, 3, 1, 2)
    return image
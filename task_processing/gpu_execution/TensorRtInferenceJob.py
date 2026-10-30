from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Final

import torch

from typing import TYPE_CHECKING

from task_processing.InputContainer import InputContainer
from task_processing.TaskConfiguration import TaskConfiguration

if TYPE_CHECKING:
    from task_processing.gpu_execution.TensorRtModel import TensorRtModel


@dataclass
class TensorRtInferenceJob:
    """
    Keeps all state necessary for an inference using a TensorRtModel. Ensures that no input accidentally goes out of scope
    before or during the inference.
    """
    request_id : Final[str]
    associated_config : Final[TaskConfiguration]
    tensor_rt_model : Final[TensorRtModel]
    stream : Final[torch.cuda.Stream]
    stream_index : Final[int]

    inputs : Final[InputContainer[torch.Tensor]]
    batch_size : int
    request_time : Final[float]
    deadline : Final[float]

    event : Final[torch.cuda.Event]
    outputs : Dict[str, torch.Tensor]

    consumed_energy : float = 0.0

    def execute_async(self) -> bool:
        return self.tensor_rt_model.infer(self)

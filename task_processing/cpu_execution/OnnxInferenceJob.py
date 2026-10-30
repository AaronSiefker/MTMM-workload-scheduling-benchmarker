from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Sequence, Final

import numpy as np
import onnxruntime
from onnxruntime import SparseTensor

from typing import TYPE_CHECKING

from task_processing.InputContainer import InputContainer
from task_processing.TaskConfiguration import TaskConfiguration

if TYPE_CHECKING:
    from task_processing.cpu_execution.OnnxModel import OnnxModel



class OnnxInferenceJobStatus(Enum):
    PREPARED = 0
    IN_PROGRESS = 1
    FINISHED = 2
    TERMINATED = 3

@dataclass
class OnnxInferenceJob:
    request_id : Final[str]
    associated_config : Final[TaskConfiguration]
    onnx_model : Final[OnnxModel]

    inputs : Final[InputContainer[np.ndarray]]
    request_time : Final[float]
    deadline : Final[float]
    run_options : Final[onnxruntime.RunOptions]

    status : OnnxInferenceJobStatus
    outputs : Sequence[np.ndarray | SparseTensor | list | dict]

    def execute(self) -> bool:
        return self.onnx_model.infer(self)

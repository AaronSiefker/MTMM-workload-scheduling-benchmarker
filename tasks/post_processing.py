from typing import Callable

import numpy as np
import torch

from task_processing.cpu_execution.OnnxInferenceJob import OnnxInferenceJob
from task_processing.gpu_execution.TensorRtInferenceJob import TensorRtInferenceJob


def gpu_post_processing_image_classification(output_key : str) -> Callable[[TensorRtInferenceJob], None]:
    def func(job : TensorRtInferenceJob):
        with torch.cuda.stream(job.stream):
            logits = job.outputs[output_key]
            _ = torch.argmax(logits, dim=1)
    return func

def cpu_post_processing_image_classification(output_index : int) -> Callable[[OnnxInferenceJob], None]:
    def func(job : OnnxInferenceJob):
        logits = job.outputs[output_index]
        _ = np.argmax(logits, axis=1)
    return func
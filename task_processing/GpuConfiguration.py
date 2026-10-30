import time
from abc import ABC, abstractmethod
from typing import Dict, Final, Callable

import torch

from task_processing.InputContainer import InputContainer
from task_processing.TaskConfiguration import TaskConfiguration
from task_processing.gpu_execution.TensorRtInferenceJob import TensorRtInferenceJob
from task_processing.input_preparation.ConfigurationInputConverter import MultidimensionalArray


class GpuConfiguration(ABC):
    def __init__(self, task_configuration : TaskConfiguration,
                 post_processing_callback : Callable[[TensorRtInferenceJob], None]):
        self.task_configuration : Final[TaskConfiguration] = task_configuration
        self._post_processing_callback : Callable[[TensorRtInferenceJob], None] = post_processing_callback

    def prepare_input(self, model_input : Dict[str, MultidimensionalArray]) -> tuple[InputContainer[torch.Tensor], float]:
        """
        Prepares the input for inference using this GpuConfiguration.
        Note that the tensors in the returned dictionary may share memory with the arrays in the original provided input.
        :param model_input: The input to prepare for inference
        :return: A dictionary representing a prepared input for this GpuConfiguration, the input preparation time in seconds
        """
        t1 = time.perf_counter()
        prepared_input = self.task_configuration.prepare_input(model_input)
        prepared_converted : Dict[str, torch.Tensor] = {}
        for key, value in prepared_input.items():
            prepared_converted[key] = torch.as_tensor(value)
        t2 = time.perf_counter()
        return InputContainer(model_input, prepared_converted), t2 - t1

    @abstractmethod
    def prepare_inference_job(self, request_id : str, input_container : InputContainer[torch.Tensor], batch_size : int,
                              request_time : float, deadline : float, stream : torch.cuda.Stream,
                              stream_index : int) -> TensorRtInferenceJob:
        pass

    @abstractmethod
    def run_inference(self, job : TensorRtInferenceJob) -> float:
        """
        Schedules the job for execution via tensorrt and blocks the thread until execution is finished.
        Once the execution is finished, the post-processing callback provided in the constructor for this instance
        is called.
        The outputs are stored in job.outputs.
        :param job: A job created via prepare_inference_job
        :return: The inference time in seconds
        """
        pass

    def execute(self, model_input : Dict[str, MultidimensionalArray], stream : torch.cuda.Stream,
                request_id : str = "", stream_index = 0, batch_size = 1) -> tuple[float, float]:
        """
        Performs inference for the given input. This method takes care of the input conversion to the configuration
        associated with this instance.
        The calling thread is blocked until inference is completed.
        Once the inference is finished, the post-processing callback provided in the constructor is called.
        :param batch_size: The input's batch size.
        :param stream_index: The cuda stream_index associated with the stream this is executed with
        :param request_id: The id associated with the request.
        :param model_input: The input to perform inference on.
        :param stream: The cuda stream the inference should be scheduled on.
        :return: the input preparation time in seconds, the inference time in seconds
        """
        input_container, input_preparation_time = self.prepare_input(model_input)
        job = self.prepare_inference_job(request_id, input_container, batch_size, -1, -1, stream, stream_index)
        inference_time = self.run_inference(job)
        return input_preparation_time, inference_time


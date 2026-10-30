import time
from typing import Dict, Final, Callable, Tuple

import numpy as np
from onnxruntime import RunOptions

from task_processing.InputContainer import InputContainer
from task_processing.TaskConfiguration import TaskConfiguration
from task_processing.cpu_execution.OnnxInferenceJob import OnnxInferenceJob
from task_processing.cpu_execution.OnnxModel import OnnxModel
from task_processing.input_preparation.ConfigurationInputConverter import MultidimensionalArray


class CpuConfiguration:
    def __init__(self, task_configuration : TaskConfiguration, onnx_model : OnnxModel,
                 post_processing_callback : Callable[[OnnxInferenceJob], None]):
        self.task_configuration : Final[TaskConfiguration] = task_configuration
        self._onnx_model : OnnxModel = onnx_model
        self._post_processing_callback : Callable[[OnnxInferenceJob], None] = post_processing_callback

    def prepare_input(self, model_input : Dict[str, MultidimensionalArray]) -> Tuple[InputContainer[np.ndarray], float]:
        """
        Prepares the input for inference using this CpuConfiguration.
        :param model_input: The input to prepare for inference
        :return: A dictionary containing a prepared input for this CpuConfiguration, the input preparation time in seconds
        """
        t1 = time.perf_counter()
        prepared_input = self.task_configuration.prepare_input(model_input)
        prepared_converted : Dict[str, np.ndarray] = {}
        for key, value in prepared_input.items():
            prepared_converted[key] = np.asarray(value)
        t2 = time.perf_counter()
        return InputContainer(model_input, prepared_converted), t2 - t1

    def prepare_inference_job(self, request_id : str, input_container : InputContainer[np.ndarray],
                              request_time : float, deadline : float, run_options : RunOptions) -> OnnxInferenceJob:
        return self._onnx_model.prepare_inference_job(request_id, self.task_configuration, input_container,
                                                      request_time, deadline, run_options)


    def run_inference(self, job : OnnxInferenceJob):
        """
        Schedules the job for execution via onnxruntime and blocks the thread until execution is finished.
        Once the execution is finished, the post-processing callback provided in the constructor for this instance
        is called.
        The outputs are stored in job.outputs.
        :param job: A job created via prepare_inference_job
        :return: The inference time in seconds
        """
        t1 = time.perf_counter()
        job.execute()
        self._post_processing_callback(job)
        t2 = time.perf_counter()
        return t2 - t1

    def execute(self, model_input : Dict[str, MultidimensionalArray], request_id : str = "") -> tuple[float, float]:
        """
        Performs inference for the given input. This method takes care of the input conversion to the configuration
        associated with this instance.
        The calling thread is blocked until inference is completed.
        :param request_id: The id associated with the request
        :param model_input: The input to perform inference on.
        :return: the input preparation time in seconds, the inference time in seconds
        """
        input_container, input_preparation_time = self.prepare_input(model_input)
        job = self.prepare_inference_job(request_id, input_container, -1, -1, RunOptions())
        inference_time = self.run_inference(job)
        return input_preparation_time, inference_time
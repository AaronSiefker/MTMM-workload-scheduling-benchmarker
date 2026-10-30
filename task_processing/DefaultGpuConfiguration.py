import time
from typing import Callable, List

import torch

from task_processing.GpuConfiguration import GpuConfiguration
from task_processing.InputContainer import InputContainer
from task_processing.TaskConfiguration import TaskConfiguration
from task_processing.gpu_execution.TensorRtInferenceJob import TensorRtInferenceJob
from task_processing.gpu_execution.TensorRtModel import TensorRtModel


class DefaultGpuConfiguration(GpuConfiguration):
    def __init__(self, task_configuration: TaskConfiguration, trt_model: TensorRtModel,
                 post_processing_callback: Callable[[TensorRtInferenceJob], None],
                 num_streams : int):
        super().__init__(task_configuration, post_processing_callback)
        self._trt_models : List[TensorRtModel] = [trt_model]
        for _ in range(num_streams - 1):
            self._trt_models.append(TensorRtModel.copy_with_new_context(trt_model))

    def prepare_inference_job(self, request_id : str, input_container : InputContainer[torch.Tensor], batch_size : int,
                              request_time : float, deadline : float, stream : torch.cuda.Stream,
                              stream_index : int) -> TensorRtInferenceJob:
        return self._trt_models[stream_index].prepare_inference_job(request_id, self.task_configuration, input_container,
                                                     batch_size, request_time, deadline, stream, stream_index)

    def run_inference(self, job : TensorRtInferenceJob) -> float:
        """
        Schedules the job for execution via tensorrt and blocks the thread until execution is finished.
        Once the execution is finished, the post-processing callback provided in the constructor for this instance
        is called.
        The outputs are stored in job.outputs.
        :param job: A job created via prepare_inference_job
        :return: The inference time in seconds
        """
        t1 = time.perf_counter()
        if not job.execute_async():
            raise RuntimeError("Unexpectedly failed to schedule job on gpu using tensorrt")
        job.event.synchronize()
        self._post_processing_callback(job)
        t2 = time.perf_counter()
        job.consumed_energy = job.associated_config.energy_statistics["default"][job.batch_size]
        return t2 - t1


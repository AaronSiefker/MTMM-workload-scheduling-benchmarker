import time
from typing import Callable, Final, List

import torch

from task_processing.GpuConfiguration import GpuConfiguration
from task_processing.InputContainer import InputContainer
from task_processing.TaskConfiguration import TaskConfiguration
from task_processing.gpu_execution.TensorRtInferenceJob import TensorRtInferenceJob
from task_processing.gpu_execution.TensorRtModel import TensorRtModel


# Required, because condition did not compile well with TensorRT. Depending on threshold either the early-exit
# was never taken or it was always taken. For the onnx-model everything worked as expected.
class EarlyExitGpuConfiguration(GpuConfiguration):
    """
    Assumes  that the intermediate prediction and confidence returned by the early model are scalars
    """
    def __init__(self, task_configuration: TaskConfiguration, early_trt_model: TensorRtModel,
                 intermediate_features_key : str, intermediate_prediction_key : str, intermediate_confidence_key : str,
                 input_threshold_key : str,
                 late_trt_model: TensorRtModel,
                 post_processing_callback: Callable[[TensorRtInferenceJob], None],
                 num_streams : int):
        super().__init__(task_configuration, post_processing_callback)
        self._early_trt_models : List[TensorRtModel] = [early_trt_model]
        for _ in range(num_streams - 1):
            self._early_trt_models.append(TensorRtModel.copy_with_new_context(early_trt_model))
        self._intermediate_features_key : Final[str] = intermediate_features_key
        self._intermediate_prediction_key : Final[str] = intermediate_prediction_key
        self._intermediate_confidence_key : Final[str] = intermediate_confidence_key
        self._input_threshold_key : Final[str] = input_threshold_key
        self._late_trt_models : List[TensorRtModel] = [late_trt_model]
        for _ in range(num_streams - 1):
            self._late_trt_models.append(TensorRtModel.copy_with_new_context(late_trt_model))

    def prepare_inference_job(self, request_id : str, input_container : InputContainer[torch.Tensor], batch_size : int,
                              request_time : float, deadline : float, stream : torch.cuda.Stream,
                              stream_index : int) -> TensorRtInferenceJob:
        return self._early_trt_models[stream_index].prepare_inference_job(request_id, self.task_configuration, input_container,
                                                           batch_size, request_time, deadline, stream, stream_index)

    def run_inference(self, job: TensorRtInferenceJob) -> float:
        # must transfer late_job.outputs into job.outputs, so that the caller can see them; caller never sees late_job
        t1 = time.perf_counter()
        if not job.execute_async():
            raise RuntimeError("Unexpectedly failed to schedule job on gpu using tensorrt")
        job.event.synchronize()
        job.consumed_energy = job.associated_config.energy_statistics["early_exit"][job.batch_size]
        late_input = job.outputs[self._intermediate_features_key]
        confidence = job.outputs[self._intermediate_confidence_key]
        if confidence <= torch.as_tensor(job.inputs.converted_input[self._input_threshold_key], device="cuda"):
            input_dict = {self._intermediate_features_key: late_input}
            late_job = self._late_trt_models[job.stream_index].prepare_inference_job(
                job.request_id, self.task_configuration, InputContainer(input_dict, input_dict), job.batch_size,
                job.request_time, job.deadline, job.stream, job.stream_index
            )
            if not late_job.execute_async():
                raise RuntimeError("Unexpectedly failed to schedule job on gpu using tensorrt")
            late_job.event.synchronize()
            job.consumed_energy = job.associated_config.energy_statistics["late_exit"][job.batch_size]
            job.outputs = late_job.outputs
        self._post_processing_callback(job)
        t2 = time.perf_counter()
        return t2 - t1
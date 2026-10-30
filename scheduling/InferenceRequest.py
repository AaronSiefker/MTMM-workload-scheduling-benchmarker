from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import StrEnum
from typing import Dict, Final

import torch
from onnxruntime import RunOptions
from torch.cuda import Stream

from task_processing.CpuConfiguration import CpuConfiguration
from task_processing.GpuConfiguration import GpuConfiguration
from task_processing.InputContainer import InputContainer
from task_processing.TaskConfiguration import TaskConfiguration
from task_processing.TaskEngine import TaskEngine
from task_processing.cpu_execution.OnnxInferenceJob import OnnxInferenceJobStatus
from task_processing.input_preparation.ConfigurationInputConverter import MultidimensionalArray

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from tasks.InferenceTask import InferenceTask

class DeviceName(StrEnum):
    NONE = "none"
    CUDA = "cuda"
    CPU = "cpu"

@dataclass(frozen=True)
class InferenceStats:
    request : InferenceRequest
    device : DeviceName
    task_config : TaskConfiguration

    finished_execution : bool
    finish_time : float
    preparation_time : float
    inference_time : float

    consumed_energy : float

class InferenceRequest:
    def __init__(self, task : InferenceTask, predecessor_request_id : str | None,
                 inference_input : Dict[str, MultidimensionalArray], request_time : float, deadline : float, batch_size : int,
                 false_positives : int, max_dependency_path_length : int, is_deadline_shared : bool):
        self.request_id : Final[str] = task.get_next_request_id()
        self.predecessor_request_id : str | None = predecessor_request_id
        self.task : Final[InferenceTask] = task
        self.task_engine : Final[TaskEngine] = self.task.get_task_engine()

        self._inference_input : Dict[str, MultidimensionalArray] = inference_input
        self.request_time : Final[float] = request_time
        # reassignable to increment with agent update time diff while training RL agent
        self.deadline : float = deadline

        self.max_dependency_path_length : Final[int] = max_dependency_path_length
        self.is_deadline_shared : Final[bool] = is_deadline_shared

        if batch_size <= 0:
            raise ValueError("Error, batch_size must be greater than 0, however the provided value for batch_size was",
                             batch_size)
        self.batch_size : Final[int] = batch_size
        if false_positives < 0:
            raise ValueError("Error, false_positives must not be negative, however the provided value for "
                             "false_positives was", false_positives)
        if false_positives > batch_size:
            raise ValueError("Error, false_positives must not be greater than batch_size."
                             f"false_positives was '{false_positives}' and batch_size was {batch_size}")
        self.false_positives : Final[int] = false_positives

        self.cancel : threading.Event = threading.Event()

        self._ongoing_request_gpu : InferenceRequest.OngoingRequestGpu | None = None
        # without this, execute may be called before that thread sees that ongoing_request is not None
        self._ongoing_lock : threading.Lock = threading.Lock()

    def prepare_input_gpu(self, gpu_config : GpuConfiguration):
        with self._ongoing_lock:
            if self._ongoing_request_gpu is not None:
                raise ValueError("Error, the input for this request was already prepared once")
            ongoing_request_gpu = InferenceRequest.OngoingRequestGpu(self._inference_input, self.batch_size, gpu_config)
            self._ongoing_request_gpu = ongoing_request_gpu
            # type checker was complaining about None possibility when calling on self._ongoing_request_gpu
        ongoing_request_gpu.prepare()

    def cancel_ongoing_request(self) -> InferenceStats:
        with self._ongoing_lock:
            if self._ongoing_request_gpu is None:
                raise ValueError("Error, request is not ongoing")
        return self._ongoing_request_gpu.cancel(self)

    def execute_gpu(self, stream : Stream, stream_index : int) -> InferenceStats:
        with self._ongoing_lock:
            if self._ongoing_request_gpu is None:
                raise ValueError("Error, request is not ongoing")
        return self._ongoing_request_gpu.execute(self, stream, stream_index)

    # noinspection DuplicatedCode
    # not actually duplicated since the types for stream and run_options are different and these different types are required
    def process_async_gpu(self, gpu_config : GpuConfiguration, stream : Stream, execute_event : threading.Event) \
            -> InferenceStats:
        input_container, preparation_time = gpu_config.prepare_input(self._inference_input)
        # TODO InferenceRequests should probably notify scheduler here that input preprocessing is done
        execute_event.wait()
        if self.cancel.is_set():
            return InferenceStats(self, DeviceName.NONE, gpu_config.task_configuration, False, -1, preparation_time, -1, 0.0)
        job = gpu_config.prepare_inference_job(self.request_id, input_container, self.batch_size
                                               , self.request_time, self.deadline, stream, 0)
        inference_time = gpu_config.run_inference(job)
        finish_time = time.perf_counter()
        return InferenceStats(self, DeviceName.CUDA, gpu_config.task_configuration, True,
                              finish_time, preparation_time, inference_time, job.consumed_energy)

    # noinspection DuplicatedCode
    # not actually duplicated since the types for stream and run_options are different and these different types are required
    def process_async_cpu(self, cpu_config : CpuConfiguration, run_options : RunOptions, execute_event : threading.Event) \
            -> InferenceStats:
        input_container, preparation_time = cpu_config.prepare_input(self._inference_input)
        # TODO InferenceRequests should probably notify scheduler here that input preprocessing is done
        execute_event.wait()
        if self.cancel.is_set():
            return InferenceStats(self, DeviceName.NONE, cpu_config.task_configuration, False, -1, preparation_time, -1, 0.0)
        job = cpu_config.prepare_inference_job(self.request_id, input_container, self.request_time, self.deadline, run_options)
        inference_time = cpu_config.run_inference(job)
        finish_time = time.perf_counter()
        return InferenceStats(self, DeviceName.CPU, cpu_config.task_configuration, job.status == OnnxInferenceJobStatus.FINISHED,
        finish_time, preparation_time, inference_time, 0.0) # never benchmarked cpu energy consumption since we didn't end up using it

    class OngoingRequestGpu:
        def __init__(self, inference_input : Dict[str, MultidimensionalArray], batch_size : int, gpu_config : GpuConfiguration):
            self._inference_input = inference_input
            self._batch_size = batch_size
            self._gpu_config : GpuConfiguration = gpu_config

            self._prepared_input_container : InputContainer[torch.Tensor] | None = None
            self._preparation_time : float = -1
            self._is_input_prepared : threading.Event = threading.Event()

        def prepare(self) -> None:
            self._prepared_input_container, self._preparation_time = self._gpu_config.prepare_input(self._inference_input)
            self._is_input_prepared.set()

        def cancel(self, request : InferenceRequest) -> InferenceStats:
            self._is_input_prepared.wait()
            return InferenceStats(request, DeviceName.NONE, self._gpu_config.task_configuration,
                                  False, -1, self._preparation_time, -1, 0.0)

        def execute(self, request : InferenceRequest, stream : Stream, stream_index : int) -> InferenceStats:
            self._is_input_prepared.wait()
            if self._prepared_input_container is None or self._preparation_time < 0:
                raise ValueError("Error, encountered a state where the input prepared event was set but the "
                                 "prepared input was not available")
            job = self._gpu_config.prepare_inference_job(request.request_id, self._prepared_input_container, self._batch_size,
                                                         request.request_time, request.deadline, stream, stream_index)
            inference_time = self._gpu_config.run_inference(job)
            finish_time = time.perf_counter()
            return InferenceStats(request, DeviceName.CUDA, self._gpu_config.task_configuration, True, finish_time,
                                  self._preparation_time, inference_time, job.consumed_energy)
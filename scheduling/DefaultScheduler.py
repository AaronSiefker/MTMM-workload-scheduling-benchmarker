import threading
from abc import abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Event
from typing import Final, List, Dict, Tuple, Any

import torch
from onnxruntime import RunOptions

from scheduling.InferenceRequest import InferenceRequest, InferenceStats
from scheduling.Scheduler import Scheduler
from scheduling.strategy.SchedulingStrategy import SchedulingStrategy
from task_processing.CpuConfiguration import CpuConfiguration
from task_processing.GpuConfiguration import GpuConfiguration
from tasks.InferenceTask import InferenceTask


class DefaultScheduler(Scheduler):
    def __init__(self):
        self._wake_up : Final[Event] = Event()

        self._pending_requests : List[InferenceRequest] = []
        self._new_requests : List[InferenceRequest] = []
        self._new_requests_lock : threading.Lock = threading.Lock()

        self._completed_requests : List[InferenceStats] = []
        self._completed_requests_lock : threading.Lock = threading.Lock()

        self._thread_pool = ThreadPoolExecutor(max_workers=6)

    # Use event which is immediately cleared after being set, so that nothing is missed
    # I can wait on event with timeout to potentially cancel cpu inference
    # add callback to executor.submit which adds future to queue and sets event
    # add function which adds inference request to queue and sets event
    # Scheduler Strategy should have function which determines timeout

    def add_request(self, request : InferenceRequest) -> None:
        with self._new_requests_lock:
            if request.predecessor_request_id == SchedulingStrategy.WARMUP_REQUEST_ID:
                return
            self._new_requests.append(request)
        self._wake_up.set()

    def add_requests(self, requests : List[InferenceRequest]) -> None:
        with self._new_requests_lock:
                self._new_requests.extend(requests)
        self._wake_up.set()

    def on_inference_request_completed(self, future : Future[InferenceStats]) -> None:
        with self._completed_requests_lock:
            if future.result().request.request_id == SchedulingStrategy.WARMUP_REQUEST_ID:
                return
            self._completed_requests.append(future.result())
        # cannot add stats to collections here, since scheduler I do not want to multi-thread that.
        # Also, scheduler has to notify strategy of completion.
        self._wake_up.set()

    def begin_preparing_input_gpu(self, request : InferenceRequest, gpu_config : GpuConfiguration):
        self._thread_pool.submit(request.prepare_input_gpu, gpu_config)

    def cancel_request_gpu(self, request : InferenceRequest):
        future = self._thread_pool.submit(request.cancel_ongoing_request)
        future.add_done_callback(self.on_inference_request_completed)

    def begin_processing_prepared_request_gpu(self, request : InferenceRequest, stream : torch.cuda.Stream,
                                              stream_index : int) -> None:
        future = self._thread_pool.submit(request.execute_gpu, stream, stream_index)
        future.add_done_callback(self.on_inference_request_completed)

    def begin_processing_request_gpu(self, request : InferenceRequest, gpu_config : GpuConfiguration,
                                     stream : torch.cuda.Stream, event : threading.Event):
        future = self._thread_pool.submit(request.process_async_gpu, gpu_config, stream, event)
        future.add_done_callback(self.on_inference_request_completed)

    def begin_processing_request_cpu(self, request : InferenceRequest, cpu_config : CpuConfiguration,
                                     run_options : RunOptions, event : threading.Event):
        future = self._thread_pool.submit(request.process_async_cpu, cpu_config, run_options, event)
        future.add_done_callback(self.on_inference_request_completed)

    @staticmethod
    def remove_missed_deadline_requests(requests : List[InferenceRequest], current_time : float) \
        -> List[InferenceRequest]:
        removed = [request for request in requests if request.deadline < current_time]
        requests[:] = [request for request in requests if request.deadline >= current_time]
        return removed

    @staticmethod
    def remove_missed_deadline_requests_with_tuple(requests : List[Tuple[InferenceRequest, Any]],
                                                              current_time : float) -> List[InferenceRequest]:
        removed = [request for request, _ in requests if request.deadline < current_time]
        requests[:] = [request for request in requests if request[0].deadline >= current_time]
        return removed

    @abstractmethod
    def warmup(self, tasks : List[InferenceTask], warmup_iterations : int, custom_batch_sizes : Dict[str, int] = None):
        pass


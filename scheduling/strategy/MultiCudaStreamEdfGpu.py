from __future__ import annotations

import time
from typing import List, Dict, Final, Tuple

import torch.cuda

from scheduling.InferenceRequest import InferenceStats, InferenceRequest
from scheduling.strategy import StrategyUtil
from scheduling.strategy.SchedulingStrategy import SchedulingStrategy
from task_processing.TaskEngine import TaskEngine
from tasks.InferenceTask import InferenceTask

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scheduling.BenchmarkScheduler import BenchmarkScheduler


class MultiCudaStreamEdfGpu(SchedulingStrategy):
    def __init__(self, benchmark_scheduler : BenchmarkScheduler, num_streams : int, is_green_context_based : bool = False,
                 is_miss_aware : bool = False):
        self._benchmark_scheduler : Final[BenchmarkScheduler] = benchmark_scheduler

        self._prepared_requests : List[InferenceRequest] = []
        self._ongoing_requests : List[InferenceRequest | None] = [None for _ in range(num_streams)]

        if is_green_context_based:
            # prevent garbage collector from cleaning up the contexts
            self._green_contexts = [torch.cuda.green_contexts.GreenContext.create(num_sms=4) for _ in range(num_streams)]
            # noinspection PyUnresolvedReferences,PyTypeChecker
            self._streams : Tuple[torch.cuda.Stream, ...] = tuple(context.Stream() for context in self._green_contexts)
        else:
            self._streams : Tuple[torch.cuda.Stream, ...] = tuple(torch.cuda.Stream() for _ in range(num_streams))

        self._canceled_request_ids : List[str] = []

        self._is_miss_aware : Final[bool] = is_miss_aware
        self._task_engines_by_task_name : Dict[str, TaskEngine] = {}


    def invoke(self, pending_requests: List[InferenceRequest]) -> None:
        now = time.perf_counter()
        to_remove = []
        for request in self._prepared_requests:
            if request is not None and now >= request.deadline:
                self._benchmark_scheduler.cancel_request_gpu(request)
                self._canceled_request_ids.append(request.request_id)
                self._benchmark_scheduler.handle_dropped_request(request, now)
                to_remove.append(request)
        self._prepared_requests[:] = [request for request in self._prepared_requests if request not in to_remove]

        # start previously prepared requests
        self._try_start_gpu_executions()

        # Miss_aware not really applicable here, since we do not actually know by how much we missed the deadline.
        self._benchmark_scheduler.remove_and_handle_missed_deadline_requests(pending_requests, now)
        self._start_preparing_requests(pending_requests)

        if self._try_start_gpu_executions():
            self._start_preparing_requests(pending_requests)

    def notify_of_completion(self, stats: InferenceStats) -> None:
        request_id = stats.request.request_id
        if request_id in self._canceled_request_ids:
            self._canceled_request_ids.remove(request_id)
            return

        if self._is_miss_aware:
            slack_time = stats.request.deadline - stats.finish_time
            if slack_time < 0:
                StrategyUtil.adjust_active_gpu_configurations_greedy(
                    self._task_engines_by_task_name.values(),
                    slack_time,
                    lambda current, other: current.mean_inference_time[1] - other.mean_inference_time[1] # batch_size key in dict, not index in list
                )

        for i in range(len(self._ongoing_requests)):
            ongoing_request = self._ongoing_requests[i]
            if ongoing_request is not None and ongoing_request.request_id == request_id:
                self._ongoing_requests[i] = None
                return

    def _start_preparing_requests(self, pending_requests : List[InferenceRequest]) -> None:
        to_prepare = len(self._streams) - len(self._prepared_requests)
        if to_prepare == 0:
            return

        for _ in range(to_prepare):
            index, next_request = StrategyUtil.select_lowest_deadline(pending_requests)
            if next_request is None:
                return
            del pending_requests[index]
            if self._is_miss_aware and (next_request.task.task not in self._task_engines_by_task_name.keys()):
                self._task_engines_by_task_name[next_request.task.task] = next_request.task_engine

            gpu_config = next_request.task_engine.active_gpu_configuration
            #gpu_config = next_request.task_engine.gpu_configurations[-1]
            self._benchmark_scheduler.begin_preparing_input_gpu(next_request, gpu_config)
            self._prepared_requests.append(next_request)

    def _try_start_gpu_executions(self) -> bool:
        out = False
        for i, stream in enumerate(self._streams):
            if len(self._prepared_requests) == 0:
                return out

            if self._ongoing_requests[i] is not None:
                continue

            next_request = self._choose_next_prepared_request()
            self._benchmark_scheduler.begin_processing_prepared_request_gpu(next_request, stream, i)
            self._ongoing_requests[i] = next_request
            out = True
        return out

    def _choose_next_prepared_request(self) -> InferenceRequest:
        request = self._prepared_requests[0]
        del self._prepared_requests[0]
        return request

    def handle_termination(self, current_time: float) -> None:
        for request in self._prepared_requests:
            if request is not None:
                self._benchmark_scheduler.handle_dropped_request(request, current_time)
                self._benchmark_scheduler.cancel_request_gpu(request)
        self._prepared_requests.clear()

    def warmup(self, tasks: List[InferenceTask], warmup_iterations, custom_batch_sizes: Dict[str, int]):
        if self._is_miss_aware:
            self._task_engines_by_task_name.update({task.task: task.get_task_engine() for task in tasks})
        for i, stream in enumerate(self._streams):
            SchedulingStrategy.warmup_gpu(tasks, warmup_iterations, custom_batch_sizes, stream, i)

    def get_num_cuda_streams(self) -> int:
        return len(self._streams)
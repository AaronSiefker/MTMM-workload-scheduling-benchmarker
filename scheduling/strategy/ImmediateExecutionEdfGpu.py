from __future__ import annotations

import threading
import time
from typing import Final, List, Dict

import torch.cuda

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scheduling.BenchmarkScheduler import BenchmarkScheduler

from scheduling.InferenceRequest import InferenceStats, InferenceRequest
from scheduling.strategy import StrategyUtil
from scheduling.strategy.SchedulingStrategy import SchedulingStrategy
from task_processing.TaskEngine import TaskEngine
from tasks.InferenceTask import InferenceTask


class ImmediateExecutionEdfGpu(SchedulingStrategy):
    def __init__(self, default_scheduler : BenchmarkScheduler, is_miss_aware : bool = False):
        self._default_scheduler : Final[BenchmarkScheduler] = default_scheduler

        self._is_miss_aware : Final[bool] = is_miss_aware
        self._task_engines_by_task_name : Dict[str, TaskEngine] = {}

        self._is_executing : bool = False
        self._stream : Final[torch.cuda.Stream] = torch.cuda.Stream()

    def invoke(self, pending_requests: List[InferenceRequest]) -> None:
        now = time.perf_counter()
        self._default_scheduler.remove_and_handle_missed_deadline_requests(pending_requests, now)

        if self._is_executing:
            return

        # noinspection DuplicatedCode
        index, next_request = StrategyUtil.select_lowest_deadline(pending_requests)
        if next_request is None:
            return
        del pending_requests[index]
        if self._is_miss_aware and (next_request.task.task not in self._task_engines_by_task_name.keys()):
            self._task_engines_by_task_name[next_request.task.task] = next_request.task_engine

        event = threading.Event()
        event.set()
        gpu_config = next_request.task_engine.active_gpu_configuration
        self._default_scheduler.begin_processing_request_gpu(next_request, gpu_config, self._stream, event)
        self._is_executing = True

    def notify_of_completion(self, stats: InferenceStats) -> None:
        if self._is_miss_aware:
            slack_time = stats.request.deadline - stats.finish_time
            if slack_time < 0:
                StrategyUtil.adjust_active_gpu_configurations_greedy(
                    self._task_engines_by_task_name.values(),
                    slack_time,
                    lambda current, other: (current.mean_inference_time[1] + current.mean_preparation_time[1]) -
                                           (other.mean_inference_time[1] + other.mean_preparation_time[1])
                )

        self._is_executing = False

    def handle_termination(self, current_time: float) -> None:
        pass

    def warmup(self, tasks: List[InferenceTask], warmup_iterations, custom_batch_sizes: Dict[str, int]):
        if self._is_miss_aware:
            self._task_engines_by_task_name.update({task.task: task.get_task_engine() for task in tasks})
        SchedulingStrategy.warmup_gpu(tasks, warmup_iterations, custom_batch_sizes, self._stream, 0)

    def get_num_cuda_streams(self) -> int:
        return 1
from __future__ import annotations

import threading
import time
from typing import List, Final, Dict

import torch.cuda

from typing import TYPE_CHECKING

from scheduling.strategy import StrategyUtil
from task_processing.TaskEngine import TaskEngine
from tasks.InferenceTask import InferenceTask

if TYPE_CHECKING:
    from scheduling.BenchmarkScheduler import BenchmarkScheduler

from scheduling.InferenceRequest import InferenceRequest, InferenceStats
from scheduling.strategy.SchedulingStrategy import SchedulingStrategy


class EarliestDeadlineFirstGpu(SchedulingStrategy):
    def __init__(self, benchmark_scheduler : BenchmarkScheduler, is_miss_aware : bool = False):
        self._benchmark_scheduler : Final[BenchmarkScheduler] = benchmark_scheduler
        # is true, if an inference request is currently supposed to be executed on GPU, NOT true solely for input preparation phase
        self._is_executing : bool = False
        # only select next_request while these are None!!!
        # next_request should be set to None, when the event is set, since then we can prepare the next input
        self._next_request : InferenceRequest | None = None
        self._next_event : threading.Event | None = None

        self._stream : Final[torch.cuda.Stream] = torch.cuda.Stream()

        self._canceled_request_ids : List[str] = []

        self._is_miss_aware : Final[bool] = is_miss_aware
        self._task_engines_by_task_name : Dict[str, TaskEngine] = {}


    def invoke(self, pending_requests: List[InferenceRequest]) -> None:
        now = time.perf_counter()
        if (self._next_request is not None) and now >= self._next_request.deadline:
            if self._next_event is None:
                raise ValueError("Error, reached a state where self._next_event was None, but self._next_request wasn't")
            # future also handles the InferenceStats created by a request canceled like this
            self._next_request.cancel.set()
            self._next_event.set()
            self._canceled_request_ids.append(self._next_request.request_id)
            self._benchmark_scheduler.handle_dropped_request(self._next_request, now)
            self._next_request = None
            self._next_event = None

        # start before overwriting self._next_request and self._next_event
        self._try_start_gpu_execution()

        # Miss_aware not really applicable here, since we do not actually know by how much we missed the deadline.
        # Additionally, for somewhat well-chosen deadlines, this should be the exception anyway
        self._benchmark_scheduler.remove_and_handle_missed_deadline_requests(pending_requests, now)
        self._try_select_next_request(pending_requests)

        # necessary to start execution if this is the first request or there were no pending requests temporarily
        if self._try_start_gpu_execution():
            # In that case, we already select and prepare the next input for inference
            self._try_select_next_request(pending_requests)


    def notify_of_completion(self, stats: InferenceStats) -> None:
        request_id = stats.request.request_id
        if request_id in self._canceled_request_ids:
            self._canceled_request_ids.remove(request_id)
        else:
            if self._is_miss_aware:
                slack_time = stats.request.deadline - stats.finish_time
                if slack_time < 0:
                    StrategyUtil.adjust_active_gpu_configurations_greedy(
                        self._task_engines_by_task_name.values(),
                        slack_time,
                        lambda current, other: current.mean_inference_time[1] - other.mean_inference_time[1] # batch_size key in dict, not index in list
                    )
            self._is_executing = False

    def _try_select_next_request(self, pending_requests : List[InferenceRequest]) -> None:
        if self._next_request is not None:
            return
        # noinspection DuplicatedCode
        index, next_request = StrategyUtil.select_lowest_deadline(pending_requests)
        if next_request is None:
            return
        del pending_requests[index]
        if self._is_miss_aware and (next_request.task.task not in self._task_engines_by_task_name.keys()):
            self._task_engines_by_task_name[next_request.task.task] = next_request.task_engine
        self._next_request = next_request
        event = threading.Event()
        self._next_event = event
        gpu_config = next_request.task_engine.active_gpu_configuration
        #gpu_config = next_request.task_engine.gpu_configurations[-1]
        # using local variables so that type checker doesn't complain about the possibility of None.
        # It cannot track that the attributes aren't null, even though we just assigned non-null values to them
        self._benchmark_scheduler.begin_processing_request_gpu(next_request, gpu_config, self._stream, event)

    def _try_start_gpu_execution(self) -> bool:
        if (self._next_request is not None) and (self._is_executing == False):
            if self._next_event is None:
                raise ValueError("Error, reached a state where self._next_event was None, but self._next_request wasn't")
            self._next_event.set()
            self._is_executing = True
            # can select next_request
            self._next_request = None
            self._next_event = None
            return True
        return False

    def handle_termination(self, current_time : float) -> None:
        if self._next_request is not None:
            # must be called here, since scheduler is stuck in _terminate until the thread finishes
            # and no longer processes completed requests
            self._benchmark_scheduler.handle_dropped_request(self._next_request, current_time)
            self._next_request.cancel.set()
            if self._next_event is not None:
                self._next_event.set()
            self._next_request = None
            self._next_event = None
        pass

    def warmup(self, tasks : List[InferenceTask], warmup_iterations : int, custom_batch_sizes : Dict[str, int]):
        if self._is_miss_aware:
            self._task_engines_by_task_name.update({task.task: task.get_task_engine() for task in tasks})
        SchedulingStrategy.warmup_gpu(tasks, warmup_iterations, custom_batch_sizes, self._stream, 0)

    def get_num_cuda_streams(self) -> int:
        return 1


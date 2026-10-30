from __future__ import annotations

import heapq
import itertools
import time
from typing import List, Dict, TYPE_CHECKING, Final, Tuple, Set

import torch

from scheduling.InferenceRequest import InferenceStats, InferenceRequest
from scheduling.strategy import StrategyUtil
from scheduling.strategy.SchedulingStrategy import SchedulingStrategy
from task_processing.GpuConfiguration import GpuConfiguration
from task_processing.TaskEngine import TaskEngine
from tasks.InferenceTask import InferenceTask

if TYPE_CHECKING:
    from scheduling.BenchmarkScheduler import BenchmarkScheduler


class ScalingEdfGpu(SchedulingStrategy):
    def __init__(self, benchmark_scheduler : BenchmarkScheduler, num_streams : int):
        # TODO use score from MyHeuristic while normalizing inference_time based on how often they are relatively executed
        self._benchmark_scheduler : Final[BenchmarkScheduler] = benchmark_scheduler

        self._prepared_requests : List[InferenceRequest] = []
        self._ongoing_requests : List[InferenceRequest | None] = [None for _ in range(num_streams)]

        self._streams : Tuple[torch.cuda.Stream, ...] = tuple(torch.cuda.Stream() for _ in range(num_streams))
        self._num_streams : Final[int] = num_streams

        self._canceled_request_ids : List[str] = []

        self._task_engines_by_task_name : Dict[str, TaskEngine] = {}
        self._num_executed_requests_by_tasks : Dict[str, int] = {}
        self._tasks_with_dependencies : Set[str] = set()

        self._num_deadlines_hit_without_miss : int = 0

    def invoke(self, pending_requests: List[InferenceRequest]) -> None:
        for request in pending_requests:
            if request.max_dependency_path_length > 0:
                self._tasks_with_dependencies.add(request.task.task)

        now = time.perf_counter()
        to_remove = []
        for request in self._prepared_requests:
            if request is not None and now >= request.deadline:
                self._benchmark_scheduler.cancel_request_gpu(request)
                self._canceled_request_ids.append(request.request_id)
                self._benchmark_scheduler.handle_dropped_request(request, now)
                to_remove.append(request)
        self._prepared_requests[:] = [request for request in self._prepared_requests if request not in to_remove]
        num_downscale_actions = len(to_remove)

        pre = len(pending_requests)
        self._benchmark_scheduler.remove_and_handle_missed_deadline_requests(pending_requests, now)
        num_removed = pre - len(pending_requests)
        num_downscale_actions += num_removed
        if num_downscale_actions > 0:
            self._num_deadlines_hit_without_miss = 0
            self._downscale_active_configs_n_level(num_downscale_actions)

        if self._num_deadlines_hit_without_miss >= 50:
            self._upscale_active_configs()
            self._num_deadlines_hit_without_miss = 0

        self._start_preparing_requests(pending_requests)

        if self._try_start_gpu_executions():
            self._start_preparing_requests(pending_requests)

    def _start_preparing_requests(self, pending_requests : List[InferenceRequest]) -> None:
        to_prepare = len(self._streams) - len(self._prepared_requests)
        if to_prepare == 0:
            return

        for _ in range(to_prepare):
            index, next_request = StrategyUtil.select_lowest_deadline(pending_requests)
            if next_request is None:
                return
            del pending_requests[index]
            if next_request.task.task not in self._task_engines_by_task_name.keys():
                self._task_engines_by_task_name[next_request.task.task] = next_request.task_engine

            gpu_config = next_request.task_engine.active_gpu_configuration
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

    def notify_of_completion(self, stats: InferenceStats) -> None:
        request_id = stats.request.request_id
        if request_id in self._canceled_request_ids:
            self._num_deadlines_hit_without_miss = 0
            self._canceled_request_ids.remove(request_id)
            return

        task_name = stats.request.task.task
        if  task_name not in self._num_executed_requests_by_tasks:
            self._num_executed_requests_by_tasks[task_name] = 0
        self._num_executed_requests_by_tasks[task_name] += 1

        slack_time = stats.request.deadline - stats.finish_time
        total_time = stats.request.deadline - stats.request.request_time
        buffer = total_time / 10
        # a small buffer
        if slack_time < 0:
            self._num_deadlines_hit_without_miss = 0
            self._downscale_active_configs(stats.request.task, slack_time)
        elif slack_time > buffer:
            self._num_deadlines_hit_without_miss += 1
        else:
            self._num_deadlines_hit_without_miss = 0

        for i in range(len(self._ongoing_requests)):
            ongoing_request = self._ongoing_requests[i]
            if ongoing_request is not None and ongoing_request.request_id == request_id:
                self._ongoing_requests[i] = None
                return

    def _upscale_active_configs(self):
        # concrete task shouldn't really matter here
        pq = self._create_priority_queue(self._num_executed_requests_by_tasks["Keyword Spotting"], False)

        if pq:
            task_engine, new_config, _ = heapq.heappop(pq)[-1]
            task_engine.active_gpu_configuration = new_config

    def _downscale_active_configs(self, task : InferenceTask, slack_time : float):
        time_diff = -slack_time
        task_requests_num = self._num_executed_requests_by_tasks[task.task]
        pq = self._create_priority_queue(task_requests_num, True)

        while pq and time_diff > 0:
            task_engine, new_config, normalized_time_gained = heapq.heappop(pq)[-1]
            time_diff -= normalized_time_gained
            task_engine.active_gpu_configuration = new_config

            pq = [
                t for t in pq
                if t[-1][0] != task_engine
            ]

    def _downscale_active_configs_n_level(self, n : int):
        # KWS might not be present at the very beginning
        num_requests = self._num_executed_requests_by_tasks["Keyword Spotting"] if "Keyword Spotting" in self._num_executed_requests_by_tasks else 1
        pq = self._create_priority_queue(num_requests, True)

        for _ in range(n):
            if pq:
                task_engine, new_config, _ = heapq.heappop(pq)[-1]
                task_engine.active_gpu_configuration = new_config

    def _create_priority_queue(self, base_num_requests : int, is_downscaling : bool):
        pq = []
        counter = itertools.count()
        for task_name, task_engine in self._task_engines_by_task_name.items():
            if task_name in self._num_executed_requests_by_tasks:
                other_task_num_requests = self._num_executed_requests_by_tasks[task_name]
            else:
                other_task_num_requests = base_num_requests
            inference_time_factor = other_task_num_requests / base_num_requests

            active_config : GpuConfiguration = task_engine.active_gpu_configuration
            for other_config in task_engine.gpu_configurations:
                time_diff = active_config.task_configuration.mean_inference_time[1] - \
                            other_config.task_configuration.mean_inference_time[1]
                normalized_time_diff = inference_time_factor * time_diff
                quality_diff = active_config.task_configuration.quality - other_config.task_configuration.quality
                if is_downscaling:
                    if normalized_time_diff > 0:
                        pq_key = self._create_downscaling_pq_key(task_engine.task, normalized_time_diff, quality_diff, next(counter))
                        heapq.heappush(pq, (pq_key, (task_engine, other_config, normalized_time_diff)))
                else:
                    if normalized_time_diff < 0:
                        pq_key = self._create_upscaling_pq_key(task_engine.task, normalized_time_diff, quality_diff, next(counter))
                        heapq.heappush(pq, (pq_key, (task_engine, other_config, normalized_time_diff)))
        return pq

    def _create_downscaling_pq_key(self, task_name : str, normalized_time_diff, quality_diff : float, tiebreaker : int) \
        -> Tuple[bool, float, int]:
        may_have_dependencies = task_name in self._tasks_with_dependencies
        #score = normalized_time_diff * (1- quality_diff)
        score = normalized_time_diff
        return may_have_dependencies, -score, tiebreaker

    def _create_upscaling_pq_key(self, task_name : str, normalized_time_diff, quality_diff : float, tiebreaker : int) \
        -> Tuple[bool, float, int]:
        may_have_dependencies, negated_score, tiebreaker = (
            self._create_downscaling_pq_key(task_name, normalized_time_diff, quality_diff, tiebreaker))
        # prioritize with dependencies and minimum additional time
        return not may_have_dependencies, -negated_score, tiebreaker


    def handle_termination(self, current_time: float) -> None:
        for request in self._prepared_requests:
            if request is not None:
                self._benchmark_scheduler.handle_dropped_request(request, current_time)
                self._benchmark_scheduler.cancel_request_gpu(request)
        self._prepared_requests.clear()

    def warmup(self, tasks: List[InferenceTask], warmup_iterations, custom_batch_sizes: Dict[str, int]):
        self._task_engines_by_task_name.update({task.task: task.get_task_engine() for task in tasks})
        for i, stream in enumerate(self._streams):
            SchedulingStrategy.warmup_gpu(tasks, warmup_iterations, custom_batch_sizes, stream, i)
        # for _, task_engine in self._task_engines_by_task_name.items():
        #     task_engine.active_gpu_configuration = task_engine.gpu_configurations[-1]

    def get_num_cuda_streams(self) -> int:
        return self._num_streams
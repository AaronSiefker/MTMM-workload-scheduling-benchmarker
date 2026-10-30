from __future__ import annotations

import itertools
import time
from typing import List, Dict, TYPE_CHECKING, Final, Tuple

import torch

from scheduling.InferenceRequest import InferenceStats, InferenceRequest
from scheduling.Scheduler import SchedulingResult
from scheduling.strategy.SchedulingStrategy import SchedulingStrategy
from task_processing.GpuConfiguration import GpuConfiguration
from task_processing.TaskEngine import TaskEngine
from tasks.InferenceTask import InferenceTask

import heapq

if TYPE_CHECKING:
    from scheduling.BenchmarkScheduler import BenchmarkScheduler


class MyHeuristic(SchedulingStrategy):
    def __init__(self, benchmark_scheduler : BenchmarkScheduler, num_streams : int):
        self._benchmark_scheduler : Final[BenchmarkScheduler] = benchmark_scheduler

        self._ongoing_requests : List[InferenceRequest | None] = [None for _ in range(num_streams)]
        self._expected_ongoing_finish_times : List[float] = [-1.0 for _ in range(num_streams)]
        self._streams: Tuple[torch.cuda.Stream, ...] = tuple(torch.cuda.Stream() for _ in range(num_streams))
        self._num_streams : int = num_streams

        self._canceled_request_ids : List[str] = []

        # Tuple[0]: num_finished_requests of task "key", Tuple[1] num dependencies triggered by request of task "key
        # potential ingredient for priorities, not used at the moment
        # self._dependency_probabilities : Dict[str, Tuple[int, int]] = {}

        self._is_queue_prepared : List[bool] = [False for _ in range(num_streams)]
        self._queues : List[List[Tuple[InferenceRequest, GpuConfiguration]]] = [[] for _ in range(num_streams)]

        self._task_engines : List[TaskEngine] = []
        # using task_enginge.active_config to store first index we consider, i.e. increasing the corresponding index to disable previous
        self._energy_violation_undo_stack : List[Tuple[TaskEngine, int]] = []
        self._num_finished_requests_since_energy_violation : int = 0

    def invoke(self, pending_requests: List[InferenceRequest]) -> None:
        # for request in pending_requests:
        #     if request.predecessor_request_id is not None:
        #         predecessor_task = InferenceTask.get_task_name_from_request_id(request.predecessor_request_id)
        #         num_requests, num_triggered_dependencies = self._dependency_probabilities[predecessor_task]
        #         self._dependency_probabilities[predecessor_task] = (num_requests, num_triggered_dependencies + 1)
        if self._num_finished_requests_since_energy_violation > 1000 and self._energy_violation_undo_stack:
            task_engine, new_active_index = self._energy_violation_undo_stack[-1]
            del self._energy_violation_undo_stack[-1]
            task_engine.active_gpu_configuration = task_engine.gpu_configurations[new_active_index]
            self._num_finished_requests_since_energy_violation = 0
            #print("scaled up:", task_engine.active_gpu_configuration.task_configuration.model_id)

        now = time.perf_counter()
        self._handle_prepared_requests(now)
        self._handle_missed_deadlines_in_queues(now)
        self._benchmark_scheduler.remove_and_handle_missed_deadline_requests(pending_requests, now)

        sorted_pending_requests = sorted(pending_requests, key=lambda r: (not r.is_deadline_shared, r.deadline))
        pending_requests.clear()

        projected_finish_times = self._get_projected_finish_times(now)
        # greedily append to queues (always append to current earliest finish time)
        for request in sorted_pending_requests:
            # else 0.0: highest priority for empty queues
            i, finish_time_queue = min(enumerate(projected_finish_times), key=lambda index_queue_pair: index_queue_pair[1][-1] if index_queue_pair[1] else 0.0)
            if self._queues[i]:
                previous_pair = self._queues[i][-1]
                previous_inference_time = MyHeuristic._get_expected_inference_time(previous_pair[0], previous_pair[1])
            else:
                previous_inference_time = 0.0
            self._queues[i].append((request, request.task_engine.active_gpu_configuration))
            current_time = finish_time_queue[-1] if finish_time_queue else now
            finish_time_queue.append(self._calculate_next_projected_finish_time(
                len(finish_time_queue), i, request, request.task_engine.active_gpu_configuration, current_time, previous_inference_time
            ))
        # sorting queues, such that shared deadlines are prioritized higher than the absolute deadline, since additional work must need
        # to be completed within the same timeframe for shared deadlines
        for i, queue in enumerate(self._queues):
            if self._is_queue_prepared[i]:
                queue[1:] = sorted(queue[1:], key=lambda r: (not r[0].is_deadline_shared, r[0].deadline))
            else:
                queue.sort(key=lambda r: (not r[0].is_deadline_shared, r[0].deadline))

        for i, queue in enumerate(self._queues):
            self._adjust_queue(i, queue, projected_finish_times[i])

        # Next up: prepare, execute and potentially prepare again
        # I do not need to throw away requests here, since that is already done earlier in this function (although not as strict as is potentially done in _handle_prepared_requests)
        for i in range(len(self._queues)):
            if self._queues[i]:
                request, config = self._queues[i][0]
                if not self._is_queue_prepared[i]:
                    self._benchmark_scheduler.begin_preparing_input_gpu(request, config)
                    self._is_queue_prepared[i] = True

                if self._ongoing_requests[i] is None:
                    self._benchmark_scheduler.begin_processing_prepared_request_gpu(request, self._streams[i], i)
                    self._ongoing_requests[i] = request
                    del self._queues[i][0]
                    self._is_queue_prepared[i] = False
                    # potentially preparing next request in queue
                    if self._queues[i]:
                        request, config = self._queues[i][0]
                        self._benchmark_scheduler.begin_preparing_input_gpu(request, config)
                        self._is_queue_prepared[i] = True

    def _handle_prepared_requests(self, now : float) -> None:
        for i, queue in enumerate(self._queues):
            if len(queue) == 0:
                continue
            prepared_request, config = queue[0]
            if not self._is_queue_prepared[i]:
                raise ValueError("Error, encountered a state where the first element in a queue was not prepared during the start of invoke."
                                 "This should never be the case")
            # Can't downscale last minute, because it is already prepared for a separate GPU config
            # do not use MyHeuristic._get_expected_inference_time here, would likely drop too many
            mean_finish_time = now + config.task_configuration.mean_inference_time[prepared_request.batch_size]
            if mean_finish_time >= prepared_request.deadline:
            # if now > prepared_request.deadline:
                self._benchmark_scheduler.cancel_request_gpu(prepared_request)
                self._canceled_request_ids.append(prepared_request.request_id)
                self._benchmark_scheduler.handle_dropped_request(prepared_request, now)

                del queue[0]
                self._is_queue_prepared[i] = False

            elif self._ongoing_requests[i] is None:
                self._benchmark_scheduler.begin_processing_prepared_request_gpu(prepared_request, self._streams[i], i)
                self._ongoing_requests[i] = prepared_request
                self._expected_ongoing_finish_times[i] = now + MyHeuristic._get_expected_inference_time(prepared_request, config)

                del queue[0]
                self._is_queue_prepared[i] = False

    def _handle_missed_deadlines_in_queues(self, now : float) -> None:
        for queue in self._queues:
            to_remove_both = []
            to_remove_requests_only = []
            for i, (request, config) in enumerate(queue):
                if i == 0 and self._is_queue_prepared[i]:
                    continue
                if now >= request.deadline:
                    # have to append both so that they are found in queue.remove(...)
                    to_remove_both.append((request, config))
                    to_remove_requests_only.append(request)
            self._benchmark_scheduler.handle_dropped_requests(to_remove_requests_only, now)
            for request in to_remove_both:
                queue.remove(request)

    def _get_projected_finish_times(self, now : float) -> List[List[float]]:
        projected_finish_times = [[] for _ in range(len(self._queues))]
        for i, queue in enumerate(self._queues):
            current_time = now if self._ongoing_requests[i] is None else self._expected_ongoing_finish_times[i]
            previous_inference_time = 0.0
            for j, (request, gpu_config) in enumerate(queue):
                current_time = self._calculate_next_projected_finish_time(
                    j, i, request, gpu_config, current_time, previous_inference_time
                )
                projected_finish_times[i].append(current_time)
                previous_inference_time = MyHeuristic._get_expected_inference_time(request, gpu_config)
        return projected_finish_times

    def _calculate_next_projected_finish_time(self, position : int, queue_num : int, request : InferenceRequest,
                                              config : GpuConfiguration,
                                              current_time : float, previous_inference_time : float) -> float:
        preparation_time = MyHeuristic._get_expected_preparation_time(request, config)
        inference_time = MyHeuristic._get_expected_inference_time(request, config)
        if position == 0 and not self._is_queue_prepared[queue_num]:
            return current_time + preparation_time + inference_time
        if position == 0:
            return current_time + inference_time

        preparation_overhead = max(0.0, preparation_time - previous_inference_time)
        return current_time + preparation_overhead + inference_time

    def _adjust_queue(self, queue_num : int, queue : List[Tuple[InferenceRequest, GpuConfiguration]],
                      projected_finish_times : List[float]) -> None:
        prioritized_adjustments = []
        adjusted_queue : List[Tuple[InferenceRequest, GpuConfiguration]] = []
        # tiebreaker
        counter = itertools.count()
        time_gained : float = 0.0
        for i, (request, config) in enumerate(queue):
            adjusted_queue.append((request, config))
            # Cannot adjust already prepared requests
            if i == 0 and self._is_queue_prepared[queue_num]:
                continue
            config_start_index = request.task_engine.get_index_of_active_gpu_configuration()
            for config_index, other_config in enumerate(request.task_engine.gpu_configurations):
                if config_index < config_start_index:
                    continue
                if MyHeuristic._get_expected_inference_time(request, other_config) < \
                    MyHeuristic._get_expected_inference_time(request, config):

                    key = MyHeuristic._compute_pq_key(request, config, other_config, next(counter))
                    heapq.heappush(prioritized_adjustments, (key, (i, other_config)))

            # time_gained not included in projected_finish_times
            request_slack = request.deadline - projected_finish_times[i] + time_gained
            if request_slack <= 0:
                current_diff = -request_slack
                while prioritized_adjustments and current_diff > 0:
                    (index, new_config) = heapq.heappop(prioritized_adjustments)[-1]
                    (req, current_config) = adjusted_queue[index]
                    time_diff = MyHeuristic._get_expected_inference_time(req, current_config) - \
                        MyHeuristic._get_expected_inference_time(req, new_config)
                    # We have potentially already switched to another config for that request. In that case,
                    # only accept it if this config is also faster than that one
                    if time_diff <= 0:
                        continue

                    # I am not adjusting prepared requests here, since I never insert configs for such a request into the priority queue
                    adjusted_queue[index] = (req, new_config)
                    current_diff -= time_diff
                    time_gained += time_diff

        queue[:] = adjusted_queue

    @staticmethod
    def _compute_pq_key(request : InferenceRequest, current_config : GpuConfiguration, other_config : GpuConfiguration, tiebreaker : int) \
        -> Tuple[bool, float, int]:
        time_diff = MyHeuristic._get_expected_inference_time(request, current_config) \
        - MyHeuristic._get_expected_inference_time(request, other_config)
        quality_diff = current_config.task_configuration.quality - other_config.task_configuration.quality
        # min-heapq key: False < True and negating score to prioritize greater scores in min-heapq, tiebreaker to prevent heapq comparing my uncomparable item
        return request.max_dependency_path_length != 0, -(time_diff * (1- quality_diff)), tiebreaker

    @staticmethod
    def _get_expected_preparation_time(request : InferenceRequest, config : GpuConfiguration) -> float:
        #return config.task_configuration.static_98th_percentile_inference_time[request.batch_size]
        return config.task_configuration.mean_preparation_time[request.batch_size] * 1.25

    @staticmethod
    def _get_expected_inference_time(request : InferenceRequest, config : GpuConfiguration) -> float:
        #return config.task_configuration.static_98th_percentile_inference_time[request.batch_size]
        return config.task_configuration.mean_inference_time[request.batch_size] * 1.25

    def notify_of_completion(self, stats: InferenceStats) -> None:
        self._num_finished_requests_since_energy_violation += 1

        request_id = stats.request.request_id
        if request_id in self._canceled_request_ids:
            self._canceled_request_ids.remove(request_id)
            return

        # num_requests, num_triggered_dependencies = self._dependency_probabilities[stats.request.task.task]
        # self._dependency_probabilities[stats.request.task.task] = (num_requests + 1, num_triggered_dependencies)

        for i in range(len(self._ongoing_requests)):
            ongoing_request = self._ongoing_requests[i]
            if ongoing_request is not None and ongoing_request.request_id == request_id:
                self._ongoing_requests[i] = None
                self._expected_ongoing_finish_times[i] = -1.0
                return

    def on_energy_violation(self, results : List[SchedulingResult]):
        self._num_finished_requests_since_energy_violation = 0
        sorted_results_by_energy : List[SchedulingResult] = sorted(results, key=lambda r: r.consumed_energy, reverse=True)

        for result in sorted_results_by_energy:
            task_engine = next((engine for engine in self._task_engines if
                                engine.active_gpu_configuration.task_configuration.model_id == result.model_id))
            # do not deactivate last config
            active_index = task_engine.get_index_of_active_gpu_configuration()
            if active_index == len(task_engine.gpu_configurations) - 1:
                continue
            task_engine.active_gpu_configuration = task_engine.gpu_configurations[active_index + 1]
            self._energy_violation_undo_stack.append((task_engine, active_index))
            #print("scaled down:", task_engine.active_gpu_configuration.task_configuration.model_id)
            return

    def handle_termination(self, current_time: float) -> None:
        for i in range(len(self._queues)):
            if self._is_queue_prepared[i]:
                request, _ = self._queues[i][0]
                self._benchmark_scheduler.handle_dropped_request(request, current_time)
                # self._benchmark_scheduler.cancel_request_gpu(request)
                del self._queues[i][0]

    def warmup(self, tasks: List[InferenceTask], warmup_iterations, custom_batch_sizes: Dict[str, int]):
        # for task in tasks:
        #     self._dependency_probabilities[task.task] = (0, 0)
        for task in tasks:
            self._task_engines.append(task.get_task_engine())
        for i, stream in enumerate(self._streams):
            SchedulingStrategy.warmup_gpu(tasks, warmup_iterations, custom_batch_sizes, stream, i)

    def get_num_cuda_streams(self) -> int:
        return len(self._streams)
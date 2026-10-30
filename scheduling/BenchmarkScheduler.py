import json
import os
import time
from dataclasses import is_dataclass, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Final, Dict, Any, List, Tuple, Sequence

import numpy as np

from scheduling.DefaultScheduler import DefaultScheduler
from scheduling.InferenceRequest import InferenceRequest, DeviceName
from scheduling.Scheduler import SchedulingResult
from scheduling.strategy.EarliestDeadlineFirstGpu import EarliestDeadlineFirstGpu
from scheduling.strategy.Emerald.EmeraldStrategy import EmeraldStrategy
from scheduling.strategy.ImmediateExecutionEdfGpu import ImmediateExecutionEdfGpu
from scheduling.strategy.MultiCudaStreamEdfGpu import MultiCudaStreamEdfGpu
from scheduling.strategy.MyHeuristic import MyHeuristic
from scheduling.strategy.RLAgentTabular import RLAgentTabular
from scheduling.strategy.ScalingEdfGpu import ScalingEdfGpu
from scheduling.strategy.SchedulingStrategy import SchedulingStrategy
from scheduling.strategy.RLAgent import RLAgent
from tasks.InferenceTask import InferenceTask

class BenchmarkScheduler(DefaultScheduler):
    class Strategy(Enum):
        EARLIEST_DEADLINE_FIRST_GPU = auto()
        MISS_AWARE_EDF_GPU = auto()

        IMMEDIATE_EXECUTION_EDF_GPU = auto()
        MISS_AWARE_IMMEDIATE_EXECUTION_EDF_GPU = auto()

        EDF_2_CUDA_STREAMS_GPU = auto()
        MISS_AWARE_EDF_2_CUDA_STREAMS_GPU = auto()

        EDF_2_CUDA_GREEN_CONTEXTS_GPU = auto()
        MISS_AWARE_EDF_2_CUDA_GREEN_CONTEXTS_GPU = auto()

        RL_AGENT = auto()

        RL_AGENT_TABULAR_TRAINING = auto()
        RL_AGENT_TABULAR = auto()

        MY_HEURISTIC_1_STREAM = auto()
        MY_HEURISTIC_2_STREAMS = auto()

        SCALING_EDF_1_STREAM = auto()
        SCALING_EDF_2_STREAMS = auto()

        EMERALD_NORMAL_PERIODS_1_STREAM = auto()
        EMERALD_NORMAL_PERIODS_2_STREAMS = auto()
        EMERALD_HARD_PERIODS_2_STREAMS = auto()

        EMERALD_MEAN_NORMAL_PERIODS_1_STREAM = auto()
        EMERALD_MEAN_NORMAL_PERIODS_2_STREAMS = auto()
        EMERALD_MEAN_HARD_PERIODS_2_STREAMS = auto()

    def __init__(self, strategy: SchedulingStrategy | Strategy, result_file_path: str | os.PathLike,
                 energy_limit : float, energy_interval_length : float, strategy_kwargs : Dict = None):
        super().__init__()
        if strategy_kwargs:
            print("strategy_kwargs:", strategy_kwargs)
        # first key: task name
        # second key: InferenceStats.request.request_id
        self._scheduling_results : Dict[str, Dict[str, SchedulingResult]] = {}

        self._result_file_path : Final[Path] = Path(result_file_path)

        self._should_terminate : bool = False

        if isinstance(strategy, BenchmarkScheduler.Strategy):
            match strategy:
                case BenchmarkScheduler.Strategy.EARLIEST_DEADLINE_FIRST_GPU:
                    self._strategy : SchedulingStrategy = EarliestDeadlineFirstGpu(self)
                case BenchmarkScheduler.Strategy.MISS_AWARE_EDF_GPU:
                    self._strategy : SchedulingStrategy = EarliestDeadlineFirstGpu(self, is_miss_aware=True)

                case BenchmarkScheduler.Strategy.IMMEDIATE_EXECUTION_EDF_GPU:
                    self._strategy : SchedulingStrategy = ImmediateExecutionEdfGpu(self)
                case BenchmarkScheduler.Strategy.MISS_AWARE_IMMEDIATE_EXECUTION_EDF_GPU:
                    self._strategy : SchedulingStrategy = ImmediateExecutionEdfGpu(self, is_miss_aware=True)

                case BenchmarkScheduler.Strategy.EDF_2_CUDA_STREAMS_GPU:
                    self._strategy : SchedulingStrategy = MultiCudaStreamEdfGpu(self, 2)
                case BenchmarkScheduler.Strategy.MISS_AWARE_EDF_2_CUDA_STREAMS_GPU:
                    self._strategy : SchedulingStrategy = MultiCudaStreamEdfGpu(self, 2, is_miss_aware=True)

                case BenchmarkScheduler.Strategy.EDF_2_CUDA_GREEN_CONTEXTS_GPU:
                    self._strategy : SchedulingStrategy = MultiCudaStreamEdfGpu(self, 2, is_green_context_based=True)
                case BenchmarkScheduler.Strategy.MISS_AWARE_EDF_2_CUDA_GREEN_CONTEXTS_GPU:
                    self._strategy : SchedulingStrategy = MultiCudaStreamEdfGpu(self, 2, is_green_context_based=True, is_miss_aware=True)

                case BenchmarkScheduler.Strategy.RL_AGENT:
                    if strategy_kwargs is None:
                        raise ValueError(f"Error, {BenchmarkScheduler.Strategy.RL_AGENT} requires a dict with the corresponding arguments")
                    self._strategy : SchedulingStrategy = RLAgent(self, **strategy_kwargs)
                case BenchmarkScheduler.Strategy.RL_AGENT_TABULAR_TRAINING:
                    if strategy_kwargs is None:
                        raise ValueError(f"Error, {BenchmarkScheduler.Strategy.RL_AGENT_TABULAR_TRAINING} requires a dict with the corresponding arguments")
                    if not strategy_kwargs["is_learning"]:
                        raise ValueError(f"Error, is_learning must be true for {BenchmarkScheduler.Strategy.RL_AGENT_TABULAR_TRAINING}")
                    self._strategy : SchedulingStrategy = RLAgentTabular(self, **strategy_kwargs)
                case BenchmarkScheduler.Strategy.RL_AGENT_TABULAR:
                    if strategy_kwargs is None:
                        raise ValueError(
                            f"Error, {BenchmarkScheduler.Strategy.RL_AGENT_TABULAR} requires a dict with the corresponding arguments")
                    if strategy_kwargs["is_learning"]:
                        raise ValueError(f"Error, is_learning must be false for {BenchmarkScheduler.Strategy.RL_AGENT_TABULAR}")
                    self._strategy : SchedulingStrategy = RLAgentTabular(self, **strategy_kwargs)

                case BenchmarkScheduler.Strategy.MY_HEURISTIC_1_STREAM:
                    self._strategy : SchedulingStrategy = MyHeuristic(self, 1)
                case BenchmarkScheduler.Strategy.MY_HEURISTIC_2_STREAMS:
                    self._strategy : SchedulingStrategy = MyHeuristic(self, 2)

                case BenchmarkScheduler.Strategy.SCALING_EDF_1_STREAM:
                    self._strategy : SchedulingStrategy = ScalingEdfGpu(self, 1)
                case BenchmarkScheduler.Strategy.SCALING_EDF_2_STREAMS:
                    self._strategy: SchedulingStrategy = ScalingEdfGpu(self, 2)

                case BenchmarkScheduler.Strategy.EMERALD_NORMAL_PERIODS_1_STREAM:
                    self._strategy : SchedulingStrategy = EmeraldStrategy(self, 1, True, True, False)
                case BenchmarkScheduler.Strategy.EMERALD_NORMAL_PERIODS_2_STREAMS:
                    self._strategy : SchedulingStrategy = EmeraldStrategy(self, 2, True, True, False)
                case BenchmarkScheduler.Strategy.EMERALD_HARD_PERIODS_2_STREAMS:
                    self._strategy : SchedulingStrategy = EmeraldStrategy(self, 2, False, True, False)

                case BenchmarkScheduler.Strategy.EMERALD_MEAN_NORMAL_PERIODS_1_STREAM:
                    self._strategy : SchedulingStrategy = EmeraldStrategy(self, 1, True, False, True)
                case BenchmarkScheduler.Strategy.EMERALD_MEAN_NORMAL_PERIODS_2_STREAMS:
                    self._strategy : SchedulingStrategy = EmeraldStrategy(self, 2, True, False, True)
                case BenchmarkScheduler.Strategy.EMERALD_MEAN_HARD_PERIODS_2_STREAMS:
                    self._strategy : SchedulingStrategy = EmeraldStrategy(self, 2, False, False, True)
        elif isinstance(strategy, SchedulingStrategy):
            self._strategy : SchedulingStrategy = strategy
        else:
            raise ValueError("unsupported type for strategy")

        self.energy_limit : Final[float] = energy_limit
        self.energy_interval_length : Final[float] = energy_interval_length

        # at runtime I have to treat energy different than during final evaluation => I cannot know how much energy was consumed without finish time
        self._active_energy_results : List[SchedulingResult] = []

    def clear_state(self):
        with self._new_requests_lock:
            self._new_requests.clear()
        with self._completed_requests_lock:
            self._completed_requests.clear()
        self._pending_requests.clear()
        self._wake_up.clear()

        self._scheduling_results.clear()

        self._active_energy_results.clear()

    def terminate(self) -> None:
        self._should_terminate = True
        self._wake_up.set()


    def activate(self) -> None:
        #diffs = []
        while True:
            self._wake_up.wait()
            # I do not need a lock to clear/set since this is always set after the other changes have been made
            # => If we lose a "set" here between wait() and clear(), we still see the corresponding changes in this iteration
            self._wake_up.clear()

            # must be handled before termination so that they are not forgotten
            self._handle_completed_requests()
            if self._should_terminate:
                self._terminate()
                print("shutting down scheduler thread pool")
                self._thread_pool.shutdown()
                print("terminated scheduler")
                #print("diffs:")
                #print(mean(diffs))
                return

            with self._new_requests_lock:
                self._pending_requests.extend(self._new_requests)
                #now = time.perf_counter()
                #for request in self._new_requests:
                    #if request.task.task == "Keyword Spotting" or request.task.task == "Face Detection" or request.task.task == "Hand Detection":
                    #    diffs.append(now - request.request_time)
                self._new_requests.clear()

            # needs to be done before invoking strategy, so that strategy knows which processors are free
            # TODO strategy should track which tasks trigger which tasks how often to make decisions which include false_positive_probability
            self._strategy.invoke(self._pending_requests)
            #gc.collect()

    def _handle_completed_requests(self):
        with self._completed_requests_lock:
            for current in self._completed_requests:
                task_name = current.task_config.task
                request_id = current.request.request_id

                if task_name not in self._scheduling_results:
                    self._scheduling_results[task_name]: Dict[str, SchedulingResult] = {}
                new_result =  SchedulingResult.create(current)
                self._scheduling_results[task_name][request_id] = new_result

                if current.finished_execution:
                    current.task_config.update(current)
                    self._handle_energy(new_result)

                self._strategy.notify_of_completion(current)
            self._completed_requests.clear()

    def _handle_energy(self, new_result : SchedulingResult):
        current_interval_finish = new_result.finish_time
        current_interval_start = current_interval_finish - self.energy_interval_length

        self._active_energy_results.append(new_result)
        self._active_energy_results = [result for result in self._active_energy_results
                                       if (result.finish_time > current_interval_start) and result.get_start_time() < current_interval_finish]
        self._active_energy_results.sort(key= lambda result: result.finish_time)

        total_interval_energy = 0.0
        for active_result in self._active_energy_results:
            overlap_start = max(current_interval_start, active_result.get_start_time())
            overlap_end = min(current_interval_finish, active_result.finish_time)

            if overlap_start >= overlap_end:
                raise ValueError("Error, active_result doesn't overlap with currently considered energy interval")

            overlap = overlap_end - overlap_start
            if overlap > active_result.inference_time:
                raise ValueError("Error, overlap was somehow larger than active_result.inference_time")

            energy_ratio = overlap / active_result.inference_time

            energy_consumed_in_interval = active_result.consumed_energy * energy_ratio
            total_interval_energy += energy_consumed_in_interval

        if total_interval_energy > self.energy_limit:
            # only counting finished results once and all have already finished at runtime unlike during the actual evaluation
            self._strategy.on_energy_violation(self._active_energy_results)
            self._active_energy_results.clear()


    def handle_dropped_request(self, request : InferenceRequest, termination_time : float):
        self._add_dropped_request_to_results(request, termination_time)
        for dependency in request.task.dependencies:
            dropped_successors = dependency.accumulate_dropped_requests(request.request_id)
            for dropped_successor in dropped_successors:
                self._add_dropped_request_to_results(dropped_successor, termination_time)

    def handle_dropped_requests(self, requests : Sequence[InferenceRequest], termination_time : float):
        for request in requests:
            self.handle_dropped_request(request, termination_time)

    def _add_dropped_request_to_results(self, request : InferenceRequest, termination_time : float):
        task_name = request.task.task
        if task_name not in self._scheduling_results:
            self._scheduling_results[task_name]: Dict[str, SchedulingResult] = {}
        self._scheduling_results[task_name][request.request_id] = SchedulingResult(
            request.request_id,
            "never scheduled", "never scheduled", 0.0, DeviceName.NONE.value,
            False, 0.0, 0.0,
            request.request_time, termination_time, request.deadline, 0.0,
            request.batch_size, request.false_positives, request.predecessor_request_id
        )

    def remove_and_handle_missed_deadline_requests(self, requests : List[InferenceRequest], current_time : float) -> None:
        # Miss_aware not really applicable here, since we do not actually now by how much we missed the deadline.
        # Additionally, for somewhat well-chosen deadlines, this should be the exception anyway
        removed = DefaultScheduler.remove_missed_deadline_requests(requests, current_time)
        self.handle_dropped_requests(removed, current_time)

    def remove_and_handle_missed_deadline_requests_with_tuple(self, requests : List[Tuple[InferenceRequest, Any]],
                                                              current_time : float) -> None:
        removed = DefaultScheduler.remove_missed_deadline_requests_with_tuple(requests, current_time)
        self.handle_dropped_requests(removed, current_time)


    def _terminate(self):
        termination_time = time.perf_counter()
        self._strategy.handle_termination(termination_time)
        with self._new_requests_lock:
            self._pending_requests.extend(self._new_requests)
        for request in self._pending_requests:
            if request.deadline <= termination_time:
                self.handle_dropped_request(request, termination_time)
        self._result_file_path.parent.mkdir(parents=True, exist_ok=True)

        with self._result_file_path.open("w", encoding="utf-8") as f:
            def func(o : Any) -> Any:
                if isinstance(o, (np.int64, np.int32, np.int16, np.int8)):
                    return int(o)
                if isinstance(o, (np.float64, np.float32, np.float16)):
                    return float(o)
                if is_dataclass(o):
                    return asdict(o)
                raise TypeError(f"Could not convert object of type '{type(o)} when serializing scheduling results")

            json.dump(self._scheduling_results, f, default=func)


    def warmup(self, tasks : List[InferenceTask], warmup_iterations : int, custom_batch_sizes : Dict[str, int] = None):
        if custom_batch_sizes is None:
            custom_batch_sizes = {}
        self._strategy.warmup(tasks, warmup_iterations, custom_batch_sizes)

    def get_num_cuda_streams(self) -> int:
        return self._strategy.get_num_cuda_streams()
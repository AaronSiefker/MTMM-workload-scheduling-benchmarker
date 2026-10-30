from __future__ import annotations

import gc
import shutil
import threading
import time
from pathlib import Path
from typing import List, Dict, Final, TYPE_CHECKING, Tuple

import torch

from scheduling.InferenceRequest import InferenceStats, InferenceRequest
from scheduling.Scheduler import SchedulingResult
from scheduling.strategy.SchedulingStrategy import SchedulingStrategy
from scheduling.strategy.tabular.QLearning import QLearningWrapper
from tasks.InferenceTask import InferenceTask

if TYPE_CHECKING:
    from scheduling.BenchmarkScheduler import BenchmarkScheduler


class RLAgentTabular(SchedulingStrategy):
    def __init__(self, benchmark_scheduler : BenchmarkScheduler, agent : QLearningWrapper,
                 is_learning : bool, episode_length : float | None, cuda_streams : List[torch.cuda.Stream],
                 reset_workload_event : threading.Event | None,
                 intermediate_model_path : Path = None):
        self._benchmark_scheduler : Final[BenchmarkScheduler] = benchmark_scheduler

        self._agent : Final[QLearningWrapper] = agent
        self._is_learning = is_learning
        if is_learning and (episode_length is None):
            raise ValueError("Error, episode_length must not be None when is_learning is True")
        self._episode_length : float | None = episode_length
        self._episode_end : float | None = None
        self._reset_workload_event : threading.Event | None = reset_workload_event

        self._streams = cuda_streams
        self._num_streams = len(cuda_streams)

        self._prepared_requests : List[InferenceRequest] = []
        self._ongoing_requests : List[InferenceRequest | None] = [None for _ in range(len(cuda_streams))]

        self._previous_state : int | None = None

        self._canceled_request_ids : List[str] = []
        self._previous_episode_leftover_request_ids : List[str] = []

        self._intermediate_model_path : Path | None = intermediate_model_path
        self._best_models : List[Tuple[int, Path] | None] = [None, None, None]


    def invoke(self, pending_requests: List[InferenceRequest]) -> None:
        now = time.perf_counter()
        if self._is_learning and self._episode_length is not None and self._episode_end is None:
            self._episode_end = now + self._episode_length
            #print("now:", now)
            #print("episode end:", self._episode_end)
        if self._is_learning and now > self._episode_end:
            if self._reset_workload_event is None:
                raise ValueError("Error, attempting to end the episode while self._reset_workload_event is None")
            self._reset_workload_event.set()
            self._on_episode_end(now)
            return

        to_remove = []
        for request in self._prepared_requests:
            if now >= request.deadline:
                self._benchmark_scheduler.cancel_request_gpu(request)
                self._canceled_request_ids.append(request.request_id)
                self._benchmark_scheduler.handle_dropped_request(request, now)
                to_remove.append(request)
        self._prepared_requests[:] = [request for request in self._prepared_requests if request not in to_remove]
        self._benchmark_scheduler.remove_and_handle_missed_deadline_requests(pending_requests, now)

        any_stream_free = any(ongoing_request is None for ongoing_request in self._ongoing_requests)
        new_state = self._agent.get_state_id(pending_requests, self._prepared_requests, any_stream_free)
        if self._previous_state is not None:
            self._agent.on_external_state_change(self._previous_state, new_state)

        num_prep_actions = min(len(self._streams) - len(self._prepared_requests), len(pending_requests))
        for _ in range(num_prep_actions):
            self._agent.take_preparation_action(self._benchmark_scheduler, pending_requests,
                                                self._prepared_requests, any_stream_free)
        free_streams = 0
        for request in self._ongoing_requests:
            if request is None:
                free_streams += 1
        num_execution_actions = min(free_streams, len(self._prepared_requests))
        for _ in range(num_execution_actions):
            self._agent.take_execution_action(self._benchmark_scheduler, pending_requests, self._prepared_requests,
                                              self._ongoing_requests, self._streams)

        self._previous_state = self._agent.get_state_id(
            pending_requests, self._prepared_requests, any(ongoing_request is None for ongoing_request in self._ongoing_requests)
        )
        # fill up again after starting executions
        # TODO potentially train an agent without this, since a model-based agent performed better without this
        num_prep_actions = min(len(self._streams) - len(self._prepared_requests), len(pending_requests))
        for _ in range(num_prep_actions):
            self._agent.take_preparation_action(self._benchmark_scheduler, pending_requests,
                                                self._prepared_requests, any_stream_free)

    def _on_episode_end(self, now : float):
        self._episode_end = None
        self._previous_episode_leftover_request_ids.clear()

        for request in self._prepared_requests:
            self._benchmark_scheduler.handle_dropped_request(request, now)
            self._benchmark_scheduler.cancel_request_gpu(request)
            self._previous_episode_leftover_request_ids.append(request.request_id)

        self._prepared_requests.clear()

        # would prevent stream index being set to None again, update doesn't happen anyway, because the transition is removed in the agent when the new episode starts,
        # but update wouldn't be a problem anyway
        # for request in self._ongoing_requests:
        #     if request is not None:
        #         self._previous_episode_leftover_request_ids.append(request.request_id)

        self._benchmark_scheduler.clear_state()

        self._update_best_models()
        self._agent.on_begin_next_episode()

        gc.collect()

    def _update_best_models(self):
        if not self._is_learning or self._intermediate_model_path is None:
            return

        new_total_episode_reward = int(self._agent.total_episode_reward)
        model_index = -1
        for i in range(len(self._best_models) - 1, -1, -1):
            current_model = self._best_models[i]
            if current_model is None:
                model_index = i
                break
            model_reward, _ = current_model
            if new_total_episode_reward > model_reward:
                model_index = i
                break

        if model_index < 0:
            return
        new_path = self._create_intermediate_model_path(new_total_episode_reward)
        self._agent.save(new_path)
        current_worst = self._best_models[0]
        if current_worst is not None:
            _, path_to_delete = current_worst
            shutil.rmtree(path_to_delete)

        del self._best_models[0]
        self._best_models.insert(model_index, (new_total_episode_reward, new_path))


    def _create_intermediate_model_path(self, reward : int) -> Path:
        if self._intermediate_model_path is None:
            raise ValueError("Error, attempting to create concrete intermediate model path while self._intermediate_model_path is None")
        return Path(str(self._intermediate_model_path) + "_ep_reward=" + str(reward))


    def get_num_cuda_streams(self) -> int:
        return self._num_streams

    def notify_of_completion(self, stats: InferenceStats) -> None:
        if stats.request.request_id in self._previous_episode_leftover_request_ids:
            self._previous_episode_leftover_request_ids.remove(stats.request.request_id)
            return

        self._agent.on_completed_request(stats)

        request_id = stats.request.request_id
        if request_id in self._canceled_request_ids:
            self._canceled_request_ids.remove(request_id)
            return

        for i in range(len(self._ongoing_requests)):
            ongoing_request = self._ongoing_requests[i]
            if ongoing_request is not None and ongoing_request.request_id == request_id:
                self._ongoing_requests[i] = None
                return

    def on_energy_violation(self, results : List[SchedulingResult]):
        self._agent.on_energy_violation(results)

    def handle_termination(self, current_time: float) -> None:
        for request in self._prepared_requests:
            self._benchmark_scheduler.handle_dropped_request(request, current_time)
            self._benchmark_scheduler.cancel_request_gpu(request)
        self._prepared_requests.clear()

    def warmup(self, tasks: List[InferenceTask], warmup_iterations, custom_batch_sizes: Dict[str, int]):
        for i, stream in enumerate(self._streams):
            SchedulingStrategy.warmup_gpu(tasks, warmup_iterations, custom_batch_sizes, stream, i)
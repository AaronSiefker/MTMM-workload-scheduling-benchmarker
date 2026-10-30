from __future__ import annotations

from enum import Enum, auto
# noinspection PyProtectedMember
from multiprocessing.connection import Connection
import threading
import time
from datetime import datetime
from typing import Any, SupportsFloat, Tuple, List, Dict

import gymnasium as gym
import numpy as np
import torch.cuda
from gymnasium.core import ObsType, ActType

import setup_helper
from scheduling.AgentWrapper import AgentWrapper
from scheduling.DefaultScheduler import DefaultScheduler
from scheduling.InferenceRequest import InferenceRequest
from scheduling.Scheduler import SchedulingResult
from scheduling.strategy.SchedulingStrategy import SchedulingStrategy
from tasks.InferenceTask import InferenceTask
from xrbench_helper import calculate_xrbench_per_inference_score_no_energy


# noinspection DuplicatedCode
class AgentTrainingEnvScheduler(DefaultScheduler, gym.Env):
    def __init__(self, agent_wrapper : AgentWrapper, cuda_streams : List[torch.cuda.Stream],
                 episode_length : float,
                 reset_workload_event : threading.Event, interrupt_workload_event : threading.Event,
                 continue_workload_event : threading.Event):
        super().__init__()
        self._agent_wrapper = agent_wrapper
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(agent_wrapper.observation_length,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Discrete(agent_wrapper.num_actions)

        self._cuda_streams : List[torch.cuda.Stream] = cuda_streams
        self._prepared_requests : List[Tuple[InferenceRequest, float]] = []
        self._ongoing_requests : List[Tuple[InferenceRequest, float] | None] = [None for _ in range(len(cuda_streams))]

        self._current_observation : np.ndarray = np.zeros(agent_wrapper.observation_length)

        # self._remaining_prep_actions : int = 0
        # self._remaining_execution_actions : int = 0
        self._next_stream_idx : int = -1
        # prepare additional actions parallel to current executions
        self._remaining_prep_actions_post : int = 0

        self._is_prep_action : bool = False
        self._is_execution_action : bool = False

        self._total_episode_reward : float = 0.0

        self._episode_length : float = episode_length
        self._current_episode_end : float = time.perf_counter() + self._episode_length

        self._leftover_executions_previous_episode : List[str] = []
        self._reset_workload_event : threading.Event = reset_workload_event
        self._interrupt_workload_event : threading.Event = interrupt_workload_event
        self._interrupt_workload_event.set()
        self._continue_workload_event : threading.Event = continue_workload_event

        self._rollout_end_time : float | None = None
        self._is_no_action : bool = False

        #self._step_end_time : float = 0.0
        #self._ignore_next : bool = False
        #self.inbetween_step_times : List[float] = []

        self._has_increased_priority = False

        self._step_end_time : float = -1

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) \
            -> tuple[ObsType, dict[str, Any]]:
        if not self._has_increased_priority:
            setup_helper.try_increase_priority(95)
            self._has_increased_priority = True

        self._reset_workload_event.set()
        super().reset(seed=seed)
        self._is_no_action = True
        print("total_episode_reward:", self._total_episode_reward)
        print("current time:", datetime.now().strftime("%H:%M:%S.%f")[:-3])

        # self._remaining_prep_actions : int = 0
        # self._remaining_execution_actions : int = 0
        self._next_stream_idx : int = -1
        self._remaining_prep_actions_post : int = 0

        self._is_prep_action : bool = False
        self._is_execution_action : bool = False

        self._total_episode_reward : float = 0.0

        self._current_episode_end: float = time.perf_counter() + self._episode_length

        self._pending_requests.clear()
        self._prepared_requests.clear()
        with self._new_requests_lock:
            self._new_requests.clear()

        self._leftover_executions_previous_episode.clear()
        with self._completed_requests_lock:
            for stats in self._completed_requests:
                self._leftover_executions_previous_episode.append(stats.request.request_id)
        for request in self._ongoing_requests:
            if request is not None:
                self._leftover_executions_previous_episode.append(request[0].request_id)

        self._ongoing_requests = [None for _ in range(len(self._cuda_streams))]

        current_observation = self._agent_wrapper.generate_observation_vector(
            self._pending_requests, self._prepared_requests, self._ongoing_requests, time.perf_counter()
        )
        self._current_observation = current_observation
        return current_observation, {}

    def step(self, action: ActType)-> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        #timing_network_current = time.perf_counter()
        #if not self._ignore_next:
        #    time_diff = timing_network_current - self._step_end_time
        #    self.inbetween_step_times.append(time_diff)
        #self._ignore_next = False
        now = time.perf_counter()
        print(now - self._step_end_time)

        if self._is_prep_action:
            request_index, gpu_config_idx = \
                self._agent_wrapper.decode_preparation_action(action, self._pending_requests, self._prepared_requests)
            self._agent_wrapper.prepare_request_default(self._pending_requests, request_index, gpu_config_idx,
                                                        self._prepared_requests, self)
        elif self._is_execution_action:
            request_index = self._agent_wrapper.decode_execution_action(
                action, self._next_stream_idx, self._prepared_requests
            )
            stream = self._cuda_streams[self._next_stream_idx]
            self._agent_wrapper.execute_request_default(
                self._prepared_requests, request_index, stream, self._next_stream_idx, self._ongoing_requests, self
            )
            self._next_stream_idx = AgentWrapper.select_next_stream_index(self._next_stream_idx, self._ongoing_requests)

        current_reward = self._handle_completed_requests()
        self._total_episode_reward += current_reward
        self._handle_missed_deadlines()
        is_episode_end : bool = False

        free_streams = 0
        for request in self._ongoing_requests:
            if request is None:
                free_streams += 1
        if (len(self._prepared_requests) == len(self._cuda_streams) and free_streams == 0) or \
            (len(self._pending_requests) == 0 and (len(self._prepared_requests) == 0 or free_streams == 0)):
            self._wake_up.wait()
            self._wake_up.clear()
            with self._new_requests_lock:
                self._pending_requests.extend(self._new_requests)
                self._new_requests.clear()
            new_reward = self._handle_completed_requests()
            self._total_episode_reward += new_reward
            current_reward += new_reward
            self._handle_missed_deadlines()
            for request in self._ongoing_requests:
                if request is None:
                    free_streams += 1
        is_no_action = self._is_no_action
        is_prep_action = self._is_prep_action
        is_execution_action = self._is_execution_action
        self._is_no_action = False
        self._is_prep_action = False
        self._is_execution_action = False
        if (is_prep_action or is_no_action) and free_streams > 0 and len(self._prepared_requests) > 0:
            self._is_prep_action = False
            self._is_execution_action = True
            if self._next_stream_idx < 0:
                self._next_stream_idx = AgentWrapper.select_next_stream_index(self._next_stream_idx,
                                                                              self._ongoing_requests)
        elif (is_execution_action or is_no_action) and len(self._prepared_requests) < len(self._cuda_streams) and len(self._pending_requests) > 0:
            self._is_prep_action = True
            self._is_execution_action = False
        else:
            self._is_no_action = True

        current_time = time.perf_counter()
        current_observation = self._agent_wrapper.generate_observation_vector(
            self._pending_requests, self._prepared_requests, self._ongoing_requests, current_time
        )
        self._current_observation = current_observation
        if current_time >= self._current_episode_end:
            is_episode_end = True

        self._step_end_time = time.perf_counter()
        return current_observation, current_reward, is_episode_end, False, {}

    # def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    #     #timing_network_current = time.perf_counter()
    #     #if not self._ignore_next:
    #     #    time_diff = timing_network_current - self._step_end_time
    #     #    self.inbetween_step_times.append(time_diff)
    #     #self._ignore_next = False
    #     current_reward = 0
    #     is_episode_end : bool = False
    #     if self._remaining_prep_actions > 0:
    #         self._remaining_prep_actions -= 1
    #         request_index, gpu_config_idx = \
    #             self._agent_wrapper.decode_preparation_action(action, self._pending_requests, self._prepared_requests)
    #         self._agent_wrapper.prepare_request_default(self._pending_requests, request_index, gpu_config_idx,
    #                                                     self._prepared_requests, self)
    #     elif self._remaining_execution_actions > 0:
    #         self._remaining_execution_actions -= 1
    #         request_index = self._agent_wrapper.decode_execution_action(
    #             action, self._next_stream_idx, self._prepared_requests
    #         )
    #         stream = self._cuda_streams[self._next_stream_idx]
    #         self._agent_wrapper.execute_request_default(
    #             self._prepared_requests, request_index, stream, self._next_stream_idx, self._ongoing_requests, self
    #         )
    #         self._next_stream_idx = AgentWrapper.select_next_stream_index(self._next_stream_idx, self._ongoing_requests)
    #     if self._remaining_prep_actions == 0 and self._remaining_execution_actions == 0:
    #         self._wake_up.wait()
    #         self._wake_up.clear()
    #         with self._new_requests_lock:
    #             self._pending_requests.extend(self._new_requests)
    #             self._new_requests.clear()
    #         current_reward = self._handle_completed_requests()
    #         self._total_episode_reward += current_reward
    #         self._handle_missed_deadlines()
    #         # there won't be more pending requests until all these actions have been selected
    #         self._remaining_prep_actions = min(len(self._cuda_streams) - len(self._prepared_requests), len(self._pending_requests))
    #         self._remaining_execution_actions = 0
    #         for request in self._ongoing_requests:
    #             if request is None:
    #                 self._remaining_execution_actions += 1
    #         # do not set to min(self._remaining_execution_actions, self._prepared_requests) since more may be prepared before the actions are selected
    #
    #         # self._next_stream_idx is less than 0 if no stream was free during last selection
    #         if self._next_stream_idx < 0:
    #             self._next_stream_idx = self._agent_wrapper.select_next_stream_index(self._next_stream_idx, self._ongoing_requests)
    #
    #     current_time = time.perf_counter()
    #     current_observation = self._agent_wrapper.generate_observation_vector(
    #         self._pending_requests, self._prepared_requests, self._ongoing_requests, current_time
    #     )
    #     self._current_observation = current_observation
    #     if current_time >= self._current_episode_end:
    #         is_episode_end = True
    #
    #     # can happen when a new request arrives while we already have max prepared requests and no cuda stream is free
    #     if self._remaining_prep_actions == 0 and self._remaining_execution_actions == 0:
    #         self._is_no_action = True
    #
    #     #self._step_end_time = time.perf_counter()
    #     # terminating without truncation, since requests are fully reset, but I don't expect a big difference
    #     return current_observation, current_reward, is_episode_end, False, {}

    def _handle_completed_requests(self) -> float:
        current_reward = 0.0
        with (self._completed_requests_lock):
            for current in self._completed_requests:
                if current.request.request_id in self._leftover_executions_previous_episode:
                    self._leftover_executions_previous_episode.remove(current.request.request_id)
                    continue
                scheduling_result = SchedulingResult.create(current)
                current_reward += calculate_xrbench_per_inference_score_no_energy(scheduling_result, True)
                for i in range(len(self._ongoing_requests)):
                    ongoing_request = self._ongoing_requests[i]
                    if ongoing_request is not None and \
                        ongoing_request[0].request_id == current.request.request_id:
                        self._ongoing_requests[i] = None
            self._completed_requests.clear()
        return current_reward / 100

    def _handle_missed_deadlines(self) -> None:
        current_time = time.perf_counter()
        self.remove_missed_deadline_requests(self._pending_requests, current_time)
        self.remove_missed_deadline_requests_with_tuple(self._prepared_requests, current_time)


    def warmup(self, tasks: List[InferenceTask], warmup_iterations: int, custom_batch_sizes: Dict[str, int] = None):
        if custom_batch_sizes is None:
            custom_batch_sizes = {}
        for i, stream in enumerate(self._cuda_streams):
            SchedulingStrategy.warmup_gpu(tasks, warmup_iterations, custom_batch_sizes, stream, i)

    def on_rollout_start(self) -> None:
        #print("started rollout; current time:", datetime.now().strftime("%H:%M:%S.%f")[:-3])

        #self._ignore_next = True
        self._continue_workload_event.set()
        if self._rollout_end_time is not None:
            time_diff = time.perf_counter() - self._rollout_end_time
            for request in self._pending_requests:
                request.deadline += time_diff
            for request, _ in self._prepared_requests:
                request.deadline += time_diff

    def on_rollout_end(self) -> None:
        #print("finished rollout; current time:", datetime.now().strftime("%H:%M:%S.%f")[:-3])
        #print("current reward:", self._total_episode_reward)
        self._interrupt_workload_event.set()
        self._rollout_end_time = time.perf_counter()

    def on_training_end(self) -> None:
        #print("training end reward:", self._total_episode_reward)
        print("shutting down scheduler thread pool")
        self._thread_pool.shutdown()
        print("terminated scheduler")

    def action_masks(self) -> np.ndarray:
        if self._is_no_action:
            return self._agent_wrapper.generate_action_mask(self._current_observation, True, -1)
        if self._is_prep_action:
            return self._agent_wrapper.generate_action_mask(self._current_observation, True, -1)
        if self._is_execution_action:
            return self._agent_wrapper.generate_action_mask(self._current_observation, False,
                                                          self._next_stream_idx)
        raise ValueError("Error, attempting to create action_mask while no prep action and no execution action "
                         "should be taken")

    # def action_masks(self) -> np.ndarray:
    #     if self._remaining_prep_actions > 0 or self._is_no_action:
    #         self._is_no_action = False
    #         return self._agent_wrapper.generate_action_mask(self._current_observation, True, -1)
    #     if self._remaining_execution_actions > 0:
    #         return self._agent_wrapper.generate_action_mask(self._current_observation, False,
    #                                                       self._next_stream_idx)
    #     if self._remaining_prep_actions_post > 0:
    #         return self._agent_wrapper.generate_action_mask(self._current_observation, True, -1)
    #     raise ValueError("Error, attempting to create action_mask while no prep action and no execution action "
    #                      "should be taken")


class AgentFunction(Enum):
    RESET = auto()
    STEP = auto()
    ACTION_MASKS = auto()
    ROLLOUT_START = auto()
    ROLLOUT_END = auto()
    TRAINING_END = auto()

class AgentTrainingEnv(gym.Env):
    def __init__(self, observation_length : int, num_actions : int,
                 pipe : Connection):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(observation_length,),
                                                dtype=np.float32)
        self.action_space = gym.spaces.Discrete(num_actions)

        self._pipe : Connection = pipe

        self._has_increased_priority = False

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) \
            -> tuple[ObsType, dict[str, Any]]:
        if not self._has_increased_priority:
            setup_helper.try_increase_priority(95)
            self._has_increased_priority = True

        super().reset(seed=seed)

        self._pipe.send((AgentFunction.RESET,))
        observation = self._pipe.recv()
        return observation, {}

    def step(self, action : ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self._pipe.send((AgentFunction.STEP, action))
        current_observation, current_reward, is_episode_end = self._pipe.recv()
        return current_observation, current_reward, is_episode_end, False, {}

    def on_rollout_start(self) -> None:

        self._pipe.send((AgentFunction.ROLLOUT_START,))
        # wait until on_rollout_start in other scheduler process is finished
        _ = self._pipe.recv()

    def on_rollout_end(self) -> None:
        #print("finished rollout child; current time:", datetime.now().strftime("%H:%M:%S.%f")[:-3])
        self._pipe.send((AgentFunction.ROLLOUT_END,))
        _ = self._pipe.recv()

    def on_training_end(self) -> None:
        self._pipe.send((AgentFunction.TRAINING_END,))
        _ = self._pipe.recv()

    def action_masks(self) -> np.ndarray:
        self._pipe.send((AgentFunction.ACTION_MASKS,))
        action_mask = self._pipe.recv()
        return action_mask


# noinspection DuplicatedCode
class AgentTrainingScheduler(DefaultScheduler):
    def __init__(self, agent_wrapper : AgentWrapper, cuda_streams : List[torch.cuda.Stream],
                 episode_length : float, pipe : Connection,
                 reset_workload_event: threading.Event,
                 interrupt_workload_event: threading.Event,
                 continue_workload_event: threading.Event
                 ):
        super().__init__()
        self._agent_wrapper = agent_wrapper
        self._pipe : Connection = pipe

        self._cuda_streams : List[torch.cuda.Stream] = cuda_streams
        self._prepared_requests : List[Tuple[InferenceRequest, float]] = []
        self._ongoing_requests : List[Tuple[InferenceRequest, float] | None] = [None for _ in range(len(cuda_streams))]

        self._reset_workload_event : threading.Event = reset_workload_event
        self._interrupt_workload_event : threading.Event = interrupt_workload_event
        self._interrupt_workload_event.set()
        self._continue_workload_event : threading.Event = continue_workload_event

        self._current_observation : np.ndarray = np.zeros(agent_wrapper.observation_length)

        self._remaining_prep_actions : int = 0
        self._remaining_execution_actions : int = 0
        self._next_stream_idx : int = -1
        # prepare additional actions parallel to current executions
        self._remaining_prep_actions_post : int = 0

        # self._is_prep_action : bool = False
        # self._is_execution_action : bool = False

        self._total_episode_reward : float = 0.0

        self._episode_length : float = episode_length
        self._current_episode_end : float = time.perf_counter() + self._episode_length

        self._leftover_executions_previous_episode: List[str] = []

        self._rollout_end_time : float | None = None
        self._is_no_action : bool = False

        self._step_end_time : float = 0.0
        self._ignore_next : bool = False
        self.inbetween_step_times : List[float] = []

        self._has_increased_priority = False

    def reset(self) -> np.ndarray:
        if not self._has_increased_priority:
            setup_helper.try_increase_priority(95)
            self._has_increased_priority = True

        self._reset_workload_event.set()
        self._is_no_action = True
        print("total_episode_reward:", self._total_episode_reward)
        print("current time:", datetime.now().strftime("%H:%M:%S.%f")[:-3])

        self._remaining_prep_actions : int = 0
        self._remaining_execution_actions : int = 0
        self._next_stream_idx : int = -1
        self._remaining_prep_actions_post : int = 0

        # self._is_prep_action : bool = False
        # self._is_execution_action : bool = False

        self._total_episode_reward : float = 0.0

        self._current_episode_end: float = time.perf_counter() + self._episode_length

        self._pending_requests.clear()
        self._prepared_requests.clear()
        with self._new_requests_lock:
            self._new_requests.clear()

        self._leftover_executions_previous_episode.clear()
        with self._completed_requests_lock:
            for stats in self._completed_requests:
                self._leftover_executions_previous_episode.append(stats.request.request_id)
        for request in self._ongoing_requests:
            if request is not None:
                self._leftover_executions_previous_episode.append(request[0].request_id)

        self._ongoing_requests = [None for _ in range(len(self._cuda_streams))]

        current_observation = self._agent_wrapper.generate_observation_vector(
            self._pending_requests, self._prepared_requests, self._ongoing_requests, time.perf_counter()
        )
        self._current_observation = current_observation
        return current_observation

    # def step(self, action : ActType) -> tuple[np.ndarray, float, bool]:
    #     #timing_network_current = time.perf_counter()
    #     #if not self._ignore_next:
    #     #    time_diff = timing_network_current - self._step_end_time
    #     #    self.inbetween_step_times.append(time_diff)
    #     #self._ignore_next = False
    #
    #     if self._is_prep_action:
    #         request_index, gpu_config_idx = \
    #             self._agent_wrapper.decode_preparation_action(action, self._pending_requests, self._prepared_requests)
    #         self._agent_wrapper.prepare_request_default(self._pending_requests, request_index, gpu_config_idx,
    #                                                     self._prepared_requests, self)
    #     elif self._is_execution_action:
    #         request_index = self._agent_wrapper.decode_execution_action(
    #             action, self._next_stream_idx, self._prepared_requests
    #         )
    #         stream = self._cuda_streams[self._next_stream_idx]
    #         self._agent_wrapper.execute_request_default(
    #             self._prepared_requests, request_index, stream, self._next_stream_idx, self._ongoing_requests, self
    #         )
    #         self._next_stream_idx = AgentWrapper.select_next_stream_index(self._next_stream_idx, self._ongoing_requests)
    #
    #     with self._new_requests_lock:
    #         self._pending_requests.extend(self._new_requests)
    #         self._new_requests.clear()
    #     current_reward = self._handle_completed_requests()
    #     self._wake_up.clear()
    #     self._total_episode_reward += current_reward
    #     self._handle_missed_deadlines()
    #     is_episode_end : bool = False
    #
    #     free_streams = 0
    #     for request in self._ongoing_requests:
    #         if request is None:
    #             free_streams += 1
    #     if (len(self._prepared_requests) == len(self._cuda_streams) and free_streams == 0) or \
    #         (len(self._pending_requests) == 0 and (len(self._prepared_requests) == 0 or free_streams == 0)):
    #         self._wake_up.wait()
    #         self._wake_up.clear()
    #         new_reward = self._handle_completed_requests()
    #         self._total_episode_reward += new_reward
    #         current_reward += new_reward
    #         self._handle_missed_deadlines()
    #         for request in self._ongoing_requests:
    #             if request is None:
    #                 free_streams += 1
    #     is_no_action = self._is_no_action
    #     is_prep_action = self._is_prep_action
    #     is_execution_action = self._is_execution_action
    #     self._is_no_action = False
    #     self._is_prep_action = False
    #     self._is_execution_action = False
    #     if (is_prep_action or is_no_action) and free_streams > 0 and len(self._prepared_requests) > 0:
    #         self._is_prep_action = False
    #         self._is_execution_action = True
    #         if self._next_stream_idx < 0:
    #             self._next_stream_idx = AgentWrapper.select_next_stream_index(self._next_stream_idx,
    #                                                                           self._ongoing_requests)
    #     elif (is_execution_action or is_no_action) and len(self._prepared_requests) < len(self._cuda_streams) and len(self._pending_requests) > 0:
    #         self._is_prep_action = True
    #         self._is_execution_action = False
    #     # allow staying in same action type if we can't switch right now
    #     elif free_streams > 0 and len(self._prepared_requests) > 0:
    #         self._is_prep_action = False
    #         self._is_execution_action = True
    #         if self._next_stream_idx < 0:
    #             self._next_stream_idx = AgentWrapper.select_next_stream_index(self._next_stream_idx,
    #                                                                           self._ongoing_requests)
    #     elif len(self._prepared_requests) < len(self._cuda_streams) and len(self._pending_requests) > 0:
    #         self._is_prep_action = True
    #         self._is_execution_action = False
    #     else:
    #         self._is_no_action = True
    #
    #     current_time = time.perf_counter()
    #     current_observation = self._agent_wrapper.generate_observation_vector(
    #         self._pending_requests, self._prepared_requests, self._ongoing_requests, current_time
    #     )
    #     self._current_observation = current_observation
    #     if current_time >= self._current_episode_end:
    #         is_episode_end = True
    #
    #     return current_observation, current_reward, is_episode_end

    def step(self, action : ActType) -> tuple[np.ndarray, float, bool]:
        timing_network_current = time.perf_counter()
        if not self._ignore_next:
            time_diff = timing_network_current - self._step_end_time
            self.inbetween_step_times.append(time_diff)
        self._ignore_next = False
        current_reward = 0
        is_episode_end : bool = False
        if self._remaining_prep_actions > 0:
            self._remaining_prep_actions -= 1
            request_index, gpu_config_idx = \
                self._agent_wrapper.decode_preparation_action(action, self._pending_requests, self._prepared_requests)
            self._agent_wrapper.prepare_request_default(self._pending_requests, request_index, gpu_config_idx,
                                                        self._prepared_requests, self)
        elif self._remaining_execution_actions > 0:
            self._remaining_execution_actions -= 1
            request_index = self._agent_wrapper.decode_execution_action(
                action, self._next_stream_idx, self._prepared_requests
            )
            stream = self._cuda_streams[self._next_stream_idx]
            self._agent_wrapper.execute_request_default(
                self._prepared_requests, request_index, stream, self._next_stream_idx, self._ongoing_requests, self
            )
            self._next_stream_idx = AgentWrapper.select_next_stream_index(self._next_stream_idx, self._ongoing_requests)
        if self._remaining_prep_actions == 0 and self._remaining_execution_actions == 0:
            self._wake_up.wait()
            self._wake_up.clear()
            with self._new_requests_lock:
                self._pending_requests.extend(self._new_requests)
                self._new_requests.clear()
            current_reward = self._handle_completed_requests()
            self._total_episode_reward += current_reward
            self._handle_missed_deadlines()
            # there won't be more pending requests until all these actions have been selected
            self._remaining_prep_actions = min(len(self._cuda_streams) - len(self._prepared_requests), len(self._pending_requests))
            self._remaining_execution_actions = 0
            for request in self._ongoing_requests:
                if request is None:
                    self._remaining_execution_actions += 1
            # do not set to min(self._remaining_execution_actions, self._prepared_requests) since more may be prepared before the actions are selected

            # self._next_stream_idx is less than 0 if no stream was free during last selection
            if self._next_stream_idx < 0:
                self._next_stream_idx = self._agent_wrapper.select_next_stream_index(self._next_stream_idx, self._ongoing_requests)

        current_time = time.perf_counter()
        current_observation = self._agent_wrapper.generate_observation_vector(
            self._pending_requests, self._prepared_requests, self._ongoing_requests, current_time
        )
        self._current_observation = current_observation
        if current_time >= self._current_episode_end:
            is_episode_end = True

        # can happen when a new request arrives while we already have max prepared requests and no cuda stream is free
        if self._remaining_prep_actions == 0 and self._remaining_execution_actions == 0:
            self._is_no_action = True

        self._step_end_time = time.perf_counter()
        # terminating without truncation, since requests are fully reset, but I don't expect a big difference
        return current_observation, current_reward, is_episode_end

    def _handle_completed_requests(self) -> float:
        current_reward = 0.0
        with (self._completed_requests_lock):
            for current in self._completed_requests:
                if current.request.request_id in self._leftover_executions_previous_episode:
                    self._leftover_executions_previous_episode.remove(current.request.request_id)
                    continue
                scheduling_result = SchedulingResult.create(current)
                current_reward += calculate_xrbench_per_inference_score_no_energy(scheduling_result, True)
                for i in range(len(self._ongoing_requests)):
                    ongoing_request = self._ongoing_requests[i]
                    if ongoing_request is not None and \
                        ongoing_request[0].request_id == current.request.request_id:
                        self._ongoing_requests[i] = None
            self._completed_requests.clear()
        return current_reward

    def _handle_missed_deadlines(self) -> None:
        current_time = time.perf_counter()
        self.remove_missed_deadline_requests(self._pending_requests, current_time)
        self.remove_missed_deadline_requests_with_tuple(self._prepared_requests, current_time)

    def warmup(self, tasks: List[InferenceTask], warmup_iterations: int, custom_batch_sizes: Dict[str, int] = None):
        if custom_batch_sizes is None:
            custom_batch_sizes = {}
        for i, stream in enumerate(self._cuda_streams):
            SchedulingStrategy.warmup_gpu(tasks, warmup_iterations, custom_batch_sizes, stream, i)

    def on_rollout_start(self) -> None:
        #print("started rollout; current time:", datetime.now().strftime("%H:%M:%S.%f")[:-3])

        self._ignore_next = True
        self._continue_workload_event.set()
        if self._rollout_end_time is not None:
            time_diff = time.perf_counter() - self._rollout_end_time
            for request in self._pending_requests:
                request.deadline += time_diff
            for request, _ in self._prepared_requests:
                request.deadline += time_diff

    def on_rollout_end(self) -> None:
        # print("finished rollout; current time:", datetime.now().strftime("%H:%M:%S.%f")[:-3])
        # print("current reward:", self._total_episode_reward)

        self._interrupt_workload_event.set()
        self._rollout_end_time = time.perf_counter()

    def on_training_end(self) -> None:
        #print("training end reward:", self._total_episode_reward)
        print("shutting down scheduler thread pool")
        self._thread_pool.shutdown()
        print("terminated scheduler")

    # def action_masks(self) -> np.ndarray:
    #     if self._is_no_action:
    #         return self._agent_wrapper.generate_action_mask(self._current_observation, True, -1)
    #     if self._is_prep_action:
    #         return self._agent_wrapper.generate_action_mask(self._current_observation, True, -1)
    #     if self._is_execution_action:
    #         return self._agent_wrapper.generate_action_mask(self._current_observation, False,
    #                                                         self._next_stream_idx)
    #     raise ValueError("Error, attempting to create action_mask while no prep action and no execution action "
    #                      "should be taken")

    def action_masks(self) -> np.ndarray:
        if self._remaining_prep_actions > 0 or self._is_no_action:
            self._is_no_action = False
            return self._agent_wrapper.generate_action_mask(self._current_observation, True, -1)
        if self._remaining_execution_actions > 0:
            return self._agent_wrapper.generate_action_mask(self._current_observation, False,
                                                          self._next_stream_idx)
        if self._remaining_prep_actions_post > 0:
            return self._agent_wrapper.generate_action_mask(self._current_observation, True, -1)
        raise ValueError("Error, attempting to create action_mask while no prep action and no execution action "
                         "should be taken")

    def handle_agent_process_requests(self):
        while True:
            received = self._pipe.recv()
            match received[0]:
                case AgentFunction.RESET:
                    observation = self.reset()
                    self._pipe.send(observation)

                case AgentFunction.STEP:
                    step_out = self.step(received[1])
                    self._pipe.send(step_out)

                case AgentFunction.ACTION_MASKS:
                    action_masks = self.action_masks()
                    self._pipe.send(action_masks)

                case AgentFunction.ROLLOUT_START:
                    self.on_rollout_start()
                    self._pipe.send(AgentFunction.ROLLOUT_START)

                case AgentFunction.ROLLOUT_END:
                    self.on_rollout_end()
                    self._pipe.send(AgentFunction.ROLLOUT_END)

                case AgentFunction.TRAINING_END:
                    self.on_training_end()
                    self._pipe.send(AgentFunction.TRAINING_END)
                    return

                case _:
                    raise ValueError(f"Error, could not decode the following message from the agent process:\n{str(received)}")
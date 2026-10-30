from __future__ import annotations

import time
from typing import List, Dict, Final, Tuple

import torch.cuda
from sb3_contrib import MaskablePPO

from scheduling.AgentWrapper import AgentWrapper
from scheduling.InferenceRequest import InferenceStats, InferenceRequest
from scheduling.strategy.SchedulingStrategy import SchedulingStrategy
from tasks.InferenceTask import InferenceTask

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scheduling.BenchmarkScheduler import BenchmarkScheduler

# def separate_process_inference(child_pipe : multiprocessing.connection.Connection, agent_path : str):
#     setup_helper.try_increase_priority()
#     agent = MaskablePPO.load(agent_path, device="cpu")
#     print("successfully started agent inference process")
#     while True:
#         command = child_pipe.recv()
#         if command[0] == 0:
#             return
#         if command[0] == 1:
#             current_observation, action_mask = command[1], command[2]
#             action, _ = agent.predict(current_observation, action_masks=action_mask, deterministic=True)
#             child_pipe.send(action)


class RLAgent(SchedulingStrategy):
    def __init__(self, benchmark_scheduler : BenchmarkScheduler, agent_path : str,
                 agent_wrapper : AgentWrapper, cuda_streams : List[torch.cuda.Stream]):
        self._benchmark_scheduler : Final[BenchmarkScheduler] = benchmark_scheduler

        self._agent_path : Final[str] = agent_path
        self._agent = MaskablePPO.load(agent_path, device="cpu")

        # parent_pipe, child_pipe = multiprocessing.Pipe()
        # self._parent_pipe = parent_pipe

        # multiprocessing.set_start_method("spawn")
        # ctx = multiprocessing.get_context("spawn")
        # # noinspection PyUnresolvedReferences
        # p = ctx.Process(target=separate_process_inference, args=(child_pipe, agent_path))
        # p.start()

        self._agent_wrapper = agent_wrapper

        if len(cuda_streams) != agent_wrapper.num_streams:
            raise ValueError("Error, the given agent wrapper expects a different number of cuda streams than was given")

        self._streams : List[torch.cuda.Stream] = cuda_streams
        self._num_streams = len(cuda_streams)

        self._prepared_requests : List[Tuple[InferenceRequest, float]] = []
        self._ongoing_requests: List[Tuple[InferenceRequest, float] | None] = [None for _ in range(len(cuda_streams))]

        self._canceled_request_ids : List[str] = []

        self._next_stream_idx: int = -1

        #self._inference_times = []

    def invoke(self, pending_requests: List[InferenceRequest]) -> None:
        now = time.perf_counter()
        to_remove = []
        for request, _ in self._prepared_requests:
            if request is not None and now >= request.deadline:
                self._benchmark_scheduler.cancel_request_gpu(request)
                self._canceled_request_ids.append(request.request_id)
                self._benchmark_scheduler.handle_dropped_request(request, now)
                to_remove.append(request)
        self._prepared_requests[:] = [request for request in self._prepared_requests if request not in to_remove]
        self._benchmark_scheduler.remove_and_handle_missed_deadline_requests(pending_requests, now)

        num_prep_actions = min(len(self._streams) - len(self._prepared_requests), len(pending_requests))
        for _ in range(num_prep_actions):
            current_observation = self._agent_wrapper.generate_observation_vector(
                pending_requests, self._prepared_requests, self._ongoing_requests, now
            )
            action_mask = self._agent_wrapper.generate_action_mask(current_observation, True, -1)
            #t1 = time.perf_counter()
            # self._parent_pipe.send((1, current_observation, action_mask))
            # action = self._parent_pipe.recv()
            with torch.no_grad():
                action, _ = self._agent.predict(current_observation, action_masks=action_mask, deterministic=True)
            #t2 = time.perf_counter()
            #if len(self._inference_times) < 10000:
            #    self._inference_times.append(t2 - t1)
            request_idx, gpu_config_idx = \
                self._agent_wrapper.decode_preparation_action(action.item(), pending_requests, self._prepared_requests)
            self._agent_wrapper.prepare_request_default(pending_requests, request_idx, gpu_config_idx,
                                                        self._prepared_requests, self._benchmark_scheduler)
        free_streams = 0
        for request in self._ongoing_requests:
            if request is None:
                free_streams += 1
        num_execution_actions = min(free_streams, len(self._prepared_requests))
        if self._next_stream_idx < 0:
            self._next_stream_idx = self._agent_wrapper.select_next_stream_index(self._next_stream_idx, self._ongoing_requests)
        for _ in range(num_execution_actions):
            current_observation = self._agent_wrapper.generate_observation_vector(
                pending_requests, self._prepared_requests, self._ongoing_requests, now
            )
            action_mask = self._agent_wrapper.generate_action_mask(current_observation, False,
                                                                   self._next_stream_idx)
            #t1 = time.perf_counter()
            # self._parent_pipe.send((1, current_observation, action_mask))
            # action = self._parent_pipe.recv()
            with torch.no_grad():
                action, _ = self._agent.predict(current_observation, action_masks=action_mask, deterministic=True)
            #t2 = time.perf_counter()
            #if len(self._inference_times) < 10000:
            #    self._inference_times.append(t2 - t1)
            #else:
            #    print(mean(self._inference_times))
            request_index = self._agent_wrapper.decode_execution_action(
                action.item(), self._next_stream_idx, self._prepared_requests
            )
            stream = self._streams[self._next_stream_idx]
            self._agent_wrapper.execute_request_default(
                self._prepared_requests, request_index, stream, self._next_stream_idx, self._ongoing_requests, self._benchmark_scheduler
            )
            self._next_stream_idx = AgentWrapper.select_next_stream_index(self._next_stream_idx, self._ongoing_requests)

    def get_num_cuda_streams(self) -> int:
        return self._num_streams

    def notify_of_completion(self, stats: InferenceStats) -> None:
        request_id = stats.request.request_id
        if request_id in self._canceled_request_ids:
            self._canceled_request_ids.remove(request_id)
            return

        for i in range(len(self._ongoing_requests)):
            ongoing_request = self._ongoing_requests[i]
            if ongoing_request is not None and ongoing_request[0].request_id == request_id:
                self._ongoing_requests[i] = None
                return

    def handle_termination(self, current_time: float) -> None:
        for request, _ in self._prepared_requests:
            if request is not None:
                self._benchmark_scheduler.handle_dropped_request(request, current_time)
                self._benchmark_scheduler.cancel_request_gpu(request)
        self._prepared_requests.clear()
        #self._parent_pipe.send((0,))


    def warmup(self, tasks: List[InferenceTask], warmup_iterations, custom_batch_sizes: Dict[str, int]):
        for i, stream in enumerate(self._streams):
            SchedulingStrategy.warmup_gpu(tasks, warmup_iterations, custom_batch_sizes, stream, i)
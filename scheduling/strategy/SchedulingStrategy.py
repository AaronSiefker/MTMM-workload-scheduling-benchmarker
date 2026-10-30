from abc import ABC, abstractmethod
from typing import List, Dict, Final

import torch

from scheduling.InferenceRequest import InferenceRequest, InferenceStats
from scheduling.Scheduler import SchedulingResult
from tasks.InferenceTask import InferenceTask


class SchedulingStrategy(ABC):

    @abstractmethod
    def invoke(self, pending_requests : List[InferenceRequest]) -> None:
        pass

    @abstractmethod
    def notify_of_completion(self, stats : InferenceStats) -> None:
        pass

    def on_energy_violation(self, results : List[SchedulingResult]):
        pass

    @abstractmethod
    def handle_termination(self, current_time : float) -> None:
        pass

    @abstractmethod
    def warmup(self, tasks : List[InferenceTask], warmup_iterations, custom_batch_sizes : Dict[str, int]):
        pass

    @abstractmethod
    def get_num_cuda_streams(self) -> int:
        pass

    WARMUP_REQUEST_ID : Final[str] = "warmup_execution"
    @staticmethod
    def warmup_gpu(tasks : List[InferenceTask], warmup_iterations : int, custom_batch_sizes : Dict[str, int],
                   stream : torch.cuda.Stream, stream_index : int):
        for i in range(warmup_iterations):
            for task in tasks:
                batch_size = 1
                if task.task in custom_batch_sizes.keys():
                    batch_size = custom_batch_sizes[task.task]
                dummy, _ = task.generate_input(batch_size)
                task.get_task_engine().active_gpu_configuration.execute(dummy, stream, SchedulingStrategy.WARMUP_REQUEST_ID,
                                                                        stream_index=stream_index)
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scheduling.InferenceRequest import InferenceRequest, InferenceStats


@dataclass(frozen=True)
class SchedulingResult:
    request_id : str

    model_id : str
    model_variant : str
    quality_score : float

    device_name : str
    finished_execution : bool
    preparation_time : float
    inference_time : float

    request_time : float
    finish_time : float
    deadline : float

    consumed_energy : float

    batch_size : int
    false_positives : int
    predecessor_request_id : str | None

    def get_start_time(self) -> float:
        return self.finish_time - self.inference_time

    @staticmethod
    def create(stats : InferenceStats) -> SchedulingResult:
        return SchedulingResult(
            stats.request.request_id,

            stats.task_config.model_id,
            stats.task_config.model_variant,
            stats.task_config.quality,

            stats.device.value,
            stats.finished_execution,
            stats.preparation_time,
            stats.inference_time,

            stats.request.request_time,
            stats.finish_time,
            stats.request.deadline,

            stats.consumed_energy,

            # convert from numpy.int64 for serializability
            int(stats.request.batch_size),
            int(stats.request.false_positives),
            stats.request.predecessor_request_id
        )


class Scheduler(ABC):
    @abstractmethod
    def add_request(self, request : InferenceRequest) -> None:
        pass

class NullScheduler(Scheduler):
    def add_request(self, request : InferenceRequest) -> None:
        print(request.deadline)
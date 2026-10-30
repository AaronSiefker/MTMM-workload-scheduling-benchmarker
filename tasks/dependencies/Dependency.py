from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Final, List, Tuple

from misc.StaticRng import StaticRng
from scheduling.InferenceRequest import InferenceRequest
from scheduling.Scheduler import Scheduler

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tasks.InferenceTask import InferenceTask


class Dependency(ABC):
    def __init__(self, task : InferenceTask, is_deadline_shared : bool,
                 predecessor_deadline_offset : float | None):
        self._task : Final[InferenceTask] = task
        self.is_deadline_shared : Final[bool] = is_deadline_shared
        self._rng : Final[StaticRng] = StaticRng()
        self._predecessor_deadline_offset : Final[float | None] = predecessor_deadline_offset

    @abstractmethod
    def generate_inference_request(self, scheduler: Scheduler, predecessor_request_id : str | None,
                                   predecessor_request_time : float, predecessor_deadline : float,
                                   false_positive_probability : float) -> None:
        pass

    @abstractmethod
    def accumulate_dropped_requests(self, predecessor_request_id : str) -> List[InferenceRequest]:
        """
        accumulates all InferenceRequests which would have been generated for the request with id 'predecessor_request_id'
         which has been dropped.
        :param predecessor_request_id: The id of the dropped request.
        :return: a list of all such InferenceRequests which would have been generated upon completion of the request
        along with their successors.
        """
        pass

    @property
    def max_dependency_path_length(self) -> int:
        return self._task.max_dependency_path_length

    def _finalize_request_time_and_deadline(self, task_generated_deadline : float, predecessor_request_time,
                                            predecessor_deadline : float) -> Tuple[float, float]:
        deadline = task_generated_deadline
        request_time = time.perf_counter()
        if self._predecessor_deadline_offset is not None:
            deadline = predecessor_deadline + self._predecessor_deadline_offset
            request_time = predecessor_request_time
        if self.is_deadline_shared:
            deadline = predecessor_deadline
            request_time = predecessor_request_time
        return request_time, deadline
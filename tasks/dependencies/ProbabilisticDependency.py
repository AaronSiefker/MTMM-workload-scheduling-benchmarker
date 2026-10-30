from typing import Final, List

from scheduling.InferenceRequest import InferenceRequest
from scheduling.Scheduler import Scheduler
from tasks.InferenceTask import InferenceTask
from tasks.dependencies.Dependency import Dependency


class ProbabilisticDependency(Dependency):
    def __init__(self, task: InferenceTask, is_deadline_shared : bool, input_created_probability: float,
                 predecessor_deadline_offset : float | None = None, batch_size: int = 1):
        super().__init__(task, is_deadline_shared, predecessor_deadline_offset)
        if input_created_probability < 0 or input_created_probability > 1:
            raise ValueError("Error, input_created_probability must be in [0, 1], "
                             "but provided input_created_probability is", input_created_probability)
        self._input_created_probability : Final[float] = input_created_probability

        if batch_size <= 0:
            raise ValueError("Error, batch_size must greater than 0, but provided batch_size is", batch_size)
        self._batch_size : Final[int] = batch_size

    def generate_inference_request(self, scheduler: Scheduler, predecessor_request_id : str | None,
                                   predecessor_request_time: float, predecessor_deadline: float,
                                   false_positive_probability: float) -> None:
        x, y  = self._rng.random(), self._rng.random()
        is_false_positive = False
        if x < self._input_created_probability:
            pass
        elif y < false_positive_probability:
            is_false_positive = True
        else:
            return
        input_array, deadline = self._task.generate_input(self._batch_size)
        request_time, deadline = self._finalize_request_time_and_deadline(deadline, predecessor_request_time, predecessor_deadline)

        num_false_positives = 1 if is_false_positive else 0
        scheduler.add_request(InferenceRequest(self._task, predecessor_request_id,
                                               input_array, request_time, deadline, self._batch_size, num_false_positives,
                                               self.max_dependency_path_length, self.is_deadline_shared))

    def accumulate_dropped_requests(self, predecessor_request_id : str) -> List[InferenceRequest]:
        x = self._rng.random()
        if x >= self._input_created_probability:
            return []
        request = InferenceRequest(self._task, predecessor_request_id, {}, -1, -1, self._batch_size,
                                0, self.max_dependency_path_length, self.is_deadline_shared)
        out = [request]
        for dependency in self._task.dependencies:
            out.extend(dependency.accumulate_dropped_requests(request.request_id))
        return out
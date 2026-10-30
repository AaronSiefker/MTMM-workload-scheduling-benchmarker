from typing import List

from scheduling.InferenceRequest import InferenceRequest
from scheduling.Scheduler import Scheduler
from tasks.InferenceTask import InferenceTask
from tasks.dependencies.Dependency import Dependency


class RandomBatchSizeDependency(Dependency):
    def __init__(self, task: InferenceTask, is_deadline_shared : bool, batch_sizes: List[int],
                 predecessor_deadline_offset : float | None = None):
        super().__init__(task, is_deadline_shared, predecessor_deadline_offset)
        if any(batch_size <= 0 for batch_size in batch_sizes):
            raise ValueError("Error, the provided batch_sizes contains a batch size which isn't greater than 0."
                             "Provided batch_sizes:", batch_sizes)
        self._batch_sizes : List[int] = list(batch_sizes)
        self._max_batch_size = max(self._batch_sizes)

    def generate_inference_request(self, scheduler: Scheduler, predecessor_request_id : str | None,
                                   predecessor_request_time: float, predecessor_deadline: float,
                                   false_positive_probability: float) -> None:
        # type system had a bit of trouble here without this annotation, got the following warning for num_false_positives
        # in InferenceRequest constructor:
        # Expected type 'int', got 'int | ndarray[tuple[Any, ...], dtype[signedinteger[_64Bit]]]' instead
        #
        # probably because the function is overloaded with different return types
        batch_size : int = self._batch_sizes[int(self._rng.random() * len(self._batch_sizes))]
        #batch_size : int = self._rng.choice(self._batch_sizes)
        input_array, deadline = self._task.generate_input(batch_size)

        # In other places numpy rng was really slow => replacing everywhere to be safe
        #num_false_positives = self._rng.binomial(batch_size, false_positive_probability)
        num_false_positives = 0
        for _ in range(batch_size):
            random_number = self._rng.random()
            if random_number < false_positive_probability:
                num_false_positives += 1
        request_time, deadline = self._finalize_request_time_and_deadline(deadline, predecessor_request_time, predecessor_deadline)

        scheduler.add_request(InferenceRequest(self._task, predecessor_request_id,
                                               input_array, request_time, deadline, batch_size, num_false_positives,
                                               self.max_dependency_path_length, self.is_deadline_shared))

    def accumulate_dropped_requests(self, predecessor_request_id : str) -> List[InferenceRequest]:
        batch_size : int = self._batch_sizes[int(self._rng.random() * len(self._batch_sizes))]
        #batch_size : int = self._rng.choice(self._batch_sizes)
        request = InferenceRequest(self._task, predecessor_request_id, {}, -1, -1, batch_size,
                                0, self.max_dependency_path_length, self.is_deadline_shared)
        out = [request]
        for dependency in self._task.dependencies:
            out.extend(dependency.accumulate_dropped_requests(request.request_id))
        return out
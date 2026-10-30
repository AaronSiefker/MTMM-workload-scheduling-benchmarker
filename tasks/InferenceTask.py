import threading
import time
from abc import abstractmethod, ABC
from typing import Final, Dict, Tuple, List, Callable

import numpy as np

from misc.StaticRng import StaticRng
from scheduling.InferenceRequest import InferenceRequest
from scheduling.Scheduler import Scheduler
from task_processing.TaskEngine import TaskEngine
from task_processing.cpu_execution.OnnxInferenceJob import OnnxInferenceJob, OnnxInferenceJobStatus
from task_processing.gpu_execution.TensorRtInferenceJob import TensorRtInferenceJob
from task_processing.input_preparation.ConfigurationInputConverter import MultidimensionalArray
from tasks.dependencies.Dependency import Dependency


class InferenceTask(ABC):
    def __init__(self, task : str, scheduler : Scheduler, relative_deadline : float,
                 dependencies : List[Dependency] | None):
        self.task : Final[str] = task
        self._scheduler : Final[Scheduler] = scheduler
        if dependencies is not None:
            self.dependencies : List[Dependency] = list(dependencies)
            self._max_dependency_path_length = max(d.max_dependency_path_length for d in dependencies) + 1
        else:
            self.dependencies : List[Dependency] = []
            self._max_dependency_path_length = 0
        self._is_deadline_shared : Final[bool] = any(dependency.is_deadline_shared for dependency in self.dependencies)

        self._inputs : List[MultidimensionalArray] = []
        self._relative_deadline : Final[float] = relative_deadline
        self._rng : StaticRng = StaticRng()

        self._request_id_counter : int = 0
        # Theoretically, generate_inference_request could be called by multiple threads from predecessor in
        # task graph. Probably not actually necessary, especially with my current setup, just to be safe
        self._lock : threading.Lock = threading.Lock()

        #self.diffs = []

    @abstractmethod
    def _generate_input_dictionary(self, input_array : MultidimensionalArray) -> Dict[str, MultidimensionalArray]:
        pass

    def generate_input(self, batch_size : int) -> Tuple[Dict[str, MultidimensionalArray], float]:
        # for some reason, this took several ms => replacing with StaticRng where all random numbers are precalculated at startup
        #chosen_inputs : MultidimensionalArray = self._rng.choice(self._inputs, size=batch_size)
        chosen_inputs = []
        num_inputs = len(self._inputs)
        for _ in range(batch_size):
            chosen_inputs.append(self._inputs[int(self._rng.random() * num_inputs)])
            #chosen_inputs.append(self._inputs[1])
        chosen_inputs =  np.array(chosen_inputs)
        input_dictionary = self._generate_input_dictionary(chosen_inputs)
        return input_dictionary, time.perf_counter() + self._relative_deadline

    # abstract, since task_engine isn't initialized during super.__init__() call yet and delaying that call is bad practice
    # as far as I know.
    @abstractmethod
    def get_task_engine(self) -> TaskEngine:
        pass

    @property
    def max_dependency_path_length(self) -> int:
        return self._max_dependency_path_length

    def generate_inference_request(self, predecessor_request_id : str | None = None, external_deadline : float | None = None) -> InferenceRequest:
        #print("pre gen:", time.perf_counter())
        input_array, deadline = self.generate_input(1)
        request_time = deadline - self._relative_deadline
        if external_deadline:
            #print("req:", request_time)
            deadline = external_deadline
            #request_time = time.perf_counter()
            request_time = deadline - self._relative_deadline # ideal request time for consistency in evaluation
            #print("external:", external_deadline)
            #self.diffs.append(deadline - request_time)
        return InferenceRequest(self, predecessor_request_id, input_array, request_time, deadline,1,
                                               0, self.max_dependency_path_length, self._is_deadline_shared)
        # self._scheduler.add_request(InferenceRequest(self, predecessor_request_id, input_array, request_time, deadline,1,
        #                                        0, self.max_dependency_path_length, self._is_deadline_shared))

    def finalize_gpu_post_processing(self, callback : Callable[[TensorRtInferenceJob], None]) \
            -> Callable[[TensorRtInferenceJob], None]:
        def func(job : TensorRtInferenceJob) -> None:
            callback(job)
            job.stream.synchronize()
            self._notify_dependencies(job)
        return func

    def finalize_cpu_post_processing(self, callback : Callable[[OnnxInferenceJob], None]) \
            -> Callable[[OnnxInferenceJob], None]:
        def func(job : OnnxInferenceJob) -> None:
            if job.status != OnnxInferenceJobStatus.FINISHED:
                return

            callback(job)
            self._notify_dependencies(job)
        return func

    def _notify_dependencies(self, job : OnnxInferenceJob | TensorRtInferenceJob):
        for dependency in self.dependencies:
            dependency.generate_inference_request(self._scheduler, job.request_id, job.request_time, job.deadline,
                                                  job.associated_config.false_positive_probability)

    id_separator : str = "|"
    def get_next_request_id(self) -> str:
        out = f"{self.task}{InferenceTask.id_separator}{self._request_id_counter}"
        with self._lock:
            self._request_id_counter += 1
        return out

    def reset_request_id_counter(self) -> None:
        with self._lock:
            self._request_id_counter = 0

    @staticmethod
    def get_task_name_from_request_id(request_id : str) -> str:
        return request_id.split(InferenceTask.id_separator)[0]
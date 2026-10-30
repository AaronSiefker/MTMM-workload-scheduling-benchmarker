from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import List, Tuple, Final

import numpy as np
import torch

from scheduling.InferenceRequest import InferenceRequest
from tasks.FaceDetection import FaceDetection
from tasks.GestureClassification import GestureClassification
from tasks.HandDetection import HandDetection
from tasks.ImageClassification import ImageClassification
from tasks.KeywordSpotting import KeywordSpotting

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scheduling.DefaultScheduler import DefaultScheduler


class AgentWrapper(ABC):
    @abstractmethod
    def generate_observation_vector(self, current_requests: List[InferenceRequest],
                                    prepared_requests: List[Tuple[InferenceRequest, float]], # second index is the time prep started
                                    ongoing_requests : List[Tuple[InferenceRequest, float] | None], # second index is the time inference started
                                    current_time : float) -> np.ndarray:
        pass

    @abstractmethod
    def generate_action_mask(self, current_observation : np.ndarray, is_preparation_action : bool, num_stream : int) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def num_streams(self) -> int:
        pass

    @property
    @abstractmethod
    def num_actions(self) -> int:
        pass

    @property
    @abstractmethod
    def observation_length(self) -> int:
        pass

    @abstractmethod
    def decode_preparation_action(self, action_idx: int, current_requests: List[InferenceRequest],
                                  prepared_requests: List[Tuple[InferenceRequest, float]]) -> Tuple[int, int]:
        # first returned index: index of request in current_requests; second returned index: gpu_config_index to use for request
        pass

    @abstractmethod
    def decode_execution_action(self, action_idx: int, stream_idx: int,
                                prepared_requests: List[Tuple[InferenceRequest, float]]) -> int:
        pass

    @staticmethod
    def prepare_request_default(current_requests: List[InferenceRequest], index: int, gpu_config_index: int,
                                prepared_requests: List[Tuple[InferenceRequest, float]],
                                scheduler: DefaultScheduler) -> None:
        if index < 0:
            return
        request = current_requests[index]
        current_time = time.perf_counter()
        gpu_config = request.task_engine.gpu_configurations[gpu_config_index]
        scheduler.begin_preparing_input_gpu(request, gpu_config)
        del current_requests[index]
        prepared_requests.append((request, current_time))

    @staticmethod
    def execute_request_default(prepared_requests : List[Tuple[InferenceRequest, float]], index : int,
                                stream : torch.cuda.Stream, stream_idx : int,
                                ongoing_requests : List[Tuple[InferenceRequest, float] | None],
                                scheduler : DefaultScheduler) -> None:
        if index < 0:
            return

        request, _ = prepared_requests[index]
        del prepared_requests[index]
        current_time = time.perf_counter()
        ongoing_requests[stream_idx] = (request, current_time)
        scheduler.begin_processing_prepared_request_gpu(request, stream, stream_idx)

    @staticmethod
    def select_next_stream_index(current_index : int, ongoing_requests : List[Tuple[InferenceRequest, float] | None]) -> int:
        # cycle through streams, i.e. do not always search from 0; might be useful if I do end up using imbalanced green contexts
        new_index = current_index
        max_index = len(ongoing_requests) - 1
        for _ in range(len(ongoing_requests)):
            new_index += 1
            if new_index == max_index + 1:
                new_index = 0
            if ongoing_requests[new_index] is None:
                return new_index
        return -1


class DefaultWorkloadAgentWrapper(AgentWrapper):
    _request_observation_length : Final[int] = 6
    _max_simultaneous_request : Final[int] = 7 #Face Detection + Hand Detection + Gesture Classification + KWS + 3x Image Classification
    _total_request_observation_length : Final[int] = _request_observation_length * _max_simultaneous_request

    _face_detection_prep_start_idx = 0
    _num_face_detection_num_configs = 5
    _hand_detection_prep_start_idx = _face_detection_prep_start_idx + _num_face_detection_num_configs
    _hand_detection_num_configs = 3
    _gesture_classification_prep_start_idx = _hand_detection_prep_start_idx + _hand_detection_num_configs
    _gesture_classification_num_configs = 3
    _kws_prep_start_idx = _gesture_classification_prep_start_idx + _gesture_classification_num_configs
    _kws_num_configs = 3
    _image_classification_1_start_idx = _kws_prep_start_idx + _kws_num_configs
    _image_classification_num_configs = 1
    _image_classification_2_start_idx = _image_classification_1_start_idx + _image_classification_num_configs
    _image_classification_3_start_idx = _image_classification_2_start_idx + _image_classification_num_configs
    _total_prep_action_count = _image_classification_3_start_idx + _image_classification_num_configs

    _prep_start_indexes = [_face_detection_prep_start_idx, _hand_detection_prep_start_idx, _gesture_classification_prep_start_idx,
                          _kws_prep_start_idx, _image_classification_1_start_idx, _image_classification_2_start_idx,
                          _image_classification_3_start_idx]
    _num_prep_actions = [_num_face_detection_num_configs, _hand_detection_num_configs, _gesture_classification_num_configs,
                         _kws_num_configs, _image_classification_num_configs, _image_classification_num_configs,
                         _image_classification_num_configs]

    _max_image_classification_requests = 3

    def __init__(self, num_streams : int):
        self._num_streams = num_streams # ongoing_requests tuple should be exactly as long as num_streams
        self._observation_length : Final[int] = (DefaultWorkloadAgentWrapper._total_request_observation_length
                                                 + num_streams  # indexes of currently executing tasks
                                                 + num_streams) # times since the respective executions started
        self._num_actions : Final[int] = (DefaultWorkloadAgentWrapper._total_prep_action_count
                                          + DefaultWorkloadAgentWrapper._max_simultaneous_request * num_streams  # execute prepared request on corresponding stream
                                          + 1  # do not prepare anything action
                                          + 1) # do not start execution for anything action

    @property
    def num_streams(self) -> int:
        return self._num_streams

    @property
    def num_actions(self) -> int:
        return self._num_actions

    @property
    def observation_length(self) -> int:
        return self._observation_length


    def generate_observation_vector(self, current_requests: List[InferenceRequest],
                                    prepared_requests: List[Tuple[InferenceRequest, float]], # second index is the time prep started
                                    ongoing_requests : List[Tuple[InferenceRequest, float] | None], # second index is the time inference started
                                    current_time : float) -> np.ndarray:
        observation = np.zeros(shape=self._observation_length, dtype=np.float32)

        num_seen_image_classification_requests : int = 0
        # prepared_requests first so that they are sufficient to decode correct image classification request during action index decoding
        for request, prep_start_time in prepared_requests:
            index = DefaultWorkloadAgentWrapper._convert_task_name_to_index(
                request.task.task, False, num_seen_image_classification_requests
            )
            if index >= DefaultWorkloadAgentWrapper._first_image_cls_idx:
                num_seen_image_classification_requests += 1

            start_index = index * DefaultWorkloadAgentWrapper._request_observation_length
            DefaultWorkloadAgentWrapper._fill_request_observation(
                request, observation, start_index, current_time, True, prep_start_time
            )
        for request in current_requests:
            index = DefaultWorkloadAgentWrapper._convert_task_name_to_index(
                request.task.task, False, num_seen_image_classification_requests
            )
            if index >= DefaultWorkloadAgentWrapper._first_image_cls_idx:
                num_seen_image_classification_requests += 1

            start_index = index * DefaultWorkloadAgentWrapper._request_observation_length
            DefaultWorkloadAgentWrapper._fill_request_observation(
                request, observation, start_index, current_time, False, 0.0
            )
        current_index = DefaultWorkloadAgentWrapper._total_request_observation_length
        for ongoing_request in ongoing_requests:
            if ongoing_request is None:
                continue
            request, inference_start_time = ongoing_request
            request_type_index = DefaultWorkloadAgentWrapper._convert_task_name_to_index(
                request.task.task, True, 0
            )
            observation[current_index] = request_type_index
            current_index += 1

            current_inference_time = current_time - inference_start_time
            observation[current_index] = current_inference_time
            current_index += 1

        return observation


    _first_image_cls_idx = 4
    @staticmethod
    def _convert_task_name_to_index(task_name : str, is_for_execution_info : bool,
                                    previous_obs_image_classification_requests : int) -> int:
        index : int = -1
        match task_name:
            case FaceDetection.task_name: index = 0
            case HandDetection.task_name: index = 1
            case GestureClassification.task_name: index = 2
            case KeywordSpotting.task_name: index = 3
            case ImageClassification.task_name: index = DefaultWorkloadAgentWrapper._first_image_cls_idx

        if is_for_execution_info:
            return index + 1

        if index == DefaultWorkloadAgentWrapper._first_image_cls_idx:
            # around rollout deadline delay, it can be possible that 4 exist at the same time
            index += min(previous_obs_image_classification_requests, DefaultWorkloadAgentWrapper._max_image_classification_requests - 1)

        return index

    # noinspection DuplicatedCode
    @staticmethod
    def _convert_preparation_action_index_to_task_name_and_config_index(action_index : int) -> Tuple[str, int, int]:
        if DefaultWorkloadAgentWrapper._face_detection_prep_start_idx \
            <= action_index < DefaultWorkloadAgentWrapper._hand_detection_prep_start_idx:
            gpu_config_index = action_index - DefaultWorkloadAgentWrapper._face_detection_prep_start_idx
            return FaceDetection.task_name, 0, gpu_config_index

        if DefaultWorkloadAgentWrapper._hand_detection_prep_start_idx \
            <= action_index < DefaultWorkloadAgentWrapper._gesture_classification_prep_start_idx:
            gpu_config_index = action_index - DefaultWorkloadAgentWrapper._hand_detection_prep_start_idx
            return HandDetection.task_name, 0, gpu_config_index

        if DefaultWorkloadAgentWrapper._gesture_classification_prep_start_idx \
            <= action_index < DefaultWorkloadAgentWrapper._kws_prep_start_idx:
            gpu_config_index = action_index - DefaultWorkloadAgentWrapper._gesture_classification_prep_start_idx
            return GestureClassification.task_name, 0, gpu_config_index

        if DefaultWorkloadAgentWrapper._kws_prep_start_idx \
            <= action_index < DefaultWorkloadAgentWrapper._image_classification_1_start_idx:
            gpu_config_index = action_index - DefaultWorkloadAgentWrapper._kws_prep_start_idx
            return KeywordSpotting.task_name, 0, gpu_config_index

        if DefaultWorkloadAgentWrapper._image_classification_1_start_idx \
            <= action_index < DefaultWorkloadAgentWrapper._image_classification_2_start_idx:
            gpu_config_index = action_index - DefaultWorkloadAgentWrapper._image_classification_1_start_idx
            return ImageClassification.task_name, 0, gpu_config_index

        if DefaultWorkloadAgentWrapper._image_classification_2_start_idx \
            <= action_index < DefaultWorkloadAgentWrapper._image_classification_3_start_idx:
            gpu_config_index = action_index - DefaultWorkloadAgentWrapper._image_classification_2_start_idx
            return ImageClassification.task_name, 1, gpu_config_index

        if DefaultWorkloadAgentWrapper._image_classification_3_start_idx \
            <= action_index < DefaultWorkloadAgentWrapper._total_prep_action_count:
            gpu_config_index = action_index - DefaultWorkloadAgentWrapper._image_classification_3_start_idx
            return ImageClassification.task_name, 2, gpu_config_index

        raise ValueError(f"Could not convert action_index '{action_index}' to task_name and gpu_config_index")

    @staticmethod
    def _convert_normalized_execution_action_index_to_task_name(action_index : int) -> Tuple[str, int]:
        task_name : str = ""

        if action_index >= DefaultWorkloadAgentWrapper._first_image_cls_idx:
            request_num = action_index - DefaultWorkloadAgentWrapper._first_image_cls_idx # select appropriate image classification request
            return ImageClassification.task_name, request_num

        match action_index:
            case 0: task_name = FaceDetection.task_name
            case 1: task_name = HandDetection.task_name
            case 2: task_name = GestureClassification.task_name
            case 3: task_name = KeywordSpotting.task_name
            case _: raise ValueError(f"could not convert action_index '{action_index}' to task_name")
        return task_name, 0

    @staticmethod
    def _fill_request_observation(request : InferenceRequest, observation : np.ndarray, start_index : int,
                                      current_time : float, is_prepared : bool, prep_start_time : float) -> None:
        current_index = start_index

        observation[current_index] = 1 # does exist
        current_index += 1

        observation[current_index] = 1 if is_prepared else 0.0 # is prepared
        current_index += 1

        observation[current_index] = (current_time - prep_start_time) if is_prepared else 0.0 # time since prep started
        current_index += 1

        observation[current_index] = request.max_dependency_path_length
        current_index += 1

        observation[current_index] = request.is_deadline_shared
        current_index += 1

        remaining_time = request.deadline - current_time
        observation[current_index] = remaining_time

    def generate_action_mask(self, current_observation: np.ndarray, is_preparation_action: bool, num_stream : int) -> np.ndarray:
        if is_preparation_action:
            return self._generate_preparation_action_mask(current_observation)
        else:
            return self._generate_execution_action_mask(current_observation, num_stream)

    def _generate_preparation_action_mask(self, current_observation : np.ndarray) -> np.ndarray:
        mask = np.zeros(self._num_actions, dtype=bool)
        for i in range(DefaultWorkloadAgentWrapper._max_simultaneous_request):
            can_be_prepared = DefaultWorkloadAgentWrapper._can_request_be_prepared(current_observation, i)
            current_start_idx = DefaultWorkloadAgentWrapper._prep_start_indexes[i]
            for prep_action_idx in range(DefaultWorkloadAgentWrapper._num_prep_actions[i]):
                mask[current_start_idx + prep_action_idx] = can_be_prepared
        mask[DefaultWorkloadAgentWrapper._total_prep_action_count] = True # Index to not prepare anything
        return mask

    @staticmethod
    def _can_request_be_prepared(current_observation : np.ndarray, request_index : int) -> bool:
        obs_index = request_index * DefaultWorkloadAgentWrapper._request_observation_length
        return current_observation[obs_index] and (not current_observation[obs_index + 1]) # first index: exists; second index: already prepared

    # decisions made for each stream one at a time, caller is responsible for selecting valid stream
    def _generate_execution_action_mask(self, current_observation : np.ndarray, stream_index : int) -> np.ndarray:
        mask = np.zeros(self._num_actions, dtype=bool)

        start_index = DefaultWorkloadAgentWrapper._total_prep_action_count + 1 # +1 to skip over index for action to not prepare anything

        for i in range(DefaultWorkloadAgentWrapper._max_simultaneous_request):
            can_be_executed = DefaultWorkloadAgentWrapper._can_request_be_executed(current_observation, i)
            stream_offset = stream_index * DefaultWorkloadAgentWrapper._max_simultaneous_request

            mask[start_index + i + stream_offset] = can_be_executed

        mask[self._num_actions - 1] = True # action index for not starting execution for any prepared task
        return mask

    @staticmethod
    def _can_request_be_executed(current_observation : np.ndarray, request_index : int) -> bool:
        obs_index = request_index * DefaultWorkloadAgentWrapper._request_observation_length
        return current_observation[obs_index] and current_observation[obs_index + 1] # first index: exists; second index: already prepared

    def decode_preparation_action(self, action_idx : int, current_requests : List[InferenceRequest],
                                  prepared_requests : List[Tuple[InferenceRequest, float]]) -> Tuple[int, int]:
            # prepared requests necessary to select correct image classification request in case there are multiple
            # first returned index: index of request in current_requests; second returned index: gpu_config_index to use for request
        if action_idx == DefaultWorkloadAgentWrapper._total_prep_action_count:
            return -1, 0 # Do nothing

        task_name, request_num, gpu_config_index = \
            DefaultWorkloadAgentWrapper._convert_preparation_action_index_to_task_name_and_config_index(action_idx)
        original_request_num = request_num
        if request_num > 0:
            request_num -= len([request for request, _ in prepared_requests if request.task.task == task_name])
        if request_num < 0:
            raise ValueError(f"Error attempting to prepare already prepared task or converted index to invalid request_num; "
                             f"action_index: {action_idx}; "
                             f"task_name: {task_name}; "
                             f"request_num: {original_request_num}")
        for i, request in enumerate(current_requests):
            if request.task.task == task_name:
                if request_num == 0:
                    return i, gpu_config_index
                request_num -= 1
        raise ValueError("Could not find an appropriate action for "
                         f"action_index: {action_idx}; "
                         f"task_name: {task_name}; "
                         f"in decode_preparation_action")

    def decode_execution_action(self, action_idx : int, stream_idx : int,
                                prepared_requests : List[Tuple[InferenceRequest, float]]) -> int:
        if action_idx == self._num_actions - 1:
            return -1 # do not start execution for any request

        start_index = DefaultWorkloadAgentWrapper._total_prep_action_count + 1  # +1 to skip over index for action to not prepare anything
        stream_offset = stream_idx * DefaultWorkloadAgentWrapper._max_simultaneous_request

        normalized_action_idx = action_idx - start_index - stream_offset
        task_name, request_num = DefaultWorkloadAgentWrapper._convert_normalized_execution_action_index_to_task_name(normalized_action_idx)

        for i, prepared_request in enumerate(prepared_requests):
            request, _ = prepared_request
            if request.task.task == task_name:
                if request_num == 0:
                    return i
                request_num -= 1
        raise ValueError("Could not find an appropriate action for "
                         f"action_index: {action_idx}; "
                         f"normalized_action_idx: {normalized_action_idx}; "
                         f"request_num: {request_num}; "
                         f"task_name: {task_name}; "
                         f"start_index: {start_index}; "
                         f"stream_offset: {stream_offset}; "
                         f"in decode_execution_action")


if __name__ == "__main__":
    test = DefaultWorkloadAgentWrapper(2)
    print("observation length:", test.observation_length)
    print("num actions:", test.num_actions)
    print(DefaultWorkloadAgentWrapper._total_prep_action_count)

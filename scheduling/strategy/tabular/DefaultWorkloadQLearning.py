from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import torch.cuda

import xrbench_helper
from scheduling.InferenceRequest import InferenceStats, InferenceRequest
from scheduling.Scheduler import SchedulingResult
from scheduling.strategy.tabular.QLearning import QLearningWrapper, QLearning
from tasks.FaceDetection import FaceDetection
from tasks.GestureClassification import GestureClassification
from tasks.HandDetection import HandDetection
from tasks.ImageClassification import ImageClassification
from tasks.KeywordSpotting import KeywordSpotting

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scheduling.DefaultScheduler import  DefaultScheduler


class DefaultWorkloadQLearning(QLearningWrapper):
    def __init__(self,
                 learning_rate : float = 0.1,
                 discount_factor : float = 0.7,
                 start_epsilon : float = 1.0,
                 min_epsilon : float = 0.08,
                 epsilon_decay : float = 0.999996, # roughly 650.000 steps until decayed to min_epsilon
                 is_learning : bool = True,
                 preparation_q_values : np.ndarray = None,
                 execution_q_values : np.ndarray = None):
        self._is_learning : bool = is_learning
        if preparation_q_values is None:
            preparation_q_values = DefaultWorkloadQLearning._create_preparation_agent_q_values()
        self._preparation_agent = QLearning(preparation_q_values, learning_rate=learning_rate,
                                            discount_factor=discount_factor, start_epsilon=start_epsilon,
                                            min_epsilon=min_epsilon, epsilon_decay=epsilon_decay)

        if execution_q_values is None:
            execution_q_values = DefaultWorkloadQLearning._create_execution_agent_q_values()
        self._execution_agent = QLearning(execution_q_values, learning_rate=learning_rate,
                                            discount_factor=discount_factor, start_epsilon=start_epsilon,
                                            min_epsilon=min_epsilon, epsilon_decay=epsilon_decay)

        # mapping request_ids to current_state, action_index and next_state
        self._preparation_transitions : Dict[str, Tuple[int, int, int]] = {}
        self._execution_transitions : Dict[str, Tuple[int, int, int]] = {}

        # kist keep the last 60 transitions around for energy_violations and clear whenever energy violated
        self._active_energy_preparation_transitions : List[Tuple[str, Tuple[int, int, int]]] = []
        self._active_energy_execution_transitions : List[Tuple[str, Tuple[int, int, int]]] = []

        self._total_episode_reward : float = 0.0
        self._predecessor_qualities : Dict[str, float] = {}

    @property
    def total_episode_reward(self) -> float:
        return self._total_episode_reward

    def take_preparation_action(self, scheduler : DefaultScheduler,
                                pending_requests: List[InferenceRequest],
                                prepared_requests: List[InferenceRequest],
                                any_stream_free : bool) -> None:
        current_state = DefaultWorkloadQLearning.compute_state_index(pending_requests, prepared_requests, any_stream_free)
        action_index = self._preparation_agent.get_action(current_state, deterministic=not self._is_learning)
        if action_index == 17:
            if self._is_learning:
                self._preparation_agent.update_q_values(current_state, action_index, 0, current_state)
            return
        if 0 <= action_index < 5:
            index, request = next((index, request) for index, request in enumerate(pending_requests)
                                                 if request.task.task == FaceDetection.task_name)
            gpu_config = request.task_engine.gpu_configurations[action_index]
        elif 5 <= action_index < 8:
            index, request =  next((index, request) for index, request in enumerate(pending_requests)
                                                 if request.task.task == HandDetection.task_name)
            gpu_config = request.task_engine.gpu_configurations[action_index - 5]
        elif 8 <= action_index < 11:
            index, request = next((index, request) for index, request in enumerate(pending_requests)
                                                        if request.task.task == GestureClassification.task_name)
            gpu_config = request.task_engine.gpu_configurations[action_index - 8]
        elif 11 <= action_index < 14:
            index, request = next((index, request) for index, request in enumerate(pending_requests)
                                      if request.task.task == KeywordSpotting.task_name)
            gpu_config = request.task_engine.gpu_configurations[action_index - 11]
        elif 14 <= action_index < 17:
            image_cls_request_num = action_index - 14
            num_prepared = len([1 for request in prepared_requests if request.task.task == ImageClassification.task_name])
            if num_prepared >= (image_cls_request_num + 1):
                print("prepared_requests:")
                for request in prepared_requests:
                    print(request.request_id)
                print("pending_requests:")
                for request in pending_requests:
                    print(request.request_id)
                raise ValueError(f"Error, attempting to prepare ImageClassification_{image_cls_request_num + 1}, "
                                 f"but there are already {num_prepared} prepared image classifications")
            pending_image_cls_request_num = image_cls_request_num - num_prepared
            image_cls_requests = [(index, request) for index, request in enumerate(pending_requests)
                                  if request.task.task == ImageClassification.task_name]
            index, request = image_cls_requests[pending_image_cls_request_num]
            gpu_config = request.task_engine.gpu_configurations[0]
        else:
            raise ValueError(f"Error, could not decode preparation action_index {action_index}")
        scheduler.begin_preparing_input_gpu(request, gpu_config)
        del pending_requests[index]
        prepared_requests.append(request)
        if self._is_learning:
            new_state = DefaultWorkloadQLearning.compute_state_index(pending_requests, prepared_requests, any_stream_free)
            self._preparation_transitions[request.request_id] = (current_state, action_index, new_state)
            self._active_energy_preparation_transitions.append((request.request_id, (current_state, action_index, new_state)))
            # delete outdated
            self._active_energy_preparation_transitions = self._active_energy_preparation_transitions[-60:]


    def take_execution_action(self, scheduler : DefaultScheduler,
                              pending_requests: List[InferenceRequest],
                              prepared_requests: List[InferenceRequest],
                              ongoing_requests : List[InferenceRequest | None],
                              cuda_streams : List[torch.cuda.Stream]) -> None:
        current_stream_index = -1
        for i in range(len(ongoing_requests)):
            if ongoing_requests[i] is None:
                current_stream_index = i
                break
        if current_stream_index < 0:
            raise ValueError("Error, take_execution_action was called while no stream is free, this is currently not the intended usage")
        current_state = DefaultWorkloadQLearning.compute_state_index(pending_requests, prepared_requests, current_stream_index >= 0)
        action_index = self._execution_agent.get_action(current_state, deterministic=not self._is_learning)
        if action_index == 7:
            if self._is_learning:
                self._execution_agent.update_q_values(current_state, action_index, 0, current_state)
            return
        if action_index == 0:
            index, request = next((index, request) for index, request in enumerate(prepared_requests)
                                                 if request.task.task == FaceDetection.task_name)
        elif action_index == 1:
            index, request = next((index, request) for index, request in enumerate(prepared_requests)
                                                 if request.task.task == HandDetection.task_name)
        elif action_index == 2:
            index, request = next((index, request) for index, request in enumerate(prepared_requests)
                                                 if request.task.task == GestureClassification.task_name)
        elif action_index == 3:
            index, request = next((index, request) for index, request in enumerate(prepared_requests)
                                                 if request.task.task == KeywordSpotting.task_name)
        elif 4 <= action_index < 7:
            image_cls_request_num = action_index - 4
            image_cls_requests = [(index, request) for index, request in enumerate(prepared_requests)
                                  if request.task.task == ImageClassification.task_name]
            index, request = image_cls_requests[image_cls_request_num]
        else:
            raise ValueError(f"Error, could not decode execution action_index {action_index}")
        scheduler.begin_processing_prepared_request_gpu(request, cuda_streams[current_stream_index], current_stream_index)
        del prepared_requests[index]
        ongoing_requests[current_stream_index] = request
        if self._is_learning:
            any_stream_free = any(ongoing_request is None for ongoing_request in ongoing_requests)
            new_state = DefaultWorkloadQLearning.compute_state_index(pending_requests, prepared_requests, any_stream_free)
            self._execution_transitions[request.request_id] = (current_state, action_index, new_state)
            self._active_energy_execution_transitions.append((request.request_id, (current_state, action_index, new_state)))
            # delete outdated
            self._active_energy_execution_transitions = self._active_energy_execution_transitions[-60:]

    def on_completed_request(self, stats: InferenceStats) -> None:
        if not self._is_learning:
            return
        preparation_transition = self._preparation_transitions.pop(stats.request.request_id, None)
        execution_transition = self._execution_transitions.pop(stats.request.request_id, None)
        if stats.finished_execution:
            reward = xrbench_helper.calculate_xrbench_per_inference_score_no_energy(SchedulingResult.create(stats), True)
            # propagated_reward (below) is useless, because two different prep actions lead to the same state
            # => then giving different rewards violates markov property

            # if stats.request.max_dependency_path_length > 0:
            #     self._predecessor_qualities[stats.request.request_id] = stats.task_config.quality
            # scheduling_result = SchedulingResult.create(stats)
            # predecessor_id = stats.request.predecessor_request_id
            # if predecessor_id is not None and predecessor_id in self._predecessor_qualities.keys():
            #     predecessor_quality = self._predecessor_qualities[predecessor_id]
            #     del self._predecessor_qualities[predecessor_id]
            #     scheduling_result = replace(scheduling_result, quality_score=predecessor_quality * stats.task_config.quality)
            # reward = xrbench_helper.calculate_xrbench_per_inference_score_no_energy(scheduling_result,True)
            #
            # # prevent dict from growing larger and larger due to unexecuted dependencies or probabilistic dependencies
            # if len(self._predecessor_qualities) > 500:
            #     # python dicts preserve insertion order, delete oldest 400, will most likely not overlap with ongoing requests
            #     self._predecessor_qualities = dict(list(self._predecessor_qualities.items())[-100:])
        else:
            reward = 0.0

        self._total_episode_reward += reward
        # protect against leftover executions from previous episode
        if preparation_transition is not None:
            self._preparation_agent.update_q_values(preparation_transition[0], preparation_transition[1], reward, preparation_transition[2])
        if execution_transition is not None:
            self._execution_agent.update_q_values(execution_transition[0], execution_transition[1], reward, execution_transition[2])

    def on_energy_violation(self, active_results : List[SchedulingResult]):
        if not self._is_learning:
            return
        # sorted_results_by_energy: List[SchedulingResult] = sorted(active_results, key=lambda r: r.consumed_energy,
        #                                                           reverse=True)
        # for result in sorted_results_by_energy:
        #     if ImageClassification.task_name in result.request_id:
        #         continue
        #     preparation_transition = next((request_id, transition)
        #                                   for (request_id, transition) in self._active_energy_preparation_transitions
        #                                   if request_id == result.request_id)[1]
        #     execution_transition = next((request_id, transition)
        #                                   for (request_id, transition) in self._active_energy_execution_transitions
        #                                   if request_id == result.request_id)[1]
        #     # heavily punish the one that consumed the most, except for Image Classification
        #     reward = -10
        #     self._preparation_agent.update_q_values(preparation_transition[0], preparation_transition[1], reward,
        #                                             preparation_transition[2])
        #     self._execution_agent.update_q_values(execution_transition[0], execution_transition[1], reward,
        #                                           execution_transition[2])
        #     self._total_episode_reward += reward
        #     return

        for result in active_results:
            preparation_transition = next((request_id, transition)
                                          for (request_id, transition) in self._active_energy_preparation_transitions
                                          if request_id == result.request_id)[1]
            execution_transition = next((request_id, transition)
                                          for (request_id, transition) in self._active_energy_execution_transitions
                                          if request_id == result.request_id)[1]
            # try light punishment first to avoid no_action being best accidentally
            # if ImageClassification.task_name in result.request_id:
            #     continue
            reward = -0.4
            self._preparation_agent.update_q_values(preparation_transition[0], preparation_transition[1], reward,
                                                    preparation_transition[2])
            # included, in case it is necessary to never execute a request in a given state, in case even the worst configuration still consumes too much energy.
            self._execution_agent.update_q_values(execution_transition[0], execution_transition[1], reward,
                                                  execution_transition[2])
            self._total_episode_reward += reward

    def on_begin_next_episode(self):
        # prevent accidental memory overflow; technically this shouldn't be able to grow infinitely, but better safe than sorry
        self._preparation_transitions.clear()
        self._execution_transitions.clear()

        self._active_energy_preparation_transitions.clear()
        self._active_energy_execution_transitions.clear()

        self._predecessor_qualities.clear()

        print("current time:", datetime.now().strftime("%H:%M:%S.%f")[:-3])
        print("Total episode reward:", self._total_episode_reward)
        self._total_episode_reward = 0

    def save(self, path : os.PathLike):
        base_path = Path(path)
        preparation_path = base_path / "preparation_agent.npy"
        execution_path = base_path / "execution_agent.npy"
        preparation_path.parent.mkdir(parents=True, exist_ok=True)

        self._preparation_agent.save_q_values(preparation_path)
        self._execution_agent.save_q_values(execution_path)

    def get_state_id(self, pending_requests : List[InferenceRequest],
                     prepared_requests : List[InferenceRequest],
                     any_stream_free : bool):
        return DefaultWorkloadQLearning.compute_state_index(pending_requests, prepared_requests, any_stream_free)

    def on_external_state_change(self, old_state : int, new_state : int):
        #return # temporarily disabled to train agents without this q-value propagation

        # simulate new state being reached via no_action to see if this helps agent during training
        if self._is_learning:
            # 17 is the no_action index for the preparation agent
            self._preparation_agent.update_q_values(old_state, 17, 0.0, new_state)
            # 7 is the no_action index for the preparation agent
            self._execution_agent.update_q_values(old_state, 7, 0.0, new_state)

    @staticmethod
    def load_q_value_arrays(path : os.PathLike) -> Tuple[np.ndarray, np.ndarray]:
        base_path = Path(path)
        preparation_path = base_path / "preparation_agent.npy"
        execution_path = base_path / "execution_agent.npy"

        preparation_q_values = np.load(preparation_path)
        execution_q_values = np.load(execution_path)
        return preparation_q_values, execution_q_values

    # 3 possibilities for request: doesn't exist (0), exists (1), prepared (2)
    _index_array_shape = (2, 3, 3, 3, 3, 3, 3, 3) # (streams free, face, hand, gesture, kws, image1, image2, image3)
    # the array essentially maps my indexes to the flattened index; incredibly inefficient w.r.t. memory, but prevents dumb mistake in manual conversion for now
    _index_array = np.arange(np.prod(_index_array_shape)).reshape(_index_array_shape)
    @staticmethod
    def compute_state_index(pending_requests: List[InferenceRequest], prepared_requests: List[InferenceRequest],
                            all_streams_free : bool) -> int:
        streams_free : int = 1 if all_streams_free else 0
        face_detection_idx = 0
        hand_detection_idx = 0
        gesture_classification_idx = 0
        kws_idx = 0
        image_classification_1_idx = 0
        image_classification_2_idx = 0
        image_classification_3_idx = 0

        seen_image_classifications = 0

        # prepared first, meaning I only need to search through prepared to find correct image classification request for execution
        for request in prepared_requests:
            match request.task.task:
                case FaceDetection.task_name:
                    if face_detection_idx != 0:
                        print(f"WARNING, overwriting already set face detection index {face_detection_idx} with 2")
                    face_detection_idx = 2
                case HandDetection.task_name:
                    if hand_detection_idx != 0:
                        print(f"WARNING, overwriting already set hand detection index {hand_detection_idx} with 2")
                    hand_detection_idx = 2
                case GestureClassification.task_name:
                    if gesture_classification_idx != 0:
                        print(f"WARNING, overwriting already set gesture classification index {gesture_classification_idx} with 2")
                    gesture_classification_idx = 2
                case KeywordSpotting.task_name:
                    if kws_idx != 0:
                        current_deadline = request.deadline
                        if kws_idx == 1:
                            other_deadline = next(request.deadline for request in pending_requests if request.task.task == KeywordSpotting.task_name)
                        else:
                            other_deadline = next(request.deadline for request in prepared_requests if request.task.task == KeywordSpotting.task_name)
                        print("deadline diff:", current_deadline - other_deadline)
                        print(f"WARNING, overwriting already set KWS index {kws_idx} with 2")
                    kws_idx = 2
                case ImageClassification.task_name:
                    if seen_image_classifications == 0:
                        if image_classification_1_idx != 0:
                            print(f"WARNING, overwriting already set image classification_1 index {image_classification_1_idx} with 2")
                        image_classification_1_idx = 2
                    elif seen_image_classifications == 1:
                        if image_classification_2_idx != 0:
                            print(f"WARNING, overwriting already set image classification_2 index {image_classification_2_idx} with 2")
                        image_classification_2_idx = 2
                    elif seen_image_classifications == 2:
                        if image_classification_3_idx != 0:
                            print(f"WARNING, overwriting already set image classification_3 index {image_classification_3_idx} with 2")
                        image_classification_3_idx = 2
                    else:
                        print("WARNING, encountered more than 3 image classification requests, not doing anything"
                              "since this should be exceptionally rare and can only happen in rare unintended edge cases")
                    seen_image_classifications += 1

        for request in pending_requests:
            match request.task.task:
                case FaceDetection.task_name:
                    if face_detection_idx != 0:
                        print(f"WARNING, overwriting already set face detection index {face_detection_idx} with 1")
                    face_detection_idx = 1
                case HandDetection.task_name:
                    if hand_detection_idx != 0:
                        print(f"WARNING, overwriting already set hand detection index {hand_detection_idx} with 1")
                    hand_detection_idx = 1
                case GestureClassification.task_name:
                    if gesture_classification_idx != 0:
                        print(f"WARNING, overwriting already set gesture classification index {gesture_classification_idx} with 1")
                    gesture_classification_idx = 1
                case KeywordSpotting.task_name:
                    if kws_idx != 0:
                        current_deadline = request.deadline
                        if kws_idx == 1:
                            other_deadline = next(request.deadline for request in pending_requests if request.task.task == KeywordSpotting.task_name)
                        else:
                            other_deadline = next(request.deadline for request in prepared_requests if request.task.task == KeywordSpotting.task_name)
                        print("deadline diff:", current_deadline - other_deadline)
                        print(f"WARNING, overwriting already set KWS index {kws_idx} with 1")
                    kws_idx = 1
                case ImageClassification.task_name:
                    if seen_image_classifications == 0:
                        if image_classification_1_idx != 0:
                            print(f"WARNING, overwriting already set image classification_1 index {image_classification_1_idx} with 1")
                        image_classification_1_idx = 1
                    elif seen_image_classifications == 1:
                        if image_classification_2_idx != 0:
                            print(f"WARNING, overwriting already set image classification_2 index {image_classification_2_idx} with 1")
                        image_classification_2_idx = 1
                    elif seen_image_classifications == 2:
                        if image_classification_3_idx != 0:
                            print(f"WARNING, overwriting already set image classification_3 index {image_classification_3_idx} with 1")
                        image_classification_3_idx = 1
                    else:
                        print("WARNING, encountered more than 3 image classification requests, not doing anything"
                              "since this should be exceptionally rare and can only happen in rare unintended edge cases")
                    seen_image_classifications += 1

        return DefaultWorkloadQLearning._index_array[streams_free][face_detection_idx][hand_detection_idx] \
            [gesture_classification_idx][kws_idx][image_classification_1_idx][image_classification_2_idx] \
            [image_classification_3_idx]

    # 5 face configs, 3 hand configs, 3 gesture configs, 3 kws configs, 3x image config, no action => 18 actions
    _preparation_q_values_shape = _index_array_shape + (18,)
    @staticmethod
    def _create_preparation_agent_q_values() -> np.ndarray:
        q_values = np.full(shape=DefaultWorkloadQLearning._preparation_q_values_shape, fill_value=QLearning.ACTION_MASK_VALUE, dtype=np.float32)
        q_values[:, 1, :, :, :, :, :, :, 0:5] = 0 # face detection configs
        q_values[:, :, 1, :, :, :, :, :, 5:8] = 0 # hand detection configs
        q_values[:, :, :, 1, :, :, :, :, 8:11] = 0 # gesture classification configs
        q_values[:, :, :, :, 1, :, :, :, 11:14] = 0 # kws configs
        q_values[:, :, :, :, :, 1, :, :, 14] = 0 # image classification 1 config
        q_values[:, :, :, :, :, :, 1, :, 15] = 0 # image classification 2 config
        q_values[:, :, :, :, :, :, :, 1, 16] = 0 # image classification 3 config
        q_values[:, :, :, :, :, :, :, :, 17] = 0  # no action is always possible

        # first -1: calculate axis size based on other axes, set second axis to size of last axis in q_values
        flattened_preparation_values = q_values.reshape(-1, q_values.shape[-1])
        return flattened_preparation_values

    # 7 possible executions, no action => 18 actions
    _execution_q_values_shape = _index_array_shape + (8,)
    @staticmethod
    def _create_execution_agent_q_values() -> np.ndarray:
        q_values = np.full(shape=DefaultWorkloadQLearning._execution_q_values_shape, fill_value=QLearning.ACTION_MASK_VALUE, dtype=np.float32)
        q_values[:, 2, :, :, :, :, :, :, 0] = 0 # face detection
        q_values[:, :, 2, :, :, :, :, :, 1] = 0 # hand detection
        q_values[:, :, :, 2, :, :, :, :, 2] = 0 # gesture classification
        q_values[:, :, :, :, 2, :, :, :, 3] = 0 # kws
        q_values[:, :, :, :, :, 2, :, :, 4] = 0 # image classification 1
        q_values[:, :, :, :, :, :, 2, :, 5] = 0 # image classification 2
        q_values[:, :, :, :, :, :, :, 2, 6] = 0 # image classification 3
        q_values[:, :, :, :, :, :, :, :, 7] = 0  # no action is always possible

        return q_values.reshape(-1, q_values.shape[-1])



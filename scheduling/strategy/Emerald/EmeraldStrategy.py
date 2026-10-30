from __future__ import annotations

import time
from enum import Enum, auto
from typing import List, Dict, Final, Tuple, TYPE_CHECKING

import torch

from scheduling.InferenceRequest import InferenceStats, InferenceRequest
from scheduling.strategy.Emerald.EmeraldHelper import get_emerald_inference_time
from scheduling.strategy.SchedulingStrategy import SchedulingStrategy
from task_processing.TaskEngine import TaskEngine
from tasks.FaceDetection import FaceDetection
from tasks.GestureClassification import GestureClassification
from tasks.HandDetection import HandDetection
from tasks.ImageClassification import ImageClassification
from tasks.InferenceTask import InferenceTask
from tasks.KeywordSpotting import KeywordSpotting

if TYPE_CHECKING:
    from scheduling.BenchmarkScheduler import BenchmarkScheduler

kws_1 = "KWS_1"
image_classification_1 = "Image Classification 1"
hand_detection = "Hand Detection"
image_classification_2 = "Image Classification 2"
gesture_classification = "Gesture Classification"
face_detection = "Face Detection"
kws_2 = "KWS_2"


class RequestStatus(Enum):
    NOT_YET_SEEN = auto()
    NOT_IN_FRAME = auto() # Image Classification 1 and 2 if they don't exist this frame

    PRESENT = auto()
    PREPARED = auto()
    EXECUTING = auto()
    FINISHED = auto()

# only creating this for frames which weren't dropped
class DefaultWorkloadFrame:
    def __init__(self, benchmark_scheduler : BenchmarkScheduler,
                 emerald_strategy : EmeraldStrategy,
                 stream_0_tasks : List[str], stream_0_configs : List[int],
                 stream_1_tasks : List[str], stream_1_configs : List[int],
                 is_mean_inference_times : bool,
                 previous_frame : DefaultWorkloadFrame | None
                 ):
        self._is_mean_inference_times : Final[bool] = is_mean_inference_times

        self._benchmark_scheduler : Final[BenchmarkScheduler] = benchmark_scheduler
        self._emerald_strategy : Final[EmeraldStrategy] = emerald_strategy
        self._previous_frame :DefaultWorkloadFrame | None = previous_frame

        self._stream_0_tasks : List[str] = stream_0_tasks
        self._stream_0_configs : List[int] = stream_0_configs
        self._next_stream_0_index : int = 0

        self._stream_1_tasks : List[str] = stream_1_tasks
        self._stream_1_configs : List[int] = stream_1_configs
        self._next_stream_1_index : int = 0

        self.is_frame_completed : bool = False

        self._face_detection : InferenceRequest | None = None
        self._face_detection_status : RequestStatus = RequestStatus.NOT_YET_SEEN

        self._hand_detection : InferenceRequest | None = None
        self._hand_detection_status : RequestStatus = RequestStatus.NOT_YET_SEEN

        self._gesture_classification : InferenceRequest | None = None
        self._gesture_classification_status : RequestStatus = RequestStatus.NOT_YET_SEEN

        self._kws_1 : InferenceRequest | None = None
        self._kws_1_status : RequestStatus = RequestStatus.NOT_YET_SEEN

        self._kws_2 : InferenceRequest | None = None
        self._kws_2_status : RequestStatus = RequestStatus.NOT_YET_SEEN

        self._image_classification_1 : InferenceRequest | None = None
        self._image_classification_1_status : RequestStatus = RequestStatus.NOT_YET_SEEN

        self._image_classification_2 : InferenceRequest | None = None
        self._image_classification_2_status : RequestStatus = RequestStatus.NOT_YET_SEEN

        if self._previous_frame is None:
            self._image_classification_2_status = RequestStatus.NOT_IN_FRAME

        self.start_time : float = -1.0

        self.is_own_predicted_stream_0_overhead : bool = False
        self.predicted_stream_0_overhead : float = 0.0 if self._previous_frame is None else self._previous_frame.predicted_stream_0_overhead
        self.has_started_processing_last_stream_0_request : bool = not (len(self._stream_0_tasks) > 0)
        final_stream_0_task_config_index = self._stream_0_configs[-1]
        final_stream_0_task = self._stream_0_tasks[-1]
        final_stream_0_task_engine = self._emerald_strategy.get_task_engine_for_request_type(final_stream_0_task)
        self._final_stream_0_inference_time = (
            get_emerald_inference_time(final_stream_0_task_engine.gpu_configurations[final_stream_0_task_config_index], self._is_mean_inference_times))

        self.is_own_predicted_stream_1_overhead : bool = False
        self.has_started_processing_last_stream_1_request: bool = True
        self.predicted_stream_1_overhead : float = 0.0 if self._previous_frame is None else self._previous_frame.predicted_stream_1_overhead
        self._final_stream_1_inference_time = 0.0
        if self._stream_1_tasks:
            final_stream_1_task_config_index = self._stream_1_configs[-1]
            final_stream_1_task = self._stream_1_tasks[-1]
            final_stream_1_task_engine = self._emerald_strategy.get_task_engine_for_request_type(final_stream_1_task)
            self._final_stream_1_inference_time = get_emerald_inference_time(
                final_stream_1_task_engine.gpu_configurations[final_stream_1_task_config_index], self._is_mean_inference_times)
            self.has_started_processing_last_stream_1_request = False

    def fill_with_pending(self, pending_requests : List[InferenceRequest]):
        self._update_requests(pending_requests)
        self._compute_is_frame_completed()

        # frame might be created before the previous frame calculated its predicted overhead => propagate later changes
        if not self.is_own_predicted_stream_0_overhead and self._previous_frame is not None:
            self.predicted_stream_0_overhead = self._previous_frame.predicted_stream_0_overhead
        if not self.is_own_predicted_stream_1_overhead and self._previous_frame is not None:
            self.predicted_stream_1_overhead = self._previous_frame.predicted_stream_1_overhead

    def _update_requests(self, pending_requests : List[InferenceRequest]):
        to_remove = []
        for request in pending_requests:
            if request.task.task == FaceDetection.task_name and self._face_detection_status == RequestStatus.NOT_YET_SEEN:
                self._face_detection = request
                self._face_detection_status = RequestStatus.PRESENT
                self.start_time = request.request_time
                to_remove.append(request)

            elif request.task.task == HandDetection.task_name and self._hand_detection_status == RequestStatus.NOT_YET_SEEN:
                self._hand_detection = request
                self._hand_detection_status = RequestStatus.PRESENT
                to_remove.append(request)

            elif request.task.task == GestureClassification.task_name and self._gesture_classification_status == RequestStatus.NOT_YET_SEEN:
                self._gesture_classification = request
                self._gesture_classification_status = RequestStatus.PRESENT
                to_remove.append(request)

            elif request.task.task == KeywordSpotting.task_name and self._kws_1_status == RequestStatus.NOT_YET_SEEN:
                self._kws_1 = request
                self._kws_1_status = RequestStatus.PRESENT
                to_remove.append(request)

            elif request.task.task == KeywordSpotting.task_name and self._kws_2_status == RequestStatus.NOT_YET_SEEN:
                self._kws_2 = request
                self._kws_2_status = RequestStatus.PRESENT
                to_remove.append(request)

            elif request.task.task == ImageClassification.task_name:
                if self._previous_frame and self._previous_frame._kws_2 is not None and \
                    (request.predecessor_request_id == self._previous_frame._kws_2.request_id):
                    self._image_classification_2 = request
                    self._image_classification_2_status = RequestStatus.PRESENT
                    to_remove.append(request)
                # self._kws_1 is not None if the status is FINISHED
                elif self._kws_1 is not None and request.predecessor_request_id == self._kws_1.request_id and \
                     (self._image_classification_1_status == RequestStatus.NOT_YET_SEEN):
                    self._image_classification_1 = request
                    self._image_classification_1_status = RequestStatus.PRESENT
                    to_remove.append(request)
                # else:
                #     raise ValueError(f"Error, encountered image classification request '{request.request_id}' while it was neither the frame start nor has "
                #                      "kws_1 already finished processing.")

        if self._previous_frame and self._previous_frame._kws_2_status == RequestStatus.FINISHED \
                and self._image_classification_2_status == RequestStatus.NOT_YET_SEEN:
            self._image_classification_2_status = RequestStatus.NOT_IN_FRAME

        if self._kws_1_status == RequestStatus.FINISHED and self._image_classification_1_status == RequestStatus.NOT_YET_SEEN:
            self._image_classification_1_status = RequestStatus.NOT_IN_FRAME

        pending_requests[:] = [request for request in pending_requests if request not in to_remove]

    def _compute_is_frame_completed(self):
        if self.is_frame_completed:
            return

        status = self._face_detection_status == RequestStatus.FINISHED
        status = status and self._hand_detection_status == RequestStatus.FINISHED
        status = status and self._gesture_classification_status == RequestStatus.FINISHED
        status = status and self._kws_1_status == RequestStatus.FINISHED
        status = status and self._kws_2_status == RequestStatus.FINISHED
        status = status and (self._image_classification_1_status == RequestStatus.FINISHED
                             or self._image_classification_1_status == RequestStatus.NOT_IN_FRAME)
        status = status and (self._image_classification_2_status == RequestStatus.FINISHED
                             or self._image_classification_2_status == RequestStatus.NOT_IN_FRAME)

        self.is_frame_completed = status

        # unassign previous frame: otherwise we will always have a reference to each frame if predecessor chain is never broken,
        # meaning they are never collected by the garbage collector, thereby eventually causing an oom error on the GPU
        if self.is_frame_completed:
            self._previous_frame = None

    def continue_processing(self, streams: Tuple[torch.cuda.Stream, ...],
                            ongoing_requests : List[InferenceRequest | None]):
        self._continue_processing_stream_0(streams, ongoing_requests)
        self._continue_processing_stream_1(streams, ongoing_requests)

    def _continue_processing_stream_0(self, streams: Tuple[torch.cuda.Stream, ...],
                                     ongoing_requests : List[InferenceRequest | None]):
        if self._next_stream_0_index < len(self._stream_0_tasks):
            current_task = self._stream_0_tasks[self._next_stream_0_index]
            current_config_index = self._stream_0_configs[self._next_stream_0_index]
            new_status = self._handle_task(current_task, current_config_index, 0,
                                           streams, ongoing_requests)
            if (new_status == RequestStatus.EXECUTING) or (new_status == RequestStatus.NOT_IN_FRAME):
                self._next_stream_0_index += 1
                self._continue_processing_stream_0(streams, ongoing_requests)

                if self._next_stream_0_index >= len(self._stream_0_tasks):
                    now = time.perf_counter()
                    self.predicted_stream_0_overhead = now + self._final_stream_0_inference_time - self.start_time
                    self.is_own_predicted_stream_0_overhead = True
                    self.has_started_processing_last_stream_0_request = True

    def _continue_processing_stream_1(self, streams: Tuple[torch.cuda.Stream, ...],
                                     ongoing_requests : List[InferenceRequest | None]):
        if self._next_stream_1_index < len(self._stream_1_tasks):
            current_task = self._stream_1_tasks[self._next_stream_1_index]
            current_config_index = self._stream_1_configs[self._next_stream_1_index]
            new_status = self._handle_task(current_task, current_config_index, 1,
                                           streams, ongoing_requests)
            if (new_status == RequestStatus.EXECUTING) or (new_status == RequestStatus.NOT_IN_FRAME):
                self._next_stream_1_index += 1
                self._continue_processing_stream_1(streams, ongoing_requests)

                if self._next_stream_1_index >= len(self._stream_1_tasks):
                    now = time.perf_counter()
                    self.predicted_stream_1_overhead = now + self._final_stream_1_inference_time - self.start_time
                    self.is_own_predicted_stream_1_overhead = True
                    self.has_started_processing_last_stream_1_request = True


    def _handle_task(self, task : str, config_index : int, stream_index : int,
                     streams: Tuple[torch.cuda.Stream, ...], ongoing_requests : List[InferenceRequest | None]) -> RequestStatus:
        if task == face_detection:
            new_status = self._handle_request(self._face_detection, self._face_detection_status, config_index,
                                              stream_index, streams, ongoing_requests)
            self._face_detection_status = new_status
            return new_status

        elif task == hand_detection:
            new_status = self._handle_request(self._hand_detection, self._hand_detection_status, config_index,
                                              stream_index, streams, ongoing_requests)
            self._hand_detection_status = new_status
            return new_status

        elif task == gesture_classification:
            new_status = self._handle_request(self._gesture_classification, self._gesture_classification_status, config_index,
                                              stream_index, streams, ongoing_requests)
            self._gesture_classification_status = new_status
            return new_status

        elif task == kws_1:
            new_status = self._handle_request(self._kws_1, self._kws_1_status, config_index,
                                              stream_index, streams, ongoing_requests)
            self._kws_1_status = new_status
            return new_status

        elif task == kws_2:
            new_status = self._handle_request(self._kws_2, self._kws_2_status, config_index,
                                              stream_index, streams, ongoing_requests)
            self._kws_2_status = new_status
            return new_status

        elif task == image_classification_1:
            new_status = self._handle_request(self._image_classification_1, self._image_classification_1_status, config_index,
                                              stream_index, streams, ongoing_requests)
            self._image_classification_1_status = new_status
            return new_status

        elif task == image_classification_2:
            new_status = self._handle_request(self._image_classification_2, self._image_classification_2_status, config_index,
                                              stream_index, streams, ongoing_requests)
            self._image_classification_2_status = new_status
            return new_status
        else:
            raise ValueError(f"Error, unknown request type '{task}'")

    def _handle_request(self, request : InferenceRequest | None, status : RequestStatus, config_index : int,
                        stream_index : int, streams: Tuple[torch.cuda.Stream, ...],
                        ongoing_requests : List[InferenceRequest | None]) -> RequestStatus:
        new_status = status
        if status == RequestStatus.PRESENT:
            if request is None:
                raise ValueError("Error, attempting to prepare a request while it is not present")
            self._benchmark_scheduler.begin_preparing_input_gpu(
                request, request.task_engine.gpu_configurations[config_index]
            )
            new_status = RequestStatus.PREPARED

        if new_status == RequestStatus.PREPARED and ongoing_requests[stream_index] is None:
            if request is None:
                raise ValueError("Error, attempting to execute a request while it is not present")
            self._benchmark_scheduler.begin_processing_prepared_request_gpu(
                request, streams[stream_index], stream_index
            )
            ongoing_requests[stream_index] = request
            new_status = RequestStatus.EXECUTING

        return new_status

    def notify_of_completion(self, stats: InferenceStats):
        finished_request = stats.request

        if self._face_detection is not None and self._face_detection.request_id == finished_request.request_id:
            self._face_detection_status = RequestStatus.FINISHED

        elif self._hand_detection is not None and self._hand_detection.request_id == finished_request.request_id:
            self._hand_detection_status = RequestStatus.FINISHED

        elif self._gesture_classification is not None and self._gesture_classification.request_id == finished_request.request_id:
            self._gesture_classification_status = RequestStatus.FINISHED

        elif self._kws_1 is not None and self._kws_1.request_id == finished_request.request_id:
            self._kws_1_status = RequestStatus.FINISHED

        elif self._kws_2 is not None and self._kws_2.request_id == finished_request.request_id:
            self._kws_2_status = RequestStatus.FINISHED

        elif self._image_classification_1 is not None and self._image_classification_1.request_id == finished_request.request_id:
            self._image_classification_1_status = RequestStatus.FINISHED

        elif self._image_classification_2 is not None and self._image_classification_2.request_id == finished_request.request_id:
            self._image_classification_2_status = RequestStatus.FINISHED
        # Since this called for every frame that currently exists, this safety check can unfortunately not exist
        # else:
        #     raise ValueError(f"Error, finished request '{finished_request.request_id}' did not belong to this frame.")

        if not self.is_frame_completed:
            self._compute_is_frame_completed()

    def handle_termination(self, current_time: float):
        self._handle_termination_for_request(self._face_detection, self._face_detection_status, current_time)
        self._handle_termination_for_request(self._hand_detection, self._hand_detection_status, current_time)
        self._handle_termination_for_request(self._gesture_classification, self._gesture_classification_status, current_time)
        self._handle_termination_for_request(self._kws_1, self._kws_1_status, current_time)
        self._handle_termination_for_request(self._kws_2, self._kws_2_status, current_time)
        self._handle_termination_for_request(self._image_classification_1, self._image_classification_1_status, current_time)
        self._handle_termination_for_request(self._image_classification_2, self._image_classification_2_status, current_time)

    def _handle_termination_for_request(self, request : InferenceRequest | None, status : RequestStatus, current_time : float):
        if status == RequestStatus.PREPARED:
            if request is None:
                raise ValueError("Error, attempting to drop prepared request which doesn't exist during termination")
            self._benchmark_scheduler.handle_dropped_request(request, current_time)
            self._benchmark_scheduler.cancel_request_gpu(request)


class EmeraldStrategy(SchedulingStrategy):
    def __init__(self, benchmark_scheduler : BenchmarkScheduler, num_streams : int, is_normal_periods :
    bool, is_for_98th_percentile : bool, is_mean_inference_times : bool):
        self._benchmark_scheduler : Final[BenchmarkScheduler] = benchmark_scheduler
        self._is_mean_inference_times : Final[bool] = is_mean_inference_times

        self._ongoing_requests : List[InferenceRequest | None] = [None for _ in range(num_streams)]

        self._num_streams : Final[int] = num_streams
        self._streams: Tuple[torch.cuda.Stream, ...] = tuple(torch.cuda.Stream() for _ in range(num_streams))

        # not needed, since I never cancel prepared requests except when terminating
        #self._canceled_request_ids : List[str] = []

        if is_for_98th_percentile:
            if is_normal_periods:
                self._period_length : float = 1.0 / 30
                if num_streams == 1:
                    self._stream_0_tasks : List[str] = [
                        kws_1, image_classification_1, hand_detection, image_classification_2, gesture_classification,
                        face_detection, kws_2
                    ]
                    self._stream_0_configs : List[int] = [0, 0, 0, 0, 0, 1, 0]
                    #self._stream_0_configs : List[int] = [2, 0, 2, 0, 2, 4, 2]

                    self._stream_1_tasks : List[str] = []
                    self._stream_1_configs : List[int] = []

                elif num_streams == 2:
                    self._stream_0_tasks : List[str] = [kws_1, face_detection, kws_2, image_classification_1]
                    self._stream_0_configs : List[int] = [0, 2, 0, 0]
                    #self._stream_0_configs : List[int] = [2, 4, 2, 0]

                    self._stream_1_tasks : List[str] = [hand_detection, image_classification_2, gesture_classification]
                    self._stream_1_configs : List[int] = [0, 0, 0]
                    #self._stream_1_configs : List[int] = [2, 0, 2]
                else:
                    raise ValueError("num_streams must be 1 or 2 for normal periods")
            else:
                self._period_length : float = 1.0 / 35
                if num_streams != 2:
                    raise ValueError("num_streams must be 2 for hard periods")
                else:
                    self._stream_0_tasks : List[str] = [kws_1, image_classification_1, hand_detection, gesture_classification]
                    self._stream_0_configs : List[int] = [1, 0, 1, 1]

                    self._stream_1_tasks : List[str] = [image_classification_2, face_detection, kws_2]
                    self._stream_1_configs : List[int] = [0, 2, 1]
        else:
            if is_normal_periods:
                self._period_length: float = 1.0 / 30
                if num_streams == 1:
                    self._stream_0_tasks: List[str] = [
                       kws_1, hand_detection, image_classification_1, gesture_classification, image_classification_2,
                        face_detection, kws_2
                    ]
                    self._stream_0_configs: List[int] = [0, 0, 0, 0, 0, 1, 0]
                    # self._stream_0_configs : List[int] = [2, 0, 2, 0, 2, 4, 2]

                    self._stream_1_tasks: List[str] = []
                    self._stream_1_configs: List[int] = []

                elif num_streams == 2:
                    self._stream_0_tasks: List[str] = [kws_1, image_classification_2, hand_detection,
                                                       image_classification_1, gesture_classification]
                    self._stream_0_configs: List[int] = [0, 0, 0, 0, 0]

                    self._stream_1_tasks: List[str] = [face_detection, kws_2]
                    self._stream_1_configs: List[int] = [0, 0]
                else:
                    raise ValueError("num_streams must be 1 or 2 for normal periods")
            else:
                self._period_length: float = 1.0 / 35
                if num_streams != 2:
                    raise ValueError("num_streams must be 2 for hard periods")
                else:
                    self._stream_0_tasks: List[str] = [kws_1, hand_detection, image_classification_2,
                                                       image_classification_1, gesture_classification]
                    self._stream_0_configs: List[int] = [0, 0, 0, 0, 0]

                    self._stream_1_tasks: List[str] = [face_detection, kws_2]
                    self._stream_1_configs: List[int] = [0, 0]

        self._predicted_stream_0_overhead : float = 0.0
        self._predicted_stream_1_overhead : float = 0.0

        self._next_frame_stream_0_configs : List[int] = self._stream_0_configs
        self._next_frame_stream_1_configs : List[int] = self._stream_1_configs

        self._prepared_requests : Dict[str, InferenceRequest] = {}

        self._task_engines_by_task_name : Dict[str, TaskEngine] = {}

        self._current_frames : List[DefaultWorkloadFrame] = []
        self._is_drop_next_kws : bool = False
        self._is_next_frame_no_predecessor : bool = False

        self._has_dropped_last_frame : bool = False

    def invoke(self, pending_requests: List[InferenceRequest]) -> None:
        now = time.perf_counter()
        if self._is_drop_next_kws:
            next_kws = next((request for request in pending_requests if request.task.task == KeywordSpotting.task_name), None)
            if next_kws is not None:
                self._benchmark_scheduler.handle_dropped_request(next_kws, now)
                pending_requests.remove(next_kws)
                self._is_drop_next_kws = False

        for frame in self._current_frames:
            frame.fill_with_pending(pending_requests)

        # new frame starts with face detection request
        if any((request.task.task == FaceDetection.task_name) for request in pending_requests):
            is_new_frame_dropped = self._execute_local_scheduler()
            if is_new_frame_dropped:
                # clear random image classifications which were never picked up since the corresponding frame was dropped
                # (frames always check if they belong in that frame)
                self._benchmark_scheduler.handle_dropped_requests(pending_requests, now)
                pending_requests.clear()
                self._is_drop_next_kws = True
                self._is_next_frame_no_predecessor = True
            else:
                # Since I am creating new frame after downscaling/upscaling, might as well just use the new configs,
                # as I can't start executions in parallel to the local scheduler anyway
                self._stream_0_configs = self._next_frame_stream_0_configs
                self._stream_1_configs = self._next_frame_stream_1_configs
                if self._is_next_frame_no_predecessor:
                    predecessor_frame = None
                    self._is_next_frame_no_predecessor = False
                else:
                    predecessor_frame = self._current_frames[-1] if self._current_frames else None
                new_frame = DefaultWorkloadFrame(self._benchmark_scheduler, self,
                                                 self._stream_0_tasks, self._stream_0_configs,
                                                 self._stream_1_tasks, self._stream_1_configs,
                                                 self._is_mean_inference_times,
                                                 predecessor_frame)
                self._current_frames.append(new_frame)
                new_frame.fill_with_pending(pending_requests)

        self.handle_current_frames()

    def handle_current_frames(self):
        if self._current_frames:
            frame = self._current_frames[0]
            frame.continue_processing(self._streams, self._ongoing_requests)

            # start preparing in next frame once previous frame has reached last request in one of its streams
            if frame.has_started_processing_last_stream_0_request and frame.has_started_processing_last_stream_1_request:
                for i, frame in enumerate(self._current_frames):
                    # we do not need to continue processing for first frame again, we have already called that above
                    if i == 0:
                        continue
                    if any((request is None) for request in self._ongoing_requests):
                        frame.continue_processing(self._streams, self._ongoing_requests)

        if self._current_frames:
            current_frame = self._current_frames[0]
            # remove completed frame from list while keeping the next "previous_frame"
            if current_frame.is_frame_completed and len(self._current_frames) > 1:
                del self._current_frames[0]


    # returns true if the frame should be dropped
    def _execute_local_scheduler(self) -> bool:
        self._next_frame_stream_0_configs = self._stream_0_configs.copy()
        self._next_frame_stream_1_configs = self._stream_1_configs.copy()

        if self._current_frames and not self._has_dropped_last_frame:
            self._predicted_stream_0_overhead = max(self._predicted_stream_0_overhead, self._current_frames[-1].predicted_stream_0_overhead)
            self._predicted_stream_1_overhead = max(self._predicted_stream_1_overhead, self._current_frames[-1].predicted_stream_1_overhead)

        self._has_dropped_last_frame = False
        drop_threshold = 3 * self._period_length
        if (self._predicted_stream_0_overhead > drop_threshold) or (self._predicted_stream_1_overhead > drop_threshold):
            #print("dropping frame")
            self._predicted_stream_0_overhead -= self._period_length
            self._predicted_stream_1_overhead -= self._period_length
            self._has_dropped_last_frame = True
            return True

        stream_0_ests, stream_1_ests = self._calculate_earliest_start_times()
        # current implementation assumes that stream 0 is used if only one stream is used
        stream_0_final_task_engine = self.get_task_engine_for_request_type(self._stream_0_tasks[-1])
        gpu_config_index = self._stream_0_configs[-1]
        expected_finish_time_stream_0 = (stream_0_ests[-1] +
                                         get_emerald_inference_time(stream_0_final_task_engine.gpu_configurations[gpu_config_index], self._is_mean_inference_times))
        #print("stream_0_ests:", stream_0_ests)
        #print("expected_finish_time_stream_0:", expected_finish_time_stream_0)

        expected_finish_time_stream_1 = 0.0
        if self._stream_1_tasks:
            stream_1_final_task_engine = self.get_task_engine_for_request_type(self._stream_1_tasks[-1])
            gpu_config_index = self._stream_1_configs[-1]
            expected_finish_time_stream_1 = (stream_1_ests[-1] +
                                             get_emerald_inference_time(stream_1_final_task_engine.gpu_configurations[gpu_config_index], self._is_mean_inference_times))
            #print("stream_1_ests:", stream_1_ests)
            #print("expected_finish_time_stream_1:", expected_finish_time_stream_1)

        if (expected_finish_time_stream_0 < 0.8 * self._period_length) and (expected_finish_time_stream_1 < 0.8 * self._period_length):
            #print("scaling up")
            stream_0_available_time = self._period_length - expected_finish_time_stream_0
            stream_1_available_time = self._period_length - expected_finish_time_stream_1

            self._next_frame_stream_0_configs = self._upscale_configs(stream_0_available_time, self._stream_0_tasks, self._stream_0_configs)
            self._next_frame_stream_1_configs = self._upscale_configs(stream_1_available_time, self._stream_1_tasks, self._stream_1_configs)

            self._predicted_stream_0_overhead = 0.0
            self._predicted_stream_1_overhead = 0.0

            return False

        stream_0_lfts, stream_1_lfts = self._calculate_latest_finish_times()

        stream_0_slacks = self._compute_slacks(stream_0_ests, stream_0_lfts, self._stream_0_tasks, self._stream_0_configs)
        stream_1_slacks = self._compute_slacks(stream_1_ests, stream_1_lfts, self._stream_1_tasks, self._stream_1_configs)

        # List[(slack, stream_index, task_index, TaskEngine, current_config_index)]
        negative_slacks = []
        for i, current_slack in enumerate(stream_0_slacks):
            if current_slack < 0:
                task_engine = self.get_task_engine_for_request_type(self._stream_0_tasks[i])
                current_config_index = self._stream_0_configs[i]

                negative_slacks.append((current_slack, 0, i, task_engine, current_config_index))

        for i, current_slack in enumerate(stream_1_slacks):
            if current_slack < 0:
                task_engine = self.get_task_engine_for_request_type(self._stream_1_tasks[i])
                current_config_index = self._stream_1_configs[i]

                negative_slacks.append((current_slack, 1, i, task_engine, current_config_index))

        # sort primarily by slack, stream_index and task_index serve as natural tiebreakers (not really necessary though)
        negative_slacks.sort()

        if len(negative_slacks) > 0:
            #print("scaling down")
            total_time_gained = self._downscale_configs(negative_slacks)

            self._predicted_stream_0_overhead = max(0.0, self._predicted_stream_0_overhead - total_time_gained)
            self._predicted_stream_1_overhead = max(0.0, self._predicted_stream_1_overhead - total_time_gained)
        return False

    # current implementation assumes that stream 0 is used if only one stream is used
    def _calculate_earliest_start_times(self) -> Tuple[List[float], List[float]]:
        # mapping request types to their expected finish time based on earliest start time
        processed_requests_for_ests : Dict[str, float] = {}
        stream_0_ests : List[float] = []
        stream_1_ests : List[float] = []

        current_stream = 0
        current_stream_0_index = 0
        current_stream_1_index = 0
        if self._current_frames and self._current_frames[-1].is_frame_completed:
            current_stream_0_time = 0.0
            current_stream_1_time = 0.0
        else:
            current_stream_0_time = self._predicted_stream_0_overhead
            current_stream_1_time = self._predicted_stream_1_overhead

        while True:
            if current_stream == 0:
                next_request_type : str = self._stream_0_tasks[current_stream_0_index]
                if ((next_request_type == image_classification_1) and (kws_1 not in processed_requests_for_ests.keys())) or \
                    ((next_request_type == gesture_classification) and (hand_detection not in processed_requests_for_ests.keys())):
                    current_stream = 1
                    continue
                if next_request_type == kws_2:
                    current_stream_0_time = max(self._period_length / 2, current_stream_0_time)
                elif next_request_type == image_classification_1:
                    current_stream_0_time = max(processed_requests_for_ests[kws_1], current_stream_0_time)
                elif next_request_type == gesture_classification:
                    current_stream_0_time = max(processed_requests_for_ests[hand_detection], current_stream_0_time)
                stream_0_ests.append(current_stream_0_time)

                task_engine = self.get_task_engine_for_request_type(next_request_type)
                gpu_config_index = self._stream_0_configs[current_stream_0_index]
                current_stream_0_time += get_emerald_inference_time(
                    task_engine.gpu_configurations[gpu_config_index], self._is_mean_inference_times)

                processed_requests_for_ests[next_request_type] = current_stream_0_time

                current_stream_0_index += 1
                if current_stream_0_index >= len(self._stream_0_tasks):
                    current_stream = 1
            else:
                next_request_type : str = self._stream_1_tasks[current_stream_1_index]
                if (next_request_type == image_classification_1 and (kws_1 not in processed_requests_for_ests.keys())) or \
                    (next_request_type == gesture_classification and (hand_detection not in processed_requests_for_ests.keys())):
                    current_stream = 0
                    continue

                if next_request_type == kws_2:
                    current_stream_1_time = max(self._period_length / 2, current_stream_1_time)
                elif next_request_type == image_classification_1:
                    current_stream_1_time = max(processed_requests_for_ests[kws_1], current_stream_1_time)
                elif next_request_type == gesture_classification:
                    current_stream_1_time = max(processed_requests_for_ests[hand_detection], current_stream_1_time)
                stream_1_ests.append(current_stream_1_time)

                task_engine = self.get_task_engine_for_request_type(next_request_type)
                gpu_config_index = self._stream_1_configs[current_stream_1_index]
                current_stream_1_time += get_emerald_inference_time(
                    task_engine.gpu_configurations[gpu_config_index], self._is_mean_inference_times)

                processed_requests_for_ests[next_request_type] = current_stream_1_time

                current_stream_1_index += 1
                if current_stream_1_index >= len(self._stream_1_tasks):
                    current_stream = 0

            if (current_stream_0_index >= len(self._stream_0_tasks)) and (current_stream_1_index >= len(self._stream_1_tasks)):
                break
        return stream_0_ests, stream_1_ests

    # current implementation assumes that stream 0 is used if only one stream is used
    def _calculate_latest_finish_times(self) -> Tuple[List[float], List[float]]:
        # mapping request types to their expected start time based on their latest finish time
        processed_requests_for_lfts: Dict[str, float] = {}
        stream_0_lfts: List[float] = []
        stream_1_lfts: List[float] = []

        current_stream = 0
        current_stream_0_index = len(self._stream_0_tasks) - 1
        current_stream_0_time = self._period_length
        current_stream_1_index = len(self._stream_1_tasks) - 1
        current_stream_1_time = self._period_length
        while True:
            if current_stream == 0:
                next_request_type: str = self._stream_0_tasks[current_stream_0_index]
                if ((next_request_type == kws_1) and (image_classification_1 not in processed_requests_for_lfts.keys())) or \
                        ((next_request_type == hand_detection) and (gesture_classification not in processed_requests_for_lfts.keys())):
                    current_stream = 1
                    continue

                # clamping latest finish times to start times of successors
                if next_request_type == kws_1:
                    current_stream_0_time = min(processed_requests_for_lfts[image_classification_1], current_stream_0_time)
                elif next_request_type == hand_detection:
                    current_stream_0_time = min(processed_requests_for_lfts[gesture_classification], current_stream_0_time)
                stream_0_lfts.append(current_stream_0_time)

                task_engine = self.get_task_engine_for_request_type(next_request_type)
                gpu_config_index = self._stream_0_configs[current_stream_0_index]
                current_stream_0_time -= get_emerald_inference_time(
                    task_engine.gpu_configurations[gpu_config_index], self._is_mean_inference_times)

                processed_requests_for_lfts[next_request_type] = current_stream_0_time

                current_stream_0_index -= 1
                if current_stream_0_index < 0:
                    current_stream = 1
            else:
                next_request_type: str = self._stream_1_tasks[current_stream_1_index]
                if ((next_request_type == kws_1) and (image_classification_1 not in processed_requests_for_lfts.keys())) or \
                        ((next_request_type == hand_detection) and (gesture_classification not in processed_requests_for_lfts.keys())):
                    current_stream = 0
                    continue
                # clamping latest finish times to start times of successors
                if next_request_type == kws_1:
                    current_stream_1_time = min(processed_requests_for_lfts[image_classification_1], current_stream_1_time)
                elif next_request_type == hand_detection:
                    current_stream_1_time = min(processed_requests_for_lfts[gesture_classification], current_stream_1_time)
                stream_1_lfts.append(current_stream_1_time)

                task_engine = self.get_task_engine_for_request_type(next_request_type)
                gpu_config_index = self._stream_1_configs[current_stream_1_index]
                current_stream_1_time -= get_emerald_inference_time(
                    task_engine.gpu_configurations[gpu_config_index], self._is_mean_inference_times)

                processed_requests_for_lfts[next_request_type] = current_stream_1_time

                current_stream_1_index -= 1
                if current_stream_1_index < 0:
                    current_stream = 0

            if (current_stream_0_index < 0) and (current_stream_1_index < 0):
                break

        return stream_0_lfts, stream_1_lfts

    def _compute_slacks(self, ests : List[float], lfts : List[float], tasks : List[str],
                        configs : List[int]) -> List[float]:
        slacks : List[float] = []

        for i in range(len(ests)):
            current_task = tasks[i]
            current_gpu_config_index = configs[i]
            est = ests[i]
            lft = lfts[i]

            task_engine = self.get_task_engine_for_request_type(current_task)
            processing_time = get_emerald_inference_time(
                task_engine.gpu_configurations[current_gpu_config_index], self._is_mean_inference_times)

            slacks.append(lft - est - processing_time)

        return slacks

    def _downscale_configs(self, negative_slacks) -> float:
        min_slack = negative_slacks[0][0]
        time_diff = -min_slack
        total_time_gained = 0.0

        all_possible_improvements: List[Tuple[float, int, int, TaskEngine, int, int]] = [
            (improvement, stream_index, task_index, task_engine, config_index, other_config_index)
            for _, stream_index, task_index, task_engine, config_index in negative_slacks
            for other_config_index in range(len(task_engine.gpu_configurations))
            if (improvement := get_emerald_inference_time(task_engine.gpu_configurations[config_index], self._is_mean_inference_times) -
                               get_emerald_inference_time(task_engine.gpu_configurations[other_config_index], self._is_mean_inference_times))
               > 0
        ]
        # TODO potentially sort in reverse order, might be slightly faster but most likely doesn't matter
        while all_possible_improvements:
            improvement, current_stream_index, current_task_index, _, current_config_idx, other_config_idx = max(all_possible_improvements, key=lambda t: t[0])
            if improvement >= time_diff:
                #  If we can select a config which overcomes the time_diff, select the best of all such configs
                min_improvement, current_stream_index, current_task_index, _, current_config_idx, other_config_idx = min(
                    (t for t in all_possible_improvements if t[0] >= time_diff),
                    key=lambda t: t[0]
                )
                if current_stream_index == 0:
                    self._next_frame_stream_0_configs[current_task_index] = other_config_idx
                    #print("scaled", self.get_task_engine_for_request_type(self._stream_0_tasks[current_task_index]).task, "from", current_config_idx, "to", other_config_idx)
                else:
                    self._next_frame_stream_1_configs[current_task_index] = other_config_idx
                    #print("scaled", self.get_task_engine_for_request_type(self._stream_1_tasks[current_task_index]).task,
                    #      "from", current_config_idx, "to", other_config_idx)
                total_time_gained += min_improvement
                break
            if current_stream_index == 0:
                self._next_frame_stream_0_configs[current_task_index] = other_config_idx
                #print("scaled", self.get_task_engine_for_request_type(self._stream_0_tasks[current_task_index]).task, "from",
                #      current_config_idx, "to", other_config_idx)
            else:
                self._next_frame_stream_1_configs[current_task_index] = other_config_idx
                #print("scaled", self.get_task_engine_for_request_type(self._stream_1_tasks[current_task_index]).task, "from",
                #      current_config_idx, "to", other_config_idx)
            total_time_gained += improvement
            time_diff -= improvement
            # Only update each request type once (identified via stream_index and task_index)
            all_possible_improvements = [
                (improvement, stream_index, task_index, task_engine, config_index, other_config_index) for \
                improvement, stream_index, task_index, task_engine, config_index, other_config_index in all_possible_improvements
                if stream_index != current_stream_index and task_index != current_task_index
            ]
        return total_time_gained

    # Since there is no concrete description how they are scaled up, I do it greedily per stream (called for each stream where possible)
    def _upscale_configs(self, available_time : float, tasks : List[str], config_indexes : List[int]) -> List[int]:
        # List[(quality_improvement, time_diff, task_index, other_config_index)]
        all_possible_improvements : List[Tuple[float, float, int, int]] = []
        for i, current_request_type in enumerate(tasks):
            task_engine = self.get_task_engine_for_request_type(current_request_type)
            current_config = task_engine.gpu_configurations[config_indexes[i]]
            for j, other_config in enumerate(task_engine.gpu_configurations):
                if other_config.task_configuration.quality > current_config.task_configuration.quality:
                    quality_improvement = other_config.task_configuration.quality - current_config.task_configuration.quality
                    time_diff = get_emerald_inference_time(other_config, self._is_mean_inference_times) - get_emerald_inference_time(current_config, self._is_mean_inference_times)

                    all_possible_improvements.append((quality_improvement, time_diff, i, j))

        all_possible_improvements.sort(reverse=True)

        new_config_indexes = config_indexes.copy()

        total_time_diff = 0.0

        while all_possible_improvements and total_time_diff + all_possible_improvements[0][1] <= available_time:
            _, time_diff, task_index, other_config_index = all_possible_improvements[0]
            del all_possible_improvements[0]
            total_time_diff += time_diff
            new_config_indexes[task_index] = other_config_index

        return new_config_indexes



    def get_task_engine_for_request_type(self, type_name : str) -> TaskEngine:
        if type_name == face_detection:
            return self._task_engines_by_task_name[FaceDetection.task_name]
        if type_name == hand_detection:
            return self._task_engines_by_task_name[HandDetection.task_name]
        if type_name == gesture_classification:
            return self._task_engines_by_task_name[GestureClassification.task_name]
        if type_name == kws_1:
            return self._task_engines_by_task_name[KeywordSpotting.task_name]
        if type_name == kws_2:
            return self._task_engines_by_task_name[KeywordSpotting.task_name]
        if type_name == image_classification_1:
            return self._task_engines_by_task_name[ImageClassification.task_name]
        if type_name == image_classification_2:
            return self._task_engines_by_task_name[ImageClassification.task_name]
        raise ValueError(f"Error, could not find appropriate task engine for '{type_name}'")


    def notify_of_completion(self, stats: InferenceStats) -> None:
        finished_request = stats.request
        for i,request in enumerate(self._ongoing_requests):
            if request is not None and request.request_id == finished_request.request_id:
                self._ongoing_requests[i] = None

        for frame in self._current_frames:
            frame.notify_of_completion(stats)

    def handle_termination(self, current_time: float) -> None:
        for frame in self._current_frames:
            frame.handle_termination(current_time)

    def warmup(self, tasks: List[InferenceTask], warmup_iterations, custom_batch_sizes: Dict[str, int]):
        self._task_engines_by_task_name.update({task.task: task.get_task_engine() for task in tasks})
        for i, stream in enumerate(self._streams):
            SchedulingStrategy.warmup_gpu(tasks, warmup_iterations, custom_batch_sizes, stream, i)

        for task in tasks:
            task.reset_request_id_counter()

    def get_num_cuda_streams(self) -> int:
        return self._num_streams
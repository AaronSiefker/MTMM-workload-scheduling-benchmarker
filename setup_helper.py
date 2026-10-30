from __future__ import annotations

import os
import sys
from typing import Final


def try_increase_priority(priority : int = 90) -> None:
    if sys.platform == "linux":
        try:
            # without this, threads are significantly delayed on Jetson, e.g. workload generator generated 3400 requests
            # over 3 minutes at 60 fps.
            # Not a great way to fix this for actual use cases, since I am starving all other processes like this. But
            # for my use case it is fine, especially since I couldn't fix it with spinning threads or multiprocessing
            # instead of multithreading.
            # priority goes from 1 (lowest) to 99 (highest) for Real-time FIFO scheduling
            param = os.sched_param(priority)
            os.sched_setscheduler(0, os.SCHED_FIFO, param)

            print("set up SCHED_FIFO")
        except PermissionError:
            print("insufficient permissions to adjust linux scheduling")


class DefaultWorkloadSettings:
    _default_hand_detection_period : float = 1.0 / 30
    _default_hand_detection_deadline : float = _default_hand_detection_period * 0.66
    def __init__(self,
                 used_strategy : str,
                 used_benchmark : str,
                 face_detection_period : float = 1.0 / 30,
                 hand_detection_period : float = _default_hand_detection_period,
                 is_hand_detection_deadline_shared : bool = False,

                 # in case it isn't shared, the deadline should be shorter than the period and the combined deadline should be equal to the period.
                 # if it is shared, just use period, since the dependency is always triggered and user would only be interested in "end result".
                 # Technically also the case for KWS, but there dependency is probabilistic (i.e. simulating user requests)
                 # and user may accept certain tasks taking longer than others
                 hand_detection_deadline : float = _default_hand_detection_deadline,
                 gesture_classification_deadline_offset : float | None = _default_hand_detection_period - _default_hand_detection_deadline,
                 keyword_spotting_period : float = 1.0 / 60,
                 is_keyword_spotting_deadline_shared : bool = False,

                 # always use instead of period for KWS since it has a probabilistic dependency, with shared
                 # dependency doesn't necessarily have to finish within the KWS period, same should be possible without shared
                 # However,kws_deadline should always be shorter than period if not shared
                 keyword_spotting_deadline : float = 1.0 / 60,
                 image_classification_deadline_offset : float = 1.0 / 30,
                 image_classification_probability : float = 0.25,
                 warmup_iterations : int = 5,
                 duration : float = 60 * 5,
                 gil_switch_interval : float = 0.002,

                 # both in Joule
                 energy_limit : float = 4 * 0.065,
                 xrbench_energy_max : float = 0.065
                 # effectively disabled
                 # energy_limit : float = 4 * 1000.065,
                 # xrbench_energy_max : float = 1000.065
                 ):
        self.used_strategy : Final[str] = used_strategy
        self.used_benchmark : Final[str] = used_benchmark

        self.face_detection_period : Final[float] = face_detection_period
        self.hand_detection_period : Final[float] = hand_detection_period
        self.is_hand_detection_deadline_shared : Final[bool] = is_hand_detection_deadline_shared
        self.hand_detection_deadline : Final[float] = hand_detection_deadline
        self.gesture_classification_deadline_offset : Final[float | None] = gesture_classification_deadline_offset
        self.keyword_spotting_period : Final[float] = keyword_spotting_period
        self.is_keyword_spotting_deadline_shared : Final[bool] = is_keyword_spotting_deadline_shared
        self.keyword_spotting_deadline : Final[float] = keyword_spotting_deadline
        self.image_classification_deadline_offset : Final[float | None] = image_classification_deadline_offset
        self.image_classification_probability : Final[float] = image_classification_probability
        self.warmup_iterations : Final[int] = warmup_iterations
        self.duration : Final[float] = duration

        self.energy_limit : Final[float] = energy_limit
        self.energy_interval_length : Final[float] = 4 * self.face_detection_period

        self.xrbench_energy_max : Final[float] = xrbench_energy_max

        self.gil_switch_interval : Final[float] = gil_switch_interval
        sys.setswitchinterval(gil_switch_interval)
        print("using gil switch interval:", sys.getswitchinterval())

    @staticmethod
    def create_with_default_periods_deadline_offsets(used_strategy : str, used_benchmark : str,
                                                     duration : int,
                                                     gil_switch_interval : float = 0.002) -> DefaultWorkloadSettings:
        return DefaultWorkloadSettings(used_strategy, used_benchmark, duration=duration, gil_switch_interval=gil_switch_interval)

    @staticmethod
    def create_with_hard_periods_deadline_offsets(used_strategy : str, used_benchmark : str,
                                                     duration : int,
                                                     gil_switch_interval : float = 0.002) -> DefaultWorkloadSettings:
        hand_detection_period: float = 1.0 / 35
        hand_detection_deadline: float = hand_detection_period * 0.66
        return DefaultWorkloadSettings(used_strategy, used_benchmark, duration=duration,
                                       face_detection_period=1.0 / 35,
                                       hand_detection_period=hand_detection_period,
                                       hand_detection_deadline=hand_detection_deadline,
                                       gesture_classification_deadline_offset=hand_detection_period - hand_detection_deadline,
                                       keyword_spotting_period=1.0 / 70,
                                       keyword_spotting_deadline=1.0 / 70,
                                       image_classification_deadline_offset=1.0 / 35,
                                       gil_switch_interval=gil_switch_interval)

    @staticmethod
    def create_with_hardest_periods_deadline_offsets(used_strategy : str, used_benchmark : str,
                                                     duration : int,
                                                     gil_switch_interval : float = 0.002) -> DefaultWorkloadSettings:
        hand_detection_period: float = 1.0 / 45
        hand_detection_deadline: float = hand_detection_period * 0.66
        return DefaultWorkloadSettings(used_strategy, used_benchmark, duration=duration,
                                       face_detection_period=1.0 / 45,
                                       hand_detection_period=hand_detection_period,
                                       hand_detection_deadline=hand_detection_deadline,
                                       gesture_classification_deadline_offset=hand_detection_period - hand_detection_deadline,
                                       keyword_spotting_period=1.0 / 90,
                                       keyword_spotting_deadline=1.0 / 90,
                                       image_classification_deadline_offset=1.0 / 45,
                                       gil_switch_interval=gil_switch_interval)

    @staticmethod
    def create_with_default_periods_shared_hand_det_deadline(used_strategy : str, used_benchmark : str,
                                                             duration : int,
                                                             gil_switch_interval : float = 0.002) -> DefaultWorkloadSettings:
        return DefaultWorkloadSettings(
            used_strategy, used_benchmark, duration=duration,
            hand_detection_deadline=DefaultWorkloadSettings._default_hand_detection_period,
            is_hand_detection_deadline_shared=True,
            gesture_classification_deadline_offset=0.0,
            gil_switch_interval=gil_switch_interval
        )

    @staticmethod
    def create_with_hard_periods_shared_hand_det_deadline(used_strategy : str, used_benchmark : str,
                                                     duration : int,
                                                     gil_switch_interval : float = 0.002) -> DefaultWorkloadSettings:
        hand_detection_period: float = 1.0 / 35
        return DefaultWorkloadSettings(used_strategy, used_benchmark, duration=duration,
                                       face_detection_period=1.0 / 35,
                                       hand_detection_period=hand_detection_period,
                                       hand_detection_deadline=hand_detection_period,
                                       is_hand_detection_deadline_shared=True,
                                       gesture_classification_deadline_offset=0.0,
                                       keyword_spotting_period=1.0 / 70,
                                       keyword_spotting_deadline=1.0 / 70,
                                       image_classification_deadline_offset=1.0 / 35,
                                       gil_switch_interval=gil_switch_interval)

    @staticmethod
    def create_with_hardest_periods_shared_hand_det_deadline(used_strategy : str, used_benchmark : str,
                                                     duration : int,
                                                     gil_switch_interval : float = 0.002) -> DefaultWorkloadSettings:
        hand_detection_period: float = 1.0 / 45
        return DefaultWorkloadSettings(used_strategy, used_benchmark, duration=duration,
                                       face_detection_period=1.0 / 45,
                                       hand_detection_period=hand_detection_period,
                                       hand_detection_deadline=hand_detection_period,
                                       is_hand_detection_deadline_shared=True,
                                       gesture_classification_deadline_offset=0.0,
                                       keyword_spotting_period=1.0 / 90,
                                       keyword_spotting_deadline=1.0 / 90,
                                       image_classification_deadline_offset=1.0 / 45,
                                       gil_switch_interval=gil_switch_interval)

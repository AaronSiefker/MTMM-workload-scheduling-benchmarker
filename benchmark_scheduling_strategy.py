import gc
import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from pprint import pprint
from statistics import mean
from time import sleep
from typing import Dict, Tuple

import torch.cuda

import setup_helper
from scheduling.AgentWrapper import DefaultWorkloadAgentWrapper
from scheduling.BenchmarkScheduler import BenchmarkScheduler
from scheduling.DefaultScheduler import DefaultScheduler
from scheduling.strategy.tabular.DefaultWorkloadQLearning import DefaultWorkloadQLearning
from setup_helper import DefaultWorkloadSettings
from tasks.FaceDetection import FaceDetection
from tasks.GestureClassification import GestureClassification
from tasks.HandDetection import HandDetection
from tasks.ImageClassification import ImageClassification
from tasks.KeywordSpotting import KeywordSpotting
from tasks.dependencies.ProbabilisticDependency import ProbabilisticDependency
from tasks.dependencies.RandomBatchSizeDependency import RandomBatchSizeDependency


def _default_workload_generation_deadline_scheduling(
        scheduler : DefaultScheduler, gpu_benchmark_results : Dict, cpu_benchmark_results : Dict,
        settings : DefaultWorkloadSettings,
        num_streams : int,
        reset_event : threading.Event | None, interrupt_event : threading.Event | None, continue_event : threading.Event | None
    ) -> Tuple[threading.Thread, threading.Event]:

    cancellation_event = threading.Event()

    def func():
        face_detection = FaceDetection(scheduler, settings.face_detection_period, gpu_benchmark_results, cpu_benchmark_results, num_streams)
        if settings.is_hand_detection_deadline_shared:
            hand_detection_deadline = settings.hand_detection_period
        else:
            hand_detection_deadline = settings.hand_detection_deadline

        gesture_classification = GestureClassification(scheduler, 0.0, gpu_benchmark_results, cpu_benchmark_results, num_streams)
        hand_detection = HandDetection(scheduler, hand_detection_deadline, gpu_benchmark_results, cpu_benchmark_results, num_streams,
                                       dependencies=[RandomBatchSizeDependency(
                                           gesture_classification, settings.is_hand_detection_deadline_shared,
                                           [1, 2, 3, 4],
                                           predecessor_deadline_offset=settings.gesture_classification_deadline_offset
                                       )]
                                       )

        image_classification = ImageClassification(scheduler, 0.0, gpu_benchmark_results, cpu_benchmark_results, num_streams)
        keyword_spotting = KeywordSpotting(scheduler, settings.keyword_spotting_deadline, gpu_benchmark_results,cpu_benchmark_results, num_streams,
                                           dependencies=[ProbabilisticDependency(
                                               image_classification, settings.is_keyword_spotting_deadline_shared,
                                               settings.image_classification_probability,
                                               predecessor_deadline_offset=settings.image_classification_deadline_offset
                                           )])
        tasks = [face_detection, hand_detection, gesture_classification, keyword_spotting, image_classification]
        custom_batch_sizes = {gesture_classification.task: 4}
        print("starting warmup; iterations:", settings.warmup_iterations)
        scheduler.warmup(tasks, settings.warmup_iterations, custom_batch_sizes)
        print("finished warmup")

        # wait a bit with input generation in order to avoid overload directly at beginning
        # WHEN ADJUSTING THIS, DO NOT FORGET TO ADJUST THE DUPLICATE IN RESET
        sleep(0.05)
        now = time.perf_counter()
        absolute_end = now + settings.duration
        # next_face_detection = now + settings.face_detection_period
        # next_hand_detection = now + settings.hand_detection_period
        # next_keyword_spotting = now + settings.keyword_spotting_period
        # starting all at the same time for EMERALD
        next_face_detection = now + settings.face_detection_period
        next_hand_detection = now + settings.face_detection_period
        next_keyword_spotting = now + settings.face_detection_period
        next_keyword_spotting_count = 0
        face_detection_count = 0
        hand_detection_count = 0
        while True:
            if interrupt_event is not None and interrupt_event.is_set():
                #print("interrupted workload generation")
                #print("next_keyword_spotting_count:", next_keyword_spotting_count)
                #print("face_detection_count:", face_detection_count)
                #print("hand_detection_count:", hand_detection_count)
                #print()
                #print()
                interrupt_time = time.perf_counter()
                if continue_event is None:
                    raise ValueError("Error, if interrupt_event is present, continue_event must also exist")
                continue_event.wait()
                #print()
                #print("continued workload generation")
                #print("next_keyword_spotting_count:", next_keyword_spotting_count)
                #print("face_detection_count:", face_detection_count)
                #print("hand_detection_count:", hand_detection_count)
                #print()
                #print()
                interrupt_event.clear()
                continue_event.clear()

                time_diff = time.perf_counter() - interrupt_time
                next_face_detection += time_diff
                next_hand_detection += time_diff
                next_keyword_spotting += time_diff
            now = time.perf_counter()
            #print("now:", now)
            if now >= absolute_end or cancellation_event.is_set():
                print("canceled workload generator")
                print("next_keyword_spotting_count:", next_keyword_spotting_count)
                print("face_detection_count:", face_detection_count)
                print("hand_detection_count:", hand_detection_count)

                #print("diffs")
                #for task in tasks:
                #    if len(task.diffs) == 0:
                #        continue
                #    print(task.task_name, ":", mean(task.diffs))
                return

            if reset_event is not None and reset_event.is_set():
                reset_event.clear()
                #print("reset workload generator")
                #print("next_keyword_spotting_count:", next_keyword_spotting_count)
                #print("face_detection_count:", face_detection_count)
                #print("hand_detection_count:", hand_detection_count)
                for task in tasks:
                    task.reset_request_id_counter()
                sleep(0.250) # provide time for garbage collection during training
                now = time.perf_counter()
                absolute_end = now + settings.duration
                next_face_detection = now + settings.face_detection_period
                next_hand_detection = now + settings.hand_detection_period
                next_keyword_spotting = now + settings.keyword_spotting_period
                next_keyword_spotting_count = 0
                face_detection_count = 0
                hand_detection_count = 0

            new_requests = []
            if now >= next_keyword_spotting:
                next_keyword_spotting_count += 1
                #print("pre:", time.perf_counter())
                new_requests.append(keyword_spotting.generate_inference_request(external_deadline=next_keyword_spotting + settings.keyword_spotting_period))
                #print("post:", time.perf_counter())
                # next_keyword_spotting = next_keyword_spotting + settings.keyword_spotting_period
                next_keyword_spotting = next_keyword_spotting + settings.keyword_spotting_period

            if now >= next_hand_detection:
                hand_detection_count += 1
                if settings.is_hand_detection_deadline_shared:
                    external_deadline = next_hand_detection + settings.hand_detection_period
                else:
                    external_deadline = next_hand_detection + settings.hand_detection_deadline
                new_requests.append(hand_detection.generate_inference_request(external_deadline=external_deadline))
                #next_hand_detection = next_hand_detection + settings.hand_detection_period
                next_hand_detection = next_hand_detection + settings.hand_detection_period

            if now >= next_face_detection:
                face_detection_count += 1
                new_requests.append(face_detection.generate_inference_request(external_deadline=next_face_detection + settings.face_detection_period))
                # like this (instead of += period), deadlines are a bit more generous, but protects against GIL induced delay
                # next_face_detection = next_face_detection + settings.face_detection_period
                next_face_detection = next_face_detection + settings.face_detection_period

            scheduler.add_requests(new_requests)
            next_wakeup = min(next_face_detection, next_hand_detection, next_keyword_spotting)
            sleep_duration = (next_wakeup - time.perf_counter())
            if sleep_duration > 0:
                time.sleep(sleep_duration)

    thread = threading.Thread(target=func)
    thread.start()
    return thread, cancellation_event


def start_benchmark_strategy_thread(strategy : BenchmarkScheduler.Strategy, result_file_path : str | os.PathLike,
                                    energy_limit : float, energy_interval_length : float, strategy_kwargs : Dict | None = None) \
        -> Tuple[threading.Thread, BenchmarkScheduler]:
    scheduler = BenchmarkScheduler(strategy, result_file_path, energy_limit, energy_interval_length, strategy_kwargs)
    scheduler_thread = threading.Thread(target=scheduler.activate)
    scheduler_thread.start()
    return scheduler_thread, scheduler

def main():
    setup_helper.try_increase_priority()

    benchmark_name = "./scheduling_results/EMERALD_Mean_1_Stream"
    print("benchmark_name:", benchmark_name)
    results_dict = Path(benchmark_name)
    results_file = results_dict / "results.json"
    settings_file = results_dict / "settings.json"

    exist_ok = True
    if (not exist_ok) and results_file.exists():
        raise ValueError(f"Error, a scheduling benchmark saved at path '{results_file} already exists, aborting")

    #used_gpu_benchmark = "benchmarking_results/fixed_freq/default-gpu_only-background_gpu_benchmark-SCHED_FIFO.json"
    used_gpu_benchmark = "benchmarking_results/fixed_freq/default-gpu_only-SCHED_FIFO.json"
    with open(used_gpu_benchmark) as f:
        gpu_benchmark_results = json.load(f)
    used_cpu_benchmark = "benchmarking_results/fixed_freq/default.json"
    with open(used_cpu_benchmark) as f:
        cpu_benchmark_results = json.load(f)

    used_strategy = BenchmarkScheduler.Strategy.EMERALD_MEAN_NORMAL_PERIODS_1_STREAM
    #cuda_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    #agent_wrapper = DefaultWorkloadAgentWrapper(2)
    #ppo_strategy_kwargs = {"agent_path": "rl_agents/default_workload_prep_then_action_online_2_streams_64_64_64.zip",
    #          "agent_wrapper": agent_wrapper, "cuda_streams": cuda_streams}

    # preparation_q_values, execution_q_values = DefaultWorkloadQLearning.load_q_value_arrays(
    #     Path("rl_agents_tabular/Default_Workload_hard_deadline_offsets_basic_reward_fake-no-action_2_Streams")
    # )
    # agent = DefaultWorkloadQLearning(
    #     is_learning=False, preparation_q_values=preparation_q_values, execution_q_values=execution_q_values
    # )
    # cuda_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    # tabular_strategy_kwargs = {
    #     "agent": agent, "is_learning": False, "episode_length": None, "cuda_streams": cuda_streams, "reset_workload_event": None
    # }

    settings = DefaultWorkloadSettings.create_with_default_periods_shared_hand_det_deadline(
        str(used_strategy), used_gpu_benchmark, 180
    )
    # settings = DefaultWorkloadSettings.create_with_default_periods_deadline_offsets(
    #     str(used_strategy), used_gpu_benchmark, 180
    # )
    # settings = DefaultWorkloadSettings.create_with_hard_periods_shared_hand_det_deadline(
    #     str(used_strategy), used_gpu_benchmark, 180
    # )
    # settings = DefaultWorkloadSettings.create_with_hard_periods_deadline_offsets(
    #     str(used_strategy), used_gpu_benchmark, 180
    # )
    # settings = DefaultWorkloadSettings.create_with_hardest_periods_shared_hand_det_deadline(
    #     str(used_strategy), used_gpu_benchmark, 180
    # )
    # settings = DefaultWorkloadSettings.create_with_hardest_periods_deadline_offsets(
    #     str(used_strategy), used_gpu_benchmark, 10
    # )
    print("settings:")
    pprint(settings.__dict__)
    print("-------------------------------")
    print("-------------------------------")

    gc.disable()
    scheduler_thread, scheduler = start_benchmark_strategy_thread(
        used_strategy, results_file, settings.energy_limit, settings.energy_interval_length,
        #tabular_strategy_kwargs
    )


    workload_thread, _ = _default_workload_generation_deadline_scheduling(
        scheduler, gpu_benchmark_results, cpu_benchmark_results, settings, scheduler.get_num_cuda_streams(),
        None, None, None)
    print("current time:", datetime.now().strftime("%H:%M:%S"))
    print("duration:", settings.duration, "seconds")
    #sleep(settings.duration)
    workload_thread.join()
    print("joined workload_thread")
    scheduler.terminate()
    gc.enable()
    scheduler_thread.join()
    print("joined scheduler_thread")

    with settings_file.open("w") as f:
        json.dump(settings.__dict__, f, indent=4)


if __name__ == "__main__":
    main()

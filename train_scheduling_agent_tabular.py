import gc
import json
import threading
from datetime import datetime
from pathlib import Path
from pprint import pprint
from time import sleep

import torch

import setup_helper
from benchmark_scheduling_strategy import start_benchmark_strategy_thread, \
    _default_workload_generation_deadline_scheduling
from scheduling.BenchmarkScheduler import BenchmarkScheduler
from scheduling.strategy.tabular.DefaultWorkloadQLearning import DefaultWorkloadQLearning
from setup_helper import DefaultWorkloadSettings

def main_2():
    setup_helper.try_increase_priority()

    # doesn't matter here
    used_gpu_benchmark = "benchmarking_results/fixed_freq/default-gpu_only-SCHED_FIFO.json"
    with open(used_gpu_benchmark) as f:
        gpu_benchmark_results = json.load(f)
    used_cpu_benchmark = "benchmarking_results/fixed_freq/default.json"
    with open(used_cpu_benchmark) as f:
        cpu_benchmark_results = json.load(f)

    # shared deadline doesn't really matter, since the tabular agent has no knowledge of deadlines anyway
    settings = DefaultWorkloadSettings.create_with_hard_periods_shared_hand_det_deadline(
        "TRAINING_RL_AGENT", used_gpu_benchmark, duration=1_000_000_000
    )
    print("settings:")
    pprint(settings.__dict__)
    print("-------------------------------")
    print("-------------------------------")

    cuda_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    agent = DefaultWorkloadQLearning(is_learning=True, discount_factor=0.7, min_epsilon=0.05)
    reset_event = threading.Event()

    agent_path = Path("rl_agents_tabular/energy_aware/DW_hp_energy_punishment_-0.4_all_eps=0.05_2_Streams")
    strategy_kwargs = {"agent": agent, "is_learning": True, "episode_length": 1.0 * 60 * 10,
                       "cuda_streams": cuda_streams, "reset_workload_event": reset_event, "intermediate_model_path": agent_path}
    gc.disable()
    scheduler_thread, scheduler = start_benchmark_strategy_thread(BenchmarkScheduler.Strategy.RL_AGENT_TABULAR_TRAINING,
                                                                  "TRAINING_TEMP",
                                                                  settings.energy_limit, settings.energy_interval_length,
                                                                  strategy_kwargs)

    workload_thread, cancellation_event = _default_workload_generation_deadline_scheduling(
        scheduler, gpu_benchmark_results, cpu_benchmark_results, settings, scheduler.get_num_cuda_streams(),
        reset_event, None, None)
    print("current time:", datetime.now().strftime("%H:%M:%S"))
    training_duration = 60 * 60 * 4
    print("training_duration:", training_duration)
    sleep(training_duration)
    cancellation_event.set()
    workload_thread.join()
    print("joined workload_thread")
    scheduler.terminate()
    gc.enable()
    scheduler_thread.join()
    print("joined scheduler_thread")

    agent.save(agent_path)
    print("saved agent at:", agent_path)



if __name__ == "__main__":
    # main()
    main_2()
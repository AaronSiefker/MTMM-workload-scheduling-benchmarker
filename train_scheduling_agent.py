import json
import multiprocessing
from multiprocessing import connection
import threading
from pprint import pprint
from statistics import mean

import torch
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

import setup_helper
from benchmark_scheduling_strategy import _default_workload_generation_deadline_scheduling
from setup_helper import DefaultWorkloadSettings
from scheduling.AgentTrainingEnvScheduler import AgentTrainingEnvScheduler, AgentTrainingEnv, AgentTrainingScheduler
from scheduling.AgentWrapper import DefaultWorkloadAgentWrapper


class SchedulerCallback(BaseCallback):
    def __init__(self, env: AgentTrainingEnvScheduler | AgentTrainingEnv):
        super().__init__()
        self._env = env

    def _on_rollout_start(self) -> None:
        self._env.on_rollout_start()

    def _on_rollout_end(self) -> None:
        self._env.on_rollout_end()

    def _on_training_end(self) -> None:
        self._env.on_training_end()

    def _on_step(self) -> bool:
        return True


# noinspection DuplicatedCode
def process_train_func(observation_length : int, num_actions : int,
                       pipe : multiprocessing.connection.Connection):
    env = AgentTrainingEnv(observation_length, num_actions, pipe)
    vec_env = DummyVecEnv([lambda: env])
    policy_kwargs = {"net_arch": [32], "activation_fn": nn.ReLU}
    callbacks = SchedulerCallback(env)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        device="cpu",
        policy_kwargs=policy_kwargs,
        n_steps=1000,
        verbose=0,
    )
    model.learn(
        total_timesteps=5000,
        callback=callbacks
    )
    model.save("rl_agents/testing_step_times")


# noinspection DuplicatedCode
def main_separate_process_training():
    setup_helper.try_increase_priority()

    used_gpu_benchmark = "benchmarking_results/fixed_freq/default-gpu_only-background_gpu_benchmark-SCHED_FIFO.json"
    with open(used_gpu_benchmark) as f:
        gpu_benchmark_results = json.load(f)
    used_cpu_benchmark = "benchmarking_results/fixed_freq/default.json"
    with open(used_cpu_benchmark) as f:
        cpu_benchmark_results = json.load(f)

    settings = DefaultWorkloadSettings.create_with_default_periods_shared_hand_det_deadline(
        "TRAINING_RL_AGENT", used_gpu_benchmark, duration=1_000_000_000
    )
    print("settings:")
    pprint(settings.__dict__)

    reset_event = threading.Event()
    interrupt_event = threading.Event()
    interrupt_event.set()
    continue_event = threading.Event()
    parent_pipe, child_pipe = multiprocessing.Pipe()

    cuda_streams = [torch.cuda.Stream()]
    print("num streams:", len(cuda_streams))
    agent_wrapper = DefaultWorkloadAgentWrapper(len(cuda_streams))
    scheduler_env = AgentTrainingScheduler(agent_wrapper, cuda_streams, 10 * 60, parent_pipe, reset_event, interrupt_event, continue_event)

    def this_process_func():
        scheduler_env.handle_agent_process_requests()
    thread = threading.Thread(target=this_process_func)
    thread.start()

    ctx = multiprocessing.get_context("spawn")
    # noinspection PyUnresolvedReferences
    p = ctx.Process(target=process_train_func, args=(agent_wrapper.observation_length, agent_wrapper.num_actions, child_pipe))
    p.start()
    workload_thread, cancellation_event = _default_workload_generation_deadline_scheduling(
        scheduler_env, gpu_benchmark_results, cpu_benchmark_results, settings, len(cuda_streams),
        reset_event, interrupt_event, continue_event
    )

    p.join()
    continue_event.set()
    cancellation_event.set()
    workload_thread.join()

    print("total step times:", len(scheduler_env.inbetween_step_times))
    print("sum step times:", sum(scheduler_env.inbetween_step_times))
    print("mean step time:", mean(scheduler_env.inbetween_step_times))


# noinspection DuplicatedCode
def main():
    setup_helper.try_increase_priority()


    used_gpu_benchmark = "benchmarking_results/fixed_freq/default-gpu_only-background_gpu_benchmark-SCHED_FIFO.json"
    with open(used_gpu_benchmark) as f:
        gpu_benchmark_results = json.load(f)
    used_cpu_benchmark = "benchmarking_results/fixed_freq/default.json"
    with open(used_cpu_benchmark) as f:
        cpu_benchmark_results = json.load(f)

    settings = DefaultWorkloadSettings.create_with_default_periods_shared_hand_det_deadline(
        "TRAINING_RL_AGENT", used_gpu_benchmark, duration=1_000_000_000
    )
    print("settings:")
    pprint(settings.__dict__)
    print("-------------------------------")
    print("-------------------------------")
    reset_event = threading.Event()
    interrupt_event = threading.Event()
    interrupt_event.set()
    continue_event = threading.Event()

    cuda_streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    print("num streams:", len(cuda_streams))
    agent_wrapper = DefaultWorkloadAgentWrapper(len(cuda_streams))
    env = AgentTrainingEnvScheduler(agent_wrapper, cuda_streams, 10 * 60, reset_event, interrupt_event, continue_event)
    vec_env = DummyVecEnv([lambda: env])

    workload_thread, cancellation_event = _default_workload_generation_deadline_scheduling(
        env, gpu_benchmark_results, cpu_benchmark_results, settings, len(cuda_streams),
        reset_event, interrupt_event, continue_event
    )
    policy_kwargs = {"net_arch": [64, 64, 64], "activation_fn": nn.ReLU}
    callbacks = SchedulerCallback(env)

    model = MaskablePPO(
        policy="MlpPolicy",
        env=vec_env,
        device="cpu",
        policy_kwargs=policy_kwargs,
        n_steps=1000,
        verbose=0,
    )
    model.learn(
        total_timesteps=2_000_000,
        callback=callbacks
    )
    continue_event.set()
    cancellation_event.set()
    workload_thread.join()

    #print("total step times:", len(env.inbetween_step_times))
    #print("sum step times:", sum(env.inbetween_step_times))
    #print("mean step time:", mean(env.inbetween_step_times))

    model.save("rl_agents/temp_test")

if __name__ == "__main__":
    #multiprocessing.set_start_method("spawn")
    main()
    # print()
    # print("with separate process")
    # print()
    # main_separate_process_training()
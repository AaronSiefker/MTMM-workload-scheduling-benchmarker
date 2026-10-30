import gc
import json
import random
import threading
from pathlib import Path
from pprint import pprint
from typing import List, Dict, TypeVar, Tuple

import numpy as np
import torch

import setup_helper
from scheduling.Scheduler import NullScheduler
from task_processing.CpuConfiguration import CpuConfiguration
from task_processing.GpuConfiguration import GpuConfiguration
from task_processing.TaskConfiguration import TaskConfiguration
from task_processing.input_preparation.ConfigurationInputConverter import MultidimensionalArray
from tasks.FaceDetection import FaceDetection
from tasks.GestureClassification import GestureClassification
from tasks.HandDetection import HandDetection
from tasks.ImageClassification import ImageClassification
from tasks.KeywordSpotting import KeywordSpotting

class DefaultWorkloadTaskContainer:
    def __init__(self, num_streams : int):
        with open("benchmarking_results/fixed_freq/default.json") as f:
            benchmark_results = json.load(f)

        self.null_scheduler = NullScheduler()
        self.face_detection = FaceDetection(self.null_scheduler, 0, benchmark_results, benchmark_results, num_streams)
        self.hand_detection = HandDetection(self.null_scheduler, 0, benchmark_results, benchmark_results, num_streams)
        self.gesture_classification = GestureClassification(self.null_scheduler, 0, benchmark_results, benchmark_results, num_streams)
        self.keyword_spotting = KeywordSpotting(self.null_scheduler, 0, benchmark_results, benchmark_results, num_streams)
        self.image_classification = ImageClassification(self.null_scheduler, 0, benchmark_results, benchmark_results, num_streams)

# cycles: maximum number of iterations before a configuration isn't picked anymore. Not every config is present in every iteration
def benchmark_default_workload(cycles : int, cycle_executions : int,
                               default_workload_task_container : DefaultWorkloadTaskContainer, stream_index : int,
                               benchmark_gpu : bool = True, benchmark_cpu : bool = True,
                               warmup_iterations : int = 20) \
        -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:

    face_detection = default_workload_task_container.face_detection
    hand_detection = default_workload_task_container.hand_detection
    gesture_classification = default_workload_task_container.gesture_classification
    keyword_spotting = default_workload_task_container.keyword_spotting
    image_classification = default_workload_task_container.image_classification
    tasks = [face_detection, hand_detection, gesture_classification, keyword_spotting, image_classification]

    preparation_results = {t.task: {} for t in tasks}
    inference_results = {t.task: {} for t in tasks}
    mapped_gpu_configs = {t.task: {} for t in tasks}
    mapped_cpu_configs = {t.task: {} for t in tasks}
    gpu_selection_counts = {t.task: {} for t in tasks}
    cpu_selection_counts = {t.task: {} for t in tasks}
    for t in tasks:
        configs_to_benchmark = []
        if benchmark_gpu:
            configs_to_benchmark += t.task_engine.gpu_configurations
        if benchmark_cpu:
            configs_to_benchmark += t.task_engine.cpu_configurations
        for config in configs_to_benchmark:
            max_batch_size = 1
            if config.task_configuration.task == gesture_classification.task:
                max_batch_size = 4
            for batch_size in range(1, max_batch_size + 1):
                config_id = _construct_config_id(config.task_configuration, batch_size)
                preparation_array = np.zeros((cycles, cycle_executions))
                inference_array = np.zeros((cycles, cycle_executions))
                preparation_results[t.task][config_id] = preparation_array
                inference_results[t.task][config_id] = inference_array
                if isinstance(config, GpuConfiguration):
                    mapped_gpu_configs[t.task][config_id] = config
                    gpu_selection_counts[t.task][config_id] = 0
                elif isinstance(config, CpuConfiguration):
                    mapped_cpu_configs[t.task][config_id] = config
                    cpu_selection_counts[t.task][config_id] = 0

    #context = torch.cuda.green_contexts.GreenContext.create(num_sms=4)
    # noinspection PyUnresolvedReferences
    #stream = context.Stream()
    stream = torch.cuda.Stream()
    def benchmark_iterations(processor_configs, mapped_processor_configs, current_config_ids, selection_counts):
        iteration = 1
        while len(processor_configs) > 2:
            gesture_classification_batch_size = 1
            if gesture_classification.task in current_config_ids:
                gesture_classification_batch_size = _get_batch_size_from_config_id(
                    current_config_ids[gesture_classification.task])
            inputs = {
                face_detection.task: [face_detection.generate_input(1)[0] for _ in range(cycle_executions)],
                hand_detection.task: [hand_detection.generate_input(1)[0] for _ in range(cycle_executions)],
                gesture_classification.task: [gesture_classification.generate_input(gesture_classification_batch_size)[0] for _ in range(cycle_executions)],
                keyword_spotting.task: [keyword_spotting.generate_input(1)[0] for _ in range(cycle_executions)],
                image_classification.task: [image_classification.generate_input(1)[0] for _ in range(cycle_executions)],
            }
            _benchmark_cycle(processor_configs, stream, stream_index, inputs, current_config_ids, preparation_results, inference_results,
                             selection_counts, num_executions=cycle_executions)
            processor_configs, current_config_ids = _randomize_cycle_selection(mapped_processor_configs, processor_configs, selection_counts, cycles)
            print("finished iteration:", iteration)
            iteration += 1

    if benchmark_gpu:
        print("benchmarking gpu")
        for i in range(warmup_iterations):
            for task in tasks:
                current_input, _ = task.generate_input(1)
                task.task_engine.active_gpu_configuration.execute(current_input, stream, stream_index=stream_index)
        gpu_configs, config_ids = _randomize_cycle_selection(mapped_gpu_configs, [], gpu_selection_counts, cycles)
        benchmark_iterations(gpu_configs, mapped_gpu_configs, config_ids, gpu_selection_counts)

    if benchmark_cpu:
        print("benchmarking cpu")
        for i in range(warmup_iterations):
            for task in tasks:
                current_input, _ = task.generate_input(1)
                task.task_engine.active_cpu_configuration.execute(current_input)
        cpu_configs, config_ids = _randomize_cycle_selection(mapped_cpu_configs, [], cpu_selection_counts, cycles)
        benchmark_iterations(cpu_configs, mapped_cpu_configs, config_ids, cpu_selection_counts)
    return preparation_results, inference_results

T = TypeVar("T", GpuConfiguration, CpuConfiguration)
def _randomize_cycle_selection(processor_configs : Dict[str, Dict[str, T]], last_cycle_configs : List[T],
                               selection_counts : Dict[str, Dict[str, int]], cycles : int) -> Tuple[List[T], Dict[str, str]]:
        configs = []
        config_ids = {}
        for task_name in processor_configs.keys():
            total_count = len(processor_configs[task_name])
            eligible = [config_id for config_id, selection_count in selection_counts[task_name].items() if selection_count < cycles]
            if total_count != 1:
                eligible = [config_id for config_id in eligible if processor_configs[task_name][config_id] not in last_cycle_configs]
            if task_name == "Gesture Classification" and total_count != 4:
                eligible = [config_id for config_id in eligible if processor_configs[task_name][config_id] not in last_cycle_configs]
            # in case we have more than one combination, but all others have already reached max cycles do not select,
            # since we are then no longer switching between configurations thereby falsifying switching cost.
            # If we only have one configuration, then we are not interested in the switching cost
            #
            # For Gesture Classification: 4 different batch sizes, so if there is just one GpuConfiguration left,
            # it is in eligible 4 times due to different possible batch sizes
            if eligible:
                config_id = random.choice(eligible)
                selection_counts[task_name][config_id] += 1
                configs.append(processor_configs[task_name][config_id])
                config_ids[task_name] = config_id
        return configs, config_ids


def _construct_config_id(task_config : TaskConfiguration, batch_size : int) -> str:
    return f"{task_config.model_id}@{task_config.model_variant}@batch_size={batch_size}"

def _get_batch_size_from_config_id(config_id : str) -> int:
    return int(config_id.split("=")[-1])

def _deconstruct_config_id(config_id : str) -> Tuple[str, str, int]:
    split_config_id = config_id.split("@")
    return split_config_id[0], split_config_id[1], _get_batch_size_from_config_id(config_id)


def _benchmark_cycle(configs : List[GpuConfiguration | CpuConfiguration], stream : torch.cuda.Stream, stream_index : int,
                     inputs : Dict[str, List[Dict[str, MultidimensionalArray]]], config_ids : Dict[str, str],
                     input_preparation_results : Dict[str, Dict[str, np.ndarray]], inference_results : Dict[str, Dict[str, np.ndarray]],
                     selection_counts : Dict[str, Dict[str, int]], num_executions : int = 5) -> None:
    for i in range(0, num_executions):
        random.shuffle(configs)
        for config in configs:
            current_input = inputs[config.task_configuration.task][i]
            if isinstance(config, GpuConfiguration):
                input_preparation_time, inference_time = config.execute(current_input, stream, stream_index=stream_index)
            else:
                input_preparation_time, inference_time = config.execute(current_input)
            task_name = config.task_configuration.task
            config_id = config_ids[config.task_configuration.task]
            cycle = selection_counts[config.task_configuration.task][config_id] - 1
            input_preparation_results[task_name][config_id][cycle][i] = input_preparation_time
            inference_results[task_name][config_id][cycle][i] = inference_time

def _remove_trailing_zeros(array : np.ndarray) -> np.ndarray:
    last_index = np.max(np.where(np.any(array != 0, axis=1)))
    return array[:last_index + 1]

def _check_results_existence(benchmark_name : str, results_directory : Path = None) -> bool:
    if results_directory is None:
        results_directory = Path("./benchmarking_results")
    json_path = results_directory / f"{benchmark_name}.json"
    return json_path.exists()

def save_benchmark_results(benchmark_name : str, benchmarking_results : Dict,
                           preparation_results : Dict, inference_results : Dict, exist_ok=False, results_directory : Path = None):
    # already checked in main, just to be extra safe here
    if (not exist_ok) and _check_results_existence(benchmark_name, results_directory=results_directory):
        raise ValueError(f"Error, a benchmark with name '{benchmark_name}' already exists. To overwrite, exist_ok must be true")

    if results_directory is None:
        results_directory = Path("./benchmarking_results")

    json_path = results_directory / f"{benchmark_name}.json"
    json_path.parent.mkdir(exist_ok=True)

    with open(json_path, "w") as f:
        json.dump(benchmarking_results, f, indent=4)

    concrete_results_directory = results_directory / benchmark_name
    concrete_results_directory.mkdir(exist_ok=exist_ok)

    with open(concrete_results_directory / "preparation_results.json", "w") as f:
        json.dump(preparation_results, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    with open(concrete_results_directory / "inference_results.json", "w") as f:
        json.dump(inference_results, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

def _start_background_inference(device : str) -> None:
    def func():
        null_scheduler = NullScheduler()
        with open("benchmarking_results/default.json") as f:
            benchmark_results = json.load(f)
        task = ImageClassification(null_scheduler, 0, benchmark_results, benchmark_results, 0)
        stream = torch.cuda.Stream()
        while True:
            dummy_input, _ = task.generate_input(1)
            if device == "cpu":
                task.task_engine.cpu_configurations[0].execute(dummy_input)
            elif device == "cuda":
                task.task_engine.gpu_configurations[0].execute(dummy_input, stream)
            else:
                raise ValueError(f"cannot perform background inference on '{device}', must be 'cpu' or 'cuda'")

    thread = threading.Thread(target=func, daemon=True)
    thread.start()
    print(f"started background work on device '{device}'")

def _start_background_gpu_benchmark(task_container : DefaultWorkloadTaskContainer, stream_index : int) -> None:
    def func():
        benchmark_default_workload(400, 5, task_container, stream_index, benchmark_cpu=False)

    thread = threading.Thread(target=func, daemon=True)
    thread.start()
    print("started background gpu benchmark")

def _start_background_cpu_benchmark(task_container : DefaultWorkloadTaskContainer) -> None:
    def func():
        benchmark_default_workload(200, 5, task_container, 0, benchmark_gpu=False)

    thread = threading.Thread(target=func, daemon=True)
    thread.start()
    print("started background cpu benchmark")

def create_benchmarking_results(preparation_results, inference_results):
    benchmarking_results = {}
    num_prep_times_greater_10ms = 0
    for task_name in inference_results.keys():
        if task_name not in benchmarking_results:
            benchmarking_results[task_name] = {}
        current = benchmarking_results[task_name]
        for config_id in inference_results[task_name].keys():
            reset_current = current

            model_id, model_variant, batch_size = _deconstruct_config_id(config_id)
            if model_id not in current:
                current[model_id] = {}
            current = current[model_id]
            if model_variant not in current:
                current[model_variant] = {}
            current = current[model_variant]
            if batch_size not in current:
                current[batch_size] = {}
            current = current[batch_size]
            preparation_times = _remove_trailing_zeros(preparation_results[task_name][config_id]).ravel()
            inference_times = _remove_trailing_zeros(inference_results[task_name][config_id])
            variant_first_inference_times = inference_times[:, 0]
            variant_inference_times = inference_times[:, 1:].ravel()
            for prep_time in preparation_times:
                if prep_time > 0.01:
                    num_prep_times_greater_10ms += 1
            current["mean_input_preparation_time"] = preparation_times.mean()
            current["80th_percentile_input_preparation_time"] = np.percentile(preparation_times, 80)
            current["90th_percentile_input_preparation_time"] = np.percentile(preparation_times, 90)
            current["99th_percentile_input_preparation_time"] = np.percentile(preparation_times, 99)
            current["worst_input_preparation_time"] = preparation_times.max()

            current["best_inference_time"] = variant_inference_times.min()
            current["mean_inference_time"] = variant_inference_times.mean()
            current["50th_percentile_inference_time"] = np.percentile(variant_inference_times, 50)
            current["80th_percentile_inference_time"] = np.percentile(variant_inference_times, 80)
            current["90th_percentile_inference_time"] = np.percentile(variant_inference_times, 90)
            current["98th_percentile_inference_time"] = np.percentile(variant_inference_times, 98)
            current["99th_percentile_inference_time"] = np.percentile(variant_inference_times, 99)
            current["worst_inference_time"] = variant_inference_times.max()

            mean_overhead = variant_first_inference_times.mean() - current["mean_inference_time"]
            if mean_overhead < 0.001:
                mean_overhead = 0
            current["variant_switching_cost"] = mean_overhead
            current["num_inferences"] = inference_times.size

            current = reset_current
    print("num_prep_times_greater_10ms:", num_prep_times_greater_10ms)
    return benchmarking_results

def main():
    setup_helper.try_increase_priority()

    benchmark_name = "fixed_freq/temp-test"
    print("benchmark_name:", benchmark_name)
    # for quick adjustments in case I do actually want to overwrite
    exist_ok = True
    if (not exist_ok) and _check_results_existence(benchmark_name):
        raise ValueError(f"Error, a benchmark with name '{benchmark_name}' already exists. To overwrite, exist_ok must be true")

    num_streams = 1
    task_container = DefaultWorkloadTaskContainer(num_streams)
    #_start_background_inference("cuda")
    #_start_background_gpu_benchmark(task_container, 1)
    #_start_background_cpu_benchmark(task_container)

    gc.disable()
    preparation_results, inference_results = benchmark_default_workload(200, 5, task_container, 0, benchmark_cpu=False)
    gc.enable()
    #pprint(preparation_results)
    benchmarking_results = create_benchmarking_results(preparation_results, inference_results)

    pprint(benchmarking_results)
    save_benchmark_results(benchmark_name, benchmarking_results, preparation_results, inference_results, exist_ok)
    print(f"saved results with benchmark_name: '{benchmark_name}'")



if __name__ == "__main__":
    main()


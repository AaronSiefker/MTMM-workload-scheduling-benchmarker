import json

import torch

from scheduling.Scheduler import NullScheduler
from tasks.GestureClassification import GestureClassification
from tasks.FaceDetection import FaceDetection
from tasks.HandDetection import HandDetection
from tasks.ImageClassification import ImageClassification
from tasks.KeywordSpotting import KeywordSpotting
from tasks.dependencies.ProbabilisticDependency import ProbabilisticDependency
from tasks.dependencies.RandomBatchSizeDependency import RandomBatchSizeDependency

if __name__ == "__main__":
    with open("benchmarking_results/default.json") as f:
        benchmark_results = json.load(f)

    print("Test")
    null_scheduler = NullScheduler()
    test = FaceDetection(null_scheduler, 0, benchmark_results, benchmark_results, 1)
    dummy_input, _ = test.generate_input(1)
    test.task_engine.gpu_configurations[4].execute(dummy_input, torch.cuda.Stream())
    test.task_engine.cpu_configurations[4].execute(dummy_input)
    test.task_engine.gpu_configurations[3].execute(dummy_input, torch.cuda.Stream())
    test.task_engine.cpu_configurations[3].execute(dummy_input)
    test.task_engine.gpu_configurations[2].execute(dummy_input, torch.cuda.Stream())
    test.task_engine.cpu_configurations[2].execute(dummy_input)
    test.task_engine.gpu_configurations[1].execute(dummy_input, torch.cuda.Stream())
    test.task_engine.cpu_configurations[1].execute(dummy_input)
    test.task_engine.gpu_configurations[0].execute(dummy_input, torch.cuda.Stream())
    test.task_engine.cpu_configurations[0].execute(dummy_input)
    test = HandDetection(null_scheduler, 0, benchmark_results, benchmark_results, 1)
    dummy_input, _ = test.generate_input(1)
    test.task_engine.gpu_configurations[2].execute(dummy_input, torch.cuda.Stream())
    test.task_engine.cpu_configurations[2].execute(dummy_input)
    test.task_engine.gpu_configurations[1].execute(dummy_input, torch.cuda.Stream())
    test.task_engine.cpu_configurations[1].execute(dummy_input)
    test.task_engine.gpu_configurations[0].execute(dummy_input, torch.cuda.Stream())
    test.task_engine.cpu_configurations[0].execute(dummy_input)
    test = GestureClassification(null_scheduler, 0, benchmark_results, benchmark_results, 1)
    dummy_input, _ = test.generate_input(1)
    test.task_engine.gpu_configurations[2].execute(dummy_input, torch.cuda.Stream())
    test.task_engine.cpu_configurations[2].execute(dummy_input)
    test.task_engine.gpu_configurations[1].execute(dummy_input, torch.cuda.Stream())
    test.task_engine.cpu_configurations[1].execute(dummy_input)
    test.task_engine.gpu_configurations[0].execute(dummy_input, torch.cuda.Stream())
    test.task_engine.cpu_configurations[0].execute(dummy_input)
    test = ImageClassification(null_scheduler, 0, benchmark_results, benchmark_results, 1)
    dummy_input, _ = test.generate_input(1)
    test.task_engine.gpu_configurations[0].execute(dummy_input, torch.cuda.Stream())
    test.task_engine.cpu_configurations[0].execute(dummy_input)
    test = KeywordSpotting(null_scheduler, 0, benchmark_results, benchmark_results, 1)
    dummy_input, _ = test.generate_input(1)
    test.task_engine.gpu_configurations[0].execute(dummy_input, torch.cuda.Stream())
    test.task_engine.cpu_configurations[0].execute(dummy_input)
    test.task_engine.gpu_configurations[1].execute(dummy_input, torch.cuda.Stream())
    test.task_engine.cpu_configurations[1].execute(dummy_input)
    test.task_engine.gpu_configurations[2].execute(dummy_input, torch.cuda.Stream())
    test.task_engine.cpu_configurations[2].execute(dummy_input)

    test1 = FaceDetection(null_scheduler, 0, benchmark_results, benchmark_results, 1)
    test2 = GestureClassification(null_scheduler, 0, benchmark_results, benchmark_results, 0, [ProbabilisticDependency(test1, True, 1)])
    print("test2._is_deadline_shared:", test2._is_deadline_shared)
    test3 = HandDetection(null_scheduler, 0, benchmark_results, benchmark_results, 0, [RandomBatchSizeDependency(test1, False, [1,2,3,4])])
    print("test3._is_deadline_shared:", test3._is_deadline_shared)
    test4 = KeywordSpotting(null_scheduler, 0, benchmark_results, benchmark_results, 0, [RandomBatchSizeDependency(test2, False, [1,2,3,4])])
    test5 = ImageClassification(null_scheduler, 0, benchmark_results, benchmark_results, 0, [
        ProbabilisticDependency(test3, False, 1),
        RandomBatchSizeDependency(test1, False,[1, 2, 3, 4])
    ])
    print("test5._is_deadline_shared:", test5._is_deadline_shared)

    test6 = ImageClassification(null_scheduler, 0, benchmark_results, benchmark_results, 0, [
        ProbabilisticDependency(test5, False, 1),
        RandomBatchSizeDependency(test1, True, [1, 2, 3, 4])
    ])
    print("test6._is_deadline_shared:", test6._is_deadline_shared)

    print(test1.max_dependency_path_length)
    print(test2.max_dependency_path_length)
    print(test3.max_dependency_path_length)
    print(test4.max_dependency_path_length)
    print(test5.max_dependency_path_length)
    print(test6.max_dependency_path_length)
    dummy_input, _ = test6.generate_input(1)
    test6.task_engine.gpu_configurations[0].execute(dummy_input, torch.cuda.Stream())
    test6.task_engine.cpu_configurations[0].execute(dummy_input)

    print("finito")

import json
import os
from dataclasses import replace, asdict
from pathlib import Path
from pprint import pprint
from statistics import mean
from typing import Dict, Any, Tuple, List

from scheduling.Scheduler import SchedulingResult
from tasks.GestureClassification import GestureClassification
from tasks.HandDetection import HandDetection
from tasks.InferenceTask import InferenceTask
from xrbench_helper import xrbench_evaluation_no_energy, xrbench_evaluation_with_energy


def _load_results(results_directory : str | os.PathLike) -> Tuple[Dict[str, Dict[str, SchedulingResult]], Dict]:
    results_dir = Path(results_directory)
    results_file : Path = results_dir / "results.json"
    with results_file.open("r") as f:
        data = json.load(f)

    results = {
        k1: {
            k2: SchedulingResult(**d)
            for k2, d in v.items()
        }
        for k1, v in data.items()
    }

    settings_file : Path = results_dir / "settings.json"
    with settings_file.open("r") as f:
        settings = json.load(f)

    return results, settings

def _propagate_quality_scores(results : Dict[str, Dict[str, SchedulingResult]]) -> Dict[str, Dict[str, SchedulingResult]]:
    out = {}
    for task in results.keys():
        propagated_task_results : Dict[str, SchedulingResult] = {}
        for result_id in results[task].keys():
            current : SchedulingResult = results[task][result_id]
            propagated_quality_score = current.quality_score
            predecessor = current.predecessor_request_id
            while predecessor is not None:
                predecessor_task = InferenceTask.get_task_name_from_request_id(predecessor)
                predecessor_result : SchedulingResult = results[predecessor_task][predecessor]
                propagated_quality_score *= predecessor_result.quality_score
                predecessor = predecessor_result.predecessor_request_id

            propagated_task_results[result_id] = replace(current, quality_score=propagated_quality_score)
        out[task] = propagated_task_results

    return out

def _prepare_evaluation_dict(results : Dict[str, Dict[str, SchedulingResult]]) -> Dict[str, Dict[str, Any]]:
    return {task_name: {} for task_name in results.keys()}

def _missed_deadline(result : SchedulingResult) -> bool:
    return result.finish_time > result.deadline


def _calculate_basic_execution_statistics(results : Dict[str, Dict[str, SchedulingResult]],
                                    evaluation : Dict[str, Any]) -> None:
    for task_name, task_results in results.items():
        num_requests = len(task_results)
        finished_requests = sum(task_result.finished_execution for task_result in task_results.values())
        unfinished_requests = num_requests - finished_requests

        total_missed_deadlines = sum((not task_result.finished_execution) or _missed_deadline(task_result)
                                     for task_result in task_results.values())

        executed_missed_deadlines = sum(task_result.finished_execution and _missed_deadline(task_result)
                                        for task_result in task_results.values())

        evaluation[task_name]["requests_total"] = num_requests
        evaluation[task_name]["requests_finished"] = finished_requests
        evaluation[task_name]["requests_unfinished"] = unfinished_requests
        evaluation[task_name]["deadlines_missed_total"] = total_missed_deadlines
        evaluation[task_name]["deadlines_missed_executed"] = executed_missed_deadlines

def _calculate_mean_quality_scores(results : Dict[str, Dict[str, SchedulingResult]], is_quality_propagated : bool,
                                      evaluation : Dict[str, Any]) -> None:
    for task_name, task_results in results.items():
        executed_results = [task_result for task_result in task_results.values() if task_result.finished_execution]
        mean_quality_score = mean(r.quality_score for r in executed_results) if executed_results else 0.0
        if is_quality_propagated:
            evaluation[task_name]["quality_score_propagated_mean"] = mean_quality_score
        else:
            evaluation[task_name]["quality_score_mean"] = mean_quality_score

def _calculate_basic_timing_statistics(results : Dict[str, Dict[str, SchedulingResult]],
                                       evaluation : Dict[str, Dict[str, Any]]) -> None:
    for task_name, task_results in results.items():
        executed_results = [task_result for task_result in task_results.values() if task_result.finished_execution]
        evaluation[task_name]["mean_preparation_time"] = mean(result.preparation_time for result in executed_results) if executed_results else 0.0
        evaluation[task_name]["worst_preparation_time"] = max(result.preparation_time for result in executed_results) if executed_results else 0.0
        evaluation[task_name]["mean_inference_time"] = mean(result.inference_time for result in executed_results) if executed_results else 0.0
        evaluation[task_name]["worst_inference_time"] = max(result.inference_time for result in executed_results) if executed_results else 0.0

def _calculate_false_positive_timing_statistics(results : Dict[str, Dict[str, SchedulingResult]],
                                                evaluation : Dict[str, Dict[str, Any]],
                                                benchmark_results : Dict) -> None:
    fp_overheads_by_task : Dict[str, List[float]] = {}
    num_fp_never_executed_by_task : Dict[str, int] = {}
    for task_name, task_results in results.items():
        fp_results = [task_result for task_result in task_results.values() if task_result.false_positives > 0]
        for fp_result in fp_results:
            if fp_result.predecessor_request_id is None:
                raise ValueError("Error, false_positives > 0 even though the result does not have a predecessor, "
                                 "result:\n", asdict(fp_result))
            predecessor_task = InferenceTask.get_task_name_from_request_id(fp_result.predecessor_request_id)
            if predecessor_task not in fp_overheads_by_task:
                fp_overheads_by_task[predecessor_task] = []

            if not fp_result.finished_execution:
                if predecessor_task not in num_fp_never_executed_by_task:
                    num_fp_never_executed_by_task[predecessor_task] = 0
                num_fp_never_executed_by_task[predecessor_task] += 1
                continue

            model_variant_stats = benchmark_results[task_name][fp_result.model_id][fp_result.model_variant]
            if fp_result.false_positives == fp_result.batch_size:
                fp_overheads_by_task[predecessor_task].append(fp_result.inference_time)
            else:
                correct_batch_size = fp_result.batch_size - fp_result.false_positives
                fp_overhead = model_variant_stats[str(fp_result.batch_size)]["mean_inference_time"] - \
                    model_variant_stats[str(correct_batch_size)]["mean_inference_time"]

                fp_overheads_by_task[predecessor_task].append(fp_overhead)

    for task_name, task_results in results.items():
        executed_results = [task_result for task_result in task_results.values() if task_result.finished_execution]
        evaluation[task_name]["false_positives_total_generated"] = len(fp_overheads_by_task[task_name]) \
            if task_name in fp_overheads_by_task else 0
        evaluation[task_name]["false_positives_mean_overhead_created"] = sum(fp_overheads_by_task[task_name]) / len(executed_results) \
            if task_name in fp_overheads_by_task else 0.0
        evaluation[task_name]["false_positives_never_executed"] = num_fp_never_executed_by_task[task_name] \
            if task_name in num_fp_never_executed_by_task else 0


def _calculate_basic_and_fp_timing_statistics(results : Dict[str, Dict[str, SchedulingResult]],
                                              evaluation : Dict[str, Any],
                                              benchmark_results : Dict):
    _calculate_basic_timing_statistics(results, evaluation)
    _calculate_false_positive_timing_statistics(results, evaluation, benchmark_results)


def basic_evaluation(results : Dict[str, Dict[str, SchedulingResult]],
                     evaluation : Dict[str, Any],
                     benchmark_results : Dict) -> None:
    propagated_results = _propagate_quality_scores(results)
    _calculate_basic_execution_statistics(results, evaluation)
    _calculate_mean_quality_scores(results, False, evaluation)
    _calculate_mean_quality_scores(propagated_results, True, evaluation)
    _calculate_basic_and_fp_timing_statistics(results, evaluation, benchmark_results)

def xrbench_full_evaluation_no_energy_default_workload(results : Dict[str, Dict[str, SchedulingResult]],
                                                       evaluation : Dict[str, Any]) -> None:
    xrbench_evaluation_no_energy(results, evaluation, False, False, spoof_deadlines=[(HandDetection.task_name, GestureClassification.task_name)])
    xrbench_evaluation_no_energy(results, evaluation, True, False, spoof_deadlines=[(HandDetection.task_name, GestureClassification.task_name)])
    propagated_results = _propagate_quality_scores(results)
    xrbench_evaluation_no_energy(propagated_results, evaluation, False, True, [HandDetection.task_name])
    xrbench_evaluation_no_energy(propagated_results, evaluation, True, True, [HandDetection.task_name])

def xrbench_full_evaluation_with_energy_default_workload(results : Dict[str, Dict[str, SchedulingResult]],
                                                         evaluation : Dict[str, Any],
                                                         energy_max : float) -> None:
    xrbench_evaluation_with_energy(results, evaluation, False, False, energy_max, spoof_deadlines=[(HandDetection.task_name, GestureClassification.task_name)])
    xrbench_evaluation_with_energy(results, evaluation, True, False, energy_max,spoof_deadlines=[(HandDetection.task_name, GestureClassification.task_name)])
    propagated_results = _propagate_quality_scores(results)
    xrbench_evaluation_with_energy(propagated_results, evaluation, False, True, energy_max,[HandDetection.task_name])
    xrbench_evaluation_with_energy(propagated_results, evaluation, True, True, energy_max,[HandDetection.task_name])


def energy_violation_evaluation(results : Dict[str, Dict[str, SchedulingResult]],
                                evaluation : Dict[str, Any],
                                interval_length : float, energy_limit : float) -> None:
    all_finished_results = []
    for result_dict in results.values():
        for result in result_dict.values():
            if result.finished_execution:
                all_finished_results.append(result)

    results_sorted_by_start_times : List[SchedulingResult] = sorted(all_finished_results, key= lambda r: r.get_start_time())
    all_start_and_finish_times : List[float] = []
    for result in results_sorted_by_start_times:
        all_start_and_finish_times.append(result.get_start_time())
        all_start_and_finish_times.append(result.finish_time)

    sorted_start_and_finish_times : List[float] = sorted(all_start_and_finish_times)

    active_results = []
    energy_violations : int = 0

    current_start_time_index = 0
    for current_interval_finish in sorted_start_and_finish_times:
        current_interval_start = current_interval_finish - interval_length

        while current_start_time_index < len(results_sorted_by_start_times) and \
            results_sorted_by_start_times[current_start_time_index].get_start_time() < current_interval_finish:
            active_results.append(results_sorted_by_start_times[current_start_time_index])
            current_start_time_index += 1

        active_results : List[SchedulingResult] = [result for result in active_results if result.finish_time > current_interval_start]

        total_interval_energy = 0.0
        for active_result in active_results:
            overlap_start = max(current_interval_start, active_result.get_start_time())
            overlap_end = min(current_interval_finish, active_result.finish_time)

            if overlap_start >= overlap_end:
                raise ValueError("Error, active_result doesn't overlap with currently considered energy interval")

            overlap = overlap_end - overlap_start
            if overlap > active_result.inference_time:
                raise ValueError("Error, overlap was somehow larger than active_result.inference_time")

            energy_ratio = overlap / active_result.inference_time

            energy_consumed_in_interval = active_result.consumed_energy * energy_ratio
            total_interval_energy += energy_consumed_in_interval

        if total_interval_energy > energy_limit:
            energy_violations += 1
            # only count each inference once for energy violations for now
            #active_results.clear()
            # do not count finished requests twice for energy violations
            active_results = [result for result in active_results if result.finish_time < current_interval_finish]

    evaluation["energy_limit_violations"] = energy_violations

def energy_violation_time_evaluation(results : Dict[str, Dict[str, SchedulingResult]],
                                evaluation : Dict[str, Any],
                                interval_length : float, energy_limit : float) -> None:
    all_finished_results = []
    for result_dict in results.values():
        for result in result_dict.values():
            if result.finished_execution:
                all_finished_results.append(result)

    results_sorted_by_start_times : List[SchedulingResult] = sorted(all_finished_results, key= lambda r: r.get_start_time())
    all_timestamps : List[float] = []
    for result in results_sorted_by_start_times:
        all_timestamps.append(result.get_start_time())
        all_timestamps.append(result.finish_time)
        all_timestamps.append(result.finish_time + interval_length) # timestamp where the request leaves the left side of the interval

    sorted_timestamps : List[float] = sorted(all_timestamps)

    active_results = []
    energy_violation_intervals : List[Tuple[float, float]] = []

    current_start_time_index = 0
    for current_interval_finish in sorted_timestamps:
        current_interval_start = current_interval_finish - interval_length

        while current_start_time_index < len(results_sorted_by_start_times) and \
            results_sorted_by_start_times[current_start_time_index].get_start_time() < current_interval_finish:
            active_results.append(results_sorted_by_start_times[current_start_time_index])
            current_start_time_index += 1

        active_results : List[SchedulingResult] = [result for result in active_results if result.finish_time > current_interval_start]

        total_interval_energy = 0.0
        for active_result in active_results:
            overlap_start = max(current_interval_start, active_result.get_start_time())
            overlap_end = min(current_interval_finish, active_result.finish_time)

            if overlap_start >= overlap_end:
                raise ValueError("Error, active_result doesn't overlap with currently considered energy interval")

            overlap = overlap_end - overlap_start
            if overlap > active_result.inference_time:
                raise ValueError("Error, overlap was somehow larger than active_result.inference_time")

            energy_ratio = overlap / active_result.inference_time

            energy_consumed_in_interval = active_result.consumed_energy * energy_ratio
            total_interval_energy += energy_consumed_in_interval

        if total_interval_energy > energy_limit:
            energy_violation_intervals.append((current_interval_start, current_interval_finish))

    if not energy_violation_intervals:
        evaluation["energy_limit_violation_time"] = 0.0
        return

    sorted_energy_violation_intervals = sorted(energy_violation_intervals)

    # merge all overlapping intervals until there are no overlapping intervals anymore
    merged_intervals : List[Tuple[float, float]] = [sorted_energy_violation_intervals[0]]

    for current_start, current_end in sorted_energy_violation_intervals[1:]:
        merged_start, merged_end = merged_intervals[-1]

        if current_start <= merged_end:
            merged_intervals[-1] = (merged_start, max(merged_end, current_end))
        # since we sorted (by start_time implicitly) before, it is guaranteed that no later interval will overlap again
        else:
            merged_intervals.append((current_start, current_end))

    evaluation["energy_limit_violation_time"] = sum(end - start for start, end in merged_intervals)
    evaluation["energy_consumed_total"] = sum(r.consumed_energy for r in all_finished_results)


def energy_violation_total_joule_overhead(results : Dict[str, Dict[str, SchedulingResult]],
                                          evaluation : Dict[str, Any],
                                          interval_length : float, energy_limit : float, duration: float) -> None:
    all_finished_results = []
    for result_dict in results.values():
        for result in result_dict.values():
            if result.finished_execution:
                all_finished_results.append(result)

    duration_factor = duration / interval_length
    total_energy_limit = energy_limit * duration_factor

    total_consumed_energy = sum(r.consumed_energy for r in all_finished_results)

    evaluation["energy_total_overhead"] = total_consumed_energy - total_energy_limit


def main():
    benchmark_name = "./scheduling_results/EMERALD_Mean_1_Stream"
    print("benchmark_name:", benchmark_name)

    results_dir = Path(benchmark_name)
    results, settings = _load_results(results_dir)
    benchmark_results_path = Path(settings["used_benchmark"])
    with open(benchmark_results_path) as f:
        benchmark_results = json.load(f)

    print("results settings:")
    pprint(settings)
    print("-------------------------------------------")
    print("results evaluation:")
    print("-------------------------------------------")
    evaluation = _prepare_evaluation_dict(results)

    basic_evaluation(results, evaluation, benchmark_results)
    xrbench_full_evaluation_no_energy_default_workload(results, evaluation)

    duration = settings["duration"]
    energy_interval_length = settings["energy_interval_length"]
    energy_limit = settings["energy_limit"]
    xrbench_energy_max = settings["xrbench_energy_max"]
    energy_violation_evaluation(results, evaluation, energy_interval_length, energy_limit)
    energy_violation_time_evaluation(results, evaluation, energy_interval_length, energy_limit)
    # doesn't really work, because it is reduced by all the intervals where the energy limit isn't being violated
    # also doesn't really work for "merged intervals" => if one had the violation right at the beginning and the other at the end,
    # there is a subinterval of length < energy_interval_length that would be considered => negative overhead possible
    #energy_violation_total_joule_overhead(results, evaluation, energy_interval_length, energy_limit, duration)
    xrbench_full_evaluation_with_energy_default_workload(results, evaluation, xrbench_energy_max)

    pprint(evaluation)


if __name__ == "__main__":
    main()
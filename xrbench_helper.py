import math
from dataclasses import replace
from statistics import mean
from typing import Dict, Tuple, List, Any

from scheduling.Scheduler import SchedulingResult


def calculate_xrbench_real_time_score(result : SchedulingResult, is_normalize : bool, k : int = 15) -> float:
    inference_time = result.finish_time - result.request_time
    inference_slack = result.deadline - result.request_time

    exponent = k * (inference_time - inference_slack)
    if is_normalize:
        exponent = exponent / inference_slack
    exponent = min(exponent, 500) # limit max exponent to prevent overflow, basically 0 anyway
    return 1 / (1 + math.exp(exponent))


def calculate_xrbench_per_inference_score_no_energy(result : SchedulingResult, is_normalize : bool) -> float:
    return calculate_xrbench_real_time_score(result, is_normalize) * result.quality_score


def calculate_xrbench_model_and_qoe_score_no_energy(task_results : Dict[str, SchedulingResult],
                                                    is_normalize : bool) -> Tuple[float, float]:
    executed_results = [task_result for task_result in task_results.values() if task_result.finished_execution]
    inference_scores = [calculate_xrbench_per_inference_score_no_energy(result, is_normalize)
                        for result in executed_results]
    return mean(inference_scores) if inference_scores else 0.0, len(executed_results) / len(task_results)


def calculate_xrbench_usage_scenario_score(model_and_qoe_scores : List[Tuple[float, float]]) -> float:
    combined_scores = [model_score * qoe_score for (model_score, qoe_score) in model_and_qoe_scores]
    return mean(combined_scores)


def xrbench_evaluation_no_energy(results : Dict[str, Dict[str, SchedulingResult]],
                                 evaluation : Dict[str, Any],
                                 is_normalize : bool, are_results_propagated : bool,
                                 ignore_tasks : List[str] = None,
                                 spoof_deadlines : List[Tuple[str, str]] = None) -> None:
    if ignore_tasks is None:
        ignore_tasks = []

    if spoof_deadlines is None:
        spoof_deadlines = []

    xrbench = {}

    model_and_qoe_scores_normal = []
    for task_name, task_results in results.items():
        if task_name in ignore_tasks:
            continue
        for task, inherit_from_task in spoof_deadlines:
            if task_name == task:
                # do not modify the original results
                modified_task_results = {}
                inherit_from_results = results[inherit_from_task]

                for original_result_id, original_result in task_results.items():
                    inherit_from_result = next((r for r in inherit_from_results.values() if r.predecessor_request_id == original_result_id), None)
                    # the very last request may not have generated a request => just ignore it, missing one is insignificant
                    if inherit_from_result is None:
                        continue
                    modified_task_results[original_result_id] = replace(original_result, deadline=inherit_from_result.deadline)
                task_results = modified_task_results
                break
        model_score, qoe_score = calculate_xrbench_model_and_qoe_score_no_energy(task_results, is_normalize)
        model_and_qoe_scores_normal.append((model_score, qoe_score))
        xrbench[task_name] = {}
        xrbench[task_name]["model_score"] = model_score
        xrbench[task_name]["qoe_score"] = qoe_score
    usage_scenario_score = calculate_xrbench_usage_scenario_score(model_and_qoe_scores_normal)
    xrbench["usage_scenario_score"] = usage_scenario_score

    if is_normalize and are_results_propagated:
        eval_key = "xrbench_normalized_propagated_no_energy"
    elif is_normalize and not are_results_propagated:
        eval_key = "xrbench_normalized_normal_no_energy"
    elif not is_normalize and are_results_propagated:
        eval_key = "xrbench_propagated_no_energy"
    elif not is_normalize and not are_results_propagated:
        eval_key = "xrbench_normal_no_energy"
    else:
        raise ValueError("Could not compute appropriate eval_key for xrbench_evaluation_no_energy")

    evaluation[eval_key] = xrbench

def calculate_xrbench_energy_score(result : SchedulingResult, energy_max : float) -> float:
    if result.consumed_energy > energy_max:
        raise ValueError(f"Error, result.consumed_energy ({result.consumed_energy}) is greater than energy_max ({energy_max})")
    return (energy_max - result.consumed_energy) / energy_max

def calculate_xrbench_per_inference_score_with_energy(result : SchedulingResult, is_normalize : bool, energy_max : float) -> float:
    return calculate_xrbench_real_time_score(result, is_normalize) * calculate_xrbench_energy_score(result, energy_max) * result.quality_score

def calculate_xrbench_model_and_qoe_score_with_energy(task_results : Dict[str, SchedulingResult],
                                                    is_normalize : bool, energy_max : float) -> Tuple[float, float]:
    executed_results = [task_result for task_result in task_results.values() if task_result.finished_execution]
    inference_scores = [calculate_xrbench_per_inference_score_with_energy(result, is_normalize, energy_max)
                        for result in executed_results]
    return mean(inference_scores) if inference_scores else 0.0, len(executed_results) / len(task_results)

def xrbench_evaluation_with_energy(results : Dict[str, Dict[str, SchedulingResult]],
                                   evaluation : Dict[str, Any],
                                   is_normalize : bool, are_results_propagated : bool,
                                   energy_max : float,
                                   ignore_tasks : List[str] = None,
                                   spoof_deadlines : List[Tuple[str, str]] = None) -> None:
    if ignore_tasks is None:
        ignore_tasks = []
    if spoof_deadlines is None:
        spoof_deadlines = []
    xrbench = {}

    model_and_qoe_scores_normal = []
    for task_name, task_results in results.items():
        if task_name in ignore_tasks:
            continue
        for task, inherit_from_task in spoof_deadlines:
            if task_name == task:
                # do not modify the original results
                modified_task_results = {}
                inherit_from_results = results[inherit_from_task]

                for original_result_id, original_result in task_results.items():
                    inherit_from_result = next(r for r in inherit_from_results.values() if r.predecessor_request_id == original_result_id)
                    modified_task_results[original_result_id] = replace(original_result, deadline=inherit_from_result.deadline)
                task_results = modified_task_results
                break
        model_score, qoe_score = calculate_xrbench_model_and_qoe_score_with_energy(task_results, is_normalize, energy_max)
        model_and_qoe_scores_normal.append((model_score, qoe_score))
        xrbench[task_name] = {}
        xrbench[task_name]["model_score"] = model_score
        xrbench[task_name]["qoe_score"] = qoe_score
    usage_scenario_score = calculate_xrbench_usage_scenario_score(model_and_qoe_scores_normal)
    xrbench["usage_scenario_score"] = usage_scenario_score

    if is_normalize and are_results_propagated:
        eval_key = "xrbench_normalized_propagated_with_energy"
    elif is_normalize and not are_results_propagated:
        eval_key = "xrbench_normalized_normal_with_energy"
    elif not is_normalize and are_results_propagated:
        eval_key = "xrbench_propagated_with_energy"
    elif not is_normalize and not are_results_propagated:
        eval_key = "xrbench_normal_with_energy"
    else:
        raise ValueError("Could not compute appropriate eval_key for xrbench_evaluation_with_energy")

    evaluation[eval_key] = xrbench


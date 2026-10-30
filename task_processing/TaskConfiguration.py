from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Final

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scheduling.InferenceRequest import InferenceStats

from task_processing.input_preparation.ConfigurationInputConverter import MultidimensionalArray, \
    ConfigurationInputConverter

@dataclass
class TaskConfiguration:
    task: Final[str]
    model_id: Final[str]
    model_warmup_cost : float

    model_variant: Final[str]
    switch_to_variant_cost : float

    input_converter: ConfigurationInputConverter

    quality : float
    # dicts mapping batch size to the corresponding mean and worst case inference times measured beforehand to support
    # dynamic batch sizes for dependency batching
    mean_inference_time : Dict[int, float]
    # static in name since it is not updated at runtime
    static_98th_percentile_inference_time : Dict[int, float]
    worst_case_inference_time : Dict[int, float]

    mean_preparation_time : Dict[int, float]
    worst_case_preparation_time : Dict[int, float]

    # mapping energy_stat_name (default, early-exit, late-exit) to batch sizes to benchmarked energy consumption
    energy_statistics : Dict[str, Dict[int, float]]

    false_positive_probability : float = 0.0


    def prepare_input(self, model_input : Dict[str, MultidimensionalArray]) -> Dict[str, MultidimensionalArray]:
        return self.input_converter.convert_input(model_input)

    def calculate_switching_cost(self, other : TaskConfiguration) -> float:
        if self.task != other.task:
            raise ValueError("Error, attempting to calculate switching cost to a configuration for a different task.", os.linesep,
                             "self.task:", self.task, os.linesep,
                             "self.model_id", self.model_id, os.linesep,
                             "self.model_variant", self.model_variant, os.linesep,
                             "other.task:", other.task, os.linesep,
                             "other.model_id", other.model_id, os.linesep,
                             "other.model_variant", other.model_variant
                             )
        if self.model_id != other.model_id:
            # assume that the variant switching time is also needed to overestimate should it not be the case
            # instead of underestimating by not including it, when it is actually required
            return other.model_warmup_cost + other.switch_to_variant_cost
        if self.model_variant != other.model_variant:
            return other.switch_to_variant_cost
        return 0

    @staticmethod
    def create(task_name : str, model_id : str, model_warmup_cost : float, model_variant : str,
               input_converter : ConfigurationInputConverter,
               model_quality : float,
               benchmark : Dict,
               energy_statistics: Dict[str, Dict[int, float]],
               batch_sizes : Iterable[int] = None,
               false_positive_probability : float = 0.0) -> TaskConfiguration:
        variant_benchmark = benchmark[task_name][model_id][model_variant]
        if batch_sizes is None:
            batch_sizes = [1]

        mean_inference_times = {}
        worst_case_inference_times = {}
        mean_preparation_times = {}
        static_98th_percentile_inference_times = {}
        worst_case_preparation_times = {}

        variant_switching_cost = 0.0

        for batch_size in batch_sizes:
            mean_inference_times[batch_size] = variant_benchmark[str(batch_size)]["mean_inference_time"]
            worst_case_inference_times[batch_size] = variant_benchmark[str(batch_size)]["worst_inference_time"]
            mean_preparation_times[batch_size] = variant_benchmark[str(batch_size)]["mean_input_preparation_time"]
            static_98th_percentile_inference_times[batch_size] = variant_benchmark[str(batch_size)]["98th_percentile_inference_time"]
            worst_case_preparation_times[batch_size] = variant_benchmark[str(batch_size)]["worst_input_preparation_time"]

            variant_switching_cost = max(variant_switching_cost, variant_benchmark[str(batch_size)]["variant_switching_cost"])

        return TaskConfiguration(
            task_name, model_id, model_warmup_cost, model_variant, variant_switching_cost,
            input_converter, model_quality, mean_inference_times, static_98th_percentile_inference_times, worst_case_inference_times,
            mean_preparation_times, worst_case_preparation_times, energy_statistics, false_positive_probability
        )

    def update(self, new_inference : InferenceStats) -> None:
        batch_size = new_inference.request.batch_size
        new_inference_time = new_inference.inference_time
        new_preparation_time = new_inference.preparation_time
        # exponential moving average
        self.mean_inference_time[batch_size] += 0.01 * (new_inference_time - self.mean_inference_time[batch_size])
        self.worst_case_inference_time[batch_size] = max(new_inference_time, self.worst_case_inference_time[batch_size])

        self.mean_preparation_time[batch_size] += 0.01 * (new_preparation_time - self.mean_preparation_time[batch_size])
        self.worst_case_preparation_time[batch_size] = max(new_preparation_time, self.worst_case_preparation_time[batch_size])
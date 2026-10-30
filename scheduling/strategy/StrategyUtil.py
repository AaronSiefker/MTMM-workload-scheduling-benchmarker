from typing import List, Tuple, Iterable, Callable

from scheduling.InferenceRequest import InferenceRequest
from task_processing.GpuConfiguration import GpuConfiguration
from task_processing.TaskConfiguration import TaskConfiguration
from task_processing.TaskEngine import TaskEngine


def select_lowest_deadline(requests: List[InferenceRequest]) -> Tuple[int, InferenceRequest | None]:
    if not requests:
        return -1, None
    return min(enumerate(requests), key=lambda pair: pair[1].deadline)

def adjust_active_gpu_configurations_greedy(task_engines : Iterable[TaskEngine], negative_slack : float,
                                            calculate_improvement : Callable[[TaskConfiguration, TaskConfiguration], float]) \
        -> None:
    """

    :param task_engines:
    :param negative_slack:
    :param calculate_improvement: argument ordering: (current config, other config)
    :return:
    """
    time_diff = -negative_slack
    all_possible_improvements: List[Tuple[float, TaskEngine, GpuConfiguration]] = [
        (improvement, task_engine, config)
        for task_engine in task_engines
        for config in task_engine.gpu_configurations
        if (improvement := calculate_improvement(task_engine.active_gpu_configuration.task_configuration,
                                                 config.task_configuration))
           > 0
    ]
    while all_possible_improvements:
        improvement, task_engine, config = max(all_possible_improvements, key=lambda t: t[0])
        if improvement >= time_diff:
            #  If we can select a config which overcomes the time_diff, select the best of all such configs
            _, task_engine, config = min(
                (t for t in all_possible_improvements if t[0] >= time_diff),
                key=lambda t: t[0]
            )
            task_engine.active_gpu_configuration = config
            return

        task_engine.active_gpu_configuration = config
        time_diff -= improvement
        # Only update each config once, do not consider it anymore
        all_possible_improvements = [
            t for t in all_possible_improvements
            if t[1] != task_engine
        ]
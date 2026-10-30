from task_processing.GpuConfiguration import GpuConfiguration


def get_emerald_inference_time(gpu_config : GpuConfiguration, is_mean : bool):
    if is_mean:
        inference_times_by_batch_size = gpu_config.task_configuration.mean_inference_time
    else:
        inference_times_by_batch_size = gpu_config.task_configuration.static_98th_percentile_inference_time

    # for Default Workload only relevant for Gesture Classification, where batch_size=4 isn't much slower than batch_size=1
    max_batch_size = max(inference_times_by_batch_size)
    return inference_times_by_batch_size[max_batch_size]

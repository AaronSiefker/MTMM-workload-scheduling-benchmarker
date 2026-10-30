from typing import List, Final

from task_processing.CpuConfiguration import CpuConfiguration
from task_processing.GpuConfiguration import GpuConfiguration


class TaskEngine:
    def __init__(self, task : str, base_gpu_configuration : GpuConfiguration,
                 base_cpu_configuration : CpuConfiguration):
        self.task : Final[str] = task

        self.gpu_configurations : List[GpuConfiguration] = []
        self.add_gpu_configuration(base_gpu_configuration)
        self.cpu_configurations : List[CpuConfiguration] = []
        self.add_cpu_configuration(base_cpu_configuration)

        self.active_gpu_configuration : GpuConfiguration = base_gpu_configuration
        self.active_cpu_configuration : CpuConfiguration = base_cpu_configuration

    def add_gpu_configuration(self, gpu_configuration : GpuConfiguration):
        if gpu_configuration.task_configuration.task != self.task:
            raise ValueError(f"Error, the provided gpu_configuration is for "
                             f"task '{gpu_configuration.task_configuration.task}, "
                             f"whereas '{self.task}’ was given for this TaskEngine")

        self.gpu_configurations.append(gpu_configuration)

    def add_cpu_configuration(self, cpu_configuration : CpuConfiguration):
        if cpu_configuration.task_configuration.task != self.task:
            raise ValueError(f"Error, the provided base_cpu_configuration is for "
                             f"task '{cpu_configuration.task_configuration.task}, "
                             f"whereas '{self.task}’ was given for this TaskEngine")

        self.cpu_configurations.append(cpu_configuration)

    def change_active_gpu_configuration(self, gpu_configuration : GpuConfiguration):
        if gpu_configuration.task_configuration.task != self.task:
            raise ValueError(f"Error, the provided gpu_configuration is for "
                             f"task '{gpu_configuration.task_configuration.task}', whereas"
                             f"this TaskEngine is for task {self.task}")
        self.active_gpu_configuration = gpu_configuration

    def get_index_of_active_gpu_configuration(self) -> int:
        for i in range(len(self.gpu_configurations)):
            if self.active_gpu_configuration.task_configuration.model_variant == self.gpu_configurations[i].task_configuration.model_variant:
                return i
        raise ValueError("Error, could not find index of active gpu config in self.gpu_configurations")

    def change_active_cpu_configuration(self, cpu_configuration : CpuConfiguration):
        if cpu_configuration.task_configuration.task != self.task:
            raise ValueError(f"Error, the provided cpu_configuration is for "
                             f"task '{cpu_configuration.task_configuration.task}', whereas"
                             f"this TaskEngine is for task {self.task}")
        self.active_cpu_configuration = cpu_configuration
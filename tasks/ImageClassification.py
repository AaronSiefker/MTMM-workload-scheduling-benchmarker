from pathlib import Path
from typing import Final, List, Dict

import numpy as np
from PIL import Image

from scheduling.Scheduler import Scheduler
from task_processing.CpuConfiguration import CpuConfiguration
from task_processing.DefaultGpuConfiguration import DefaultGpuConfiguration
from task_processing.TaskConfiguration import TaskConfiguration
from task_processing.TaskEngine import TaskEngine
from task_processing.cpu_execution.OnnxModel import OnnxModel
from task_processing.gpu_execution.TensorRtEngine import TensorRtEngine
from task_processing.gpu_execution.TensorRtModel import TensorRtModel
from task_processing.input_preparation.ConfigurationInputConverter import ConfigurationInputConverter, \
    MultidimensionalArray
from task_processing.input_preparation.ImageNetConverter import ImageNetConverter
from tasks.InferenceTask import InferenceTask
from tasks.dependencies.Dependency import Dependency
from tasks.post_processing import gpu_post_processing_image_classification, cpu_post_processing_image_classification


class ImageClassification(InferenceTask):
    task_name : Final[str] = "Image Classification"
    def __init__(self, scheduler : Scheduler, relative_deadline, gpu_benchmark_results : Dict, cpu_benchmark_results : Dict,
                 num_streams : int,
                 dependencies : List[Dependency] | None = None):
        super().__init__(ImageClassification.task_name, scheduler, relative_deadline, dependencies)

        model_id : str = "dinov2-vits14"
        image_key : str = "image"
        input_converter : ConfigurationInputConverter = ImageNetConverter(image_key)
        # NOTE: warmup taken from pytorch, didn't write it into my notes since here I only have this model anyway
        # for now, can extract from .npy-file later if necessary
        trt_engine_variant = "fp16"
        trt_engine_path : str = f"../TensorRT/Image_Classification/dinov2_{trt_engine_variant}.engine"

        gpu_task_configuration = TaskConfiguration.create(
            self.task, f"trt-{model_id}_{trt_engine_variant}", 0.110, "default",
            input_converter, 0.877, gpu_benchmark_results, {"default": {1: 0.022488}}
        )
        onnx_model_path : str = "../OnnxModels/Image_Classification/dinov2_vits14_Image_Classification.onnx"
        cpu_task_configuration = TaskConfiguration.create(
            self.task, f"onnx-{model_id}_fp32", 0.0, "default",
            input_converter, 0.8772, cpu_benchmark_results, {}
        )
        trt_model = TensorRtModel(TensorRtEngine(trt_engine_path))
        onnx_model = OnnxModel(onnx_model_path)
        self.task_engine : Final[TaskEngine] = TaskEngine(self.task,
            DefaultGpuConfiguration(
                gpu_task_configuration, trt_model,
                self.finalize_gpu_post_processing(gpu_post_processing_image_classification("logits")), num_streams),
            CpuConfiguration(
                cpu_task_configuration, onnx_model,
                self.finalize_cpu_post_processing(cpu_post_processing_image_classification(0)))
        )
        self._fill_inputs()

    def _fill_inputs(self):
        datapath = Path("~/scheduler_inputs/Image_Classification").expanduser()
        for file in datapath.iterdir():
            if file.is_file():
                image = Image.open(file).convert("RGB")
                if image is None:
                    raise ValueError("Could not load image:", file)
                self._inputs.append(np.array(image))

    def _generate_input_dictionary(self, input_array: MultidimensionalArray) -> Dict[str, MultidimensionalArray]:
        return {"image": input_array}

    def get_task_engine(self) -> TaskEngine:
        return self.task_engine

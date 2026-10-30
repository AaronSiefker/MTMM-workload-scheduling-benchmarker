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
from task_processing.input_preparation.ComposedConverter import ComposedConverter
from task_processing.input_preparation.ConfigurationInputConverter import ConfigurationInputConverter, \
    MultidimensionalArray
from task_processing.input_preparation.ImageNetConverter import ImageNetConverter
from task_processing.input_preparation.ImageResizeConverter import ImageResizeConverter
from tasks.InferenceTask import InferenceTask
from tasks.dependencies.Dependency import Dependency
from tasks.post_processing import gpu_post_processing_image_classification, cpu_post_processing_image_classification


class GestureClassification(InferenceTask):
    task_name : Final[str] = "Gesture Classification"
    def __init__(self, scheduler : Scheduler, relative_deadline, gpu_benchmark_results : Dict, cpu_benchmark_results : Dict,
                 num_streams : int,
                 dependencies : List[Dependency] | None = None):
        super().__init__(GestureClassification.task_name, scheduler, relative_deadline, dependencies)

        model_id : str = "MobileNetV3-Small"
        trt_engine_variant = "fp16"
        gpu_model_id : str = f"trt-{model_id}_{trt_engine_variant}"
        cpu_model_id : str = f"onnx-{model_id}_fp32"
        image_key : str = "x"
        image_net_converter : ImageNetConverter = ImageNetConverter(image_key)
        input_converter_size_256 : ConfigurationInputConverter = ComposedConverter(
            [ImageResizeConverter(image_key, 256), image_net_converter]
        )
        input_converter_size_192 : ConfigurationInputConverter = ComposedConverter(
            [ImageResizeConverter(image_key, 192), image_net_converter]
        )
        input_converter_size_160 : ConfigurationInputConverter = ComposedConverter(
            [ImageResizeConverter(image_key, 160), image_net_converter]
        )
        trt_engine_path : str = f"../TensorRT/Gesture_Classification/MobileNetV3Small_batched_{trt_engine_variant}.engine"
        batch_sizes = [1, 2, 3, 4]
        model_variant = "img_size=256"
        gpu_config_256 = TaskConfiguration.create(
            self.task, gpu_model_id, 0.147, model_variant, input_converter_size_256,
            0.9750, gpu_benchmark_results,
            {"default": {1: 0.003325, 2:  0.003961, 3: 0.005433, 4: 0.006944}},
            batch_sizes=batch_sizes
        )
        model_variant = "img_size=192"
        gpu_config_192 = TaskConfiguration.create(
            self.task, gpu_model_id, 0.147, model_variant, input_converter_size_192,
            0.9356, gpu_benchmark_results,
            {"default": {1: 0.002268, 2: 0.002877, 3: 0.003801, 4: 0.004700}},
            batch_sizes=batch_sizes
        )
        model_variant = "img_size=160"
        gpu_config_160 = TaskConfiguration.create(
            self.task, gpu_model_id, 0.147, model_variant, input_converter_size_160,
            0.8392, gpu_benchmark_results,
            {"default": {1: 0.002185, 2: 0.002369, 3: 0.003060, 4: 0.003733}},
            batch_sizes=batch_sizes
        )
        onnx_model_path = "../OnnxModels/Gesture_Classification/mobilenetv3_small_gesture_classification_batched_fp32.onnx"
        cpu_config_256 = TaskConfiguration.create(
            self.task, cpu_model_id, 0.0, "img_size=256", input_converter_size_256,
            0.9750, cpu_benchmark_results, {}, batch_sizes=batch_sizes
        )
        cpu_config_192 = TaskConfiguration.create(
            self.task, cpu_model_id, 0.0, "img_size=192", input_converter_size_192,
            0.9354, cpu_benchmark_results, {}, batch_sizes=batch_sizes
        )
        cpu_config_160 = TaskConfiguration.create(
            self.task, cpu_model_id, 0.0, "img_size=160", input_converter_size_160,
            0.8384, cpu_benchmark_results, {}, batch_sizes=batch_sizes
        )

        trt_model = TensorRtModel(TensorRtEngine(trt_engine_path))
        onnx_model = OnnxModel(onnx_model_path)
        gpu_post_processing = self.finalize_gpu_post_processing(gpu_post_processing_image_classification("linear_1"))
        cpu_post_processing = self.finalize_cpu_post_processing(cpu_post_processing_image_classification(0))
        self.task_engine : Final[TaskEngine] = TaskEngine(
            self.task, DefaultGpuConfiguration(gpu_config_256, trt_model, gpu_post_processing, num_streams),
            CpuConfiguration(cpu_config_256, onnx_model, cpu_post_processing)
        )
        self.task_engine.add_gpu_configuration(DefaultGpuConfiguration(
            gpu_config_192, trt_model, gpu_post_processing, num_streams
        ))
        self.task_engine.add_gpu_configuration(DefaultGpuConfiguration(
            gpu_config_160, trt_model, gpu_post_processing, num_streams
        ))

        self.task_engine.add_cpu_configuration(CpuConfiguration(
            cpu_config_192, onnx_model, cpu_post_processing
        ))
        self.task_engine.add_cpu_configuration(CpuConfiguration(
            cpu_config_160, onnx_model, cpu_post_processing
        ))
        self._fill_inputs()

    def _fill_inputs(self):
        datapath = Path("~/scheduler_inputs/Gesture_Classification").expanduser()
        for file in datapath.iterdir():
            if file.is_file():
                image = Image.open(file).convert("RGB")
                if image is None:
                    raise ValueError("Could not load image:", file)
                self._inputs.append(np.array(image))

    def _generate_input_dictionary(self, input_array: MultidimensionalArray) -> Dict[str, MultidimensionalArray]:
        return {"x": input_array}

    def get_task_engine(self) -> TaskEngine:
        return self.task_engine

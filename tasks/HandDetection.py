from pathlib import Path
from typing import Final, Dict, List

import cv2
import numpy as np
import torch
from torchvision.ops import nms as torch_nms

from scheduling.Scheduler import Scheduler
from task_processing.CpuConfiguration import CpuConfiguration
from task_processing.DefaultGpuConfiguration import DefaultGpuConfiguration
from task_processing.TaskConfiguration import TaskConfiguration
from task_processing.TaskEngine import TaskEngine
from task_processing.cpu_execution.OnnxInferenceJob import OnnxInferenceJob, OnnxInferenceJobStatus
from task_processing.cpu_execution.OnnxModel import OnnxModel
from task_processing.gpu_execution.TensorRtEngine import TensorRtEngine
from task_processing.gpu_execution.TensorRtInferenceJob import TensorRtInferenceJob
from task_processing.gpu_execution.TensorRtModel import TensorRtModel
from task_processing.input_preparation.ConfigurationInputConverter import ConfigurationInputConverter, \
    MultidimensionalArray
from tasks.InferenceTask import InferenceTask
from tasks.dependencies.Dependency import Dependency


class HandDetection(InferenceTask):
    class HandDetectionInputConverter(ConfigurationInputConverter):
        def __init__(self, image_key : str, size : int):
            self._image_key : Final[str] = image_key
            self._size : Final[int] = size

        def convert_input(self, model_input: Dict[str, MultidimensionalArray]) -> Dict[str, MultidimensionalArray]:
            image = np.asarray(model_input[self._image_key])
            if len(image.shape) == 4 and image.shape[0] == 1:
                image = image[0]
            if len(image.shape) != 3 and image.shape[2] != 3:
                raise ValueError("Error, this HandDetectionInputConverter can only convert images loaded via "
                                 "cv2.imread with len(image.shape) == 3 and image.shape[2] == 3;"
                                 "image.shape:", image.shape)
            out = model_input.copy()
            # if it reached this point, this shouldn't fail, originally complains about possibility of Any or None
            # but with previous checks this should always be an np.ndarray
            # noinspection PyTypeChecker
            out[self._image_key] = cv2.dnn.blobFromImage(image, 1 / 255.0, (self._size, self._size), swapRB=True, crop=False)
            return out


    image_key : Final[str] = "input"
    task_name : Final[str] = "Hand Detection"
    def __init__(self, scheduler : Scheduler, relative_deadline, gpu_benchmark_results : Dict, cpu_benchmark_results : Dict,
                 num_streams : int,
                 dependencies : List[Dependency] | None = None):
        super().__init__(HandDetection.task_name, scheduler, relative_deadline, dependencies)

        model_id = "yolov3-tiny"
        trt_engine_variant = "fp16"
        gpu_model_id = f"trt-{model_id}_{trt_engine_variant}"
        cpu_model_id = f"onnx-{model_id}_fp32"
        input_converter_size_512 : ConfigurationInputConverter = HandDetection.HandDetectionInputConverter(HandDetection.image_key, 512)
        input_converter_size_416 : ConfigurationInputConverter = HandDetection.HandDetectionInputConverter(HandDetection.image_key, 416)
        input_converter_size_320 : ConfigurationInputConverter = HandDetection.HandDetectionInputConverter(HandDetection.image_key, 320)
        trt_engine_path = f"../TensorRT/Hand_Detection/yolov3_tiny_{trt_engine_variant}.engine"
        model_variant = "img_size=512"
        gpu_config_512 = TaskConfiguration.create(
            self.task, gpu_model_id, 0.190, model_variant, input_converter_size_512,
            0.9321, gpu_benchmark_results, {"default": {1: 0.017762}},
            false_positive_probability=0.025370675453047775
        )
        model_variant = "img_size=416"
        gpu_config_416 = TaskConfiguration.create(
            self.task, gpu_model_id, 0.190, model_variant, input_converter_size_416,
            0.9029, gpu_benchmark_results, {"default": {1: 0.012820}},
            false_positive_probability=0.027196652719665274
        )
        model_variant = "img_size=320"
        gpu_config_320 = TaskConfiguration.create(
            self.task, gpu_model_id, 0.190, model_variant, input_converter_size_320,
            0.8240, gpu_benchmark_results, {"default": {1: 0.008357}},
            false_positive_probability=0.060324825986078884
        )
        onnx_model_path = "../OnnxModels/Hand_Detection/yolov3-tiny_hand_detection_fp32.onnx"
        cpu_config_512 = TaskConfiguration.create(
            self.task, cpu_model_id, 0.0, "img_size=512", input_converter_size_512,
            0.9326, cpu_benchmark_results, {}, false_positive_probability=0.024719841793012523
        )
        cpu_config_416 = TaskConfiguration.create(
            self.task, cpu_model_id, 0.0, "img_size=416", input_converter_size_416,
            0.9034, cpu_benchmark_results, {}, false_positive_probability=0.027168234064785787
        )
        cpu_config_320 = TaskConfiguration.create(
            self.task, cpu_model_id, 0.0, "img_size=320", input_converter_size_320,
            0.8247, cpu_benchmark_results, {}, false_positive_probability=0.060933281912842266
        )
        trt_model = TensorRtModel(TensorRtEngine(trt_engine_path))
        onnx_model = OnnxModel(onnx_model_path)
        gpu_post_processing = self.finalize_gpu_post_processing(HandDetection.gpu_post_processing)
        cpu_post_processing = self.finalize_cpu_post_processing(HandDetection.cpu_post_processing)
        self.task_engine : Final[TaskEngine] = TaskEngine(self.task,
            DefaultGpuConfiguration(gpu_config_512, trt_model, gpu_post_processing, num_streams),
            CpuConfiguration(cpu_config_512, onnx_model, cpu_post_processing)
        )
        self.task_engine.add_gpu_configuration(DefaultGpuConfiguration(
            gpu_config_416, trt_model, gpu_post_processing, num_streams
        ))
        self.task_engine.add_gpu_configuration(DefaultGpuConfiguration(
            gpu_config_320, trt_model, gpu_post_processing, num_streams
        ))

        self.task_engine.add_cpu_configuration(CpuConfiguration(
            cpu_config_416, onnx_model, cpu_post_processing
        ))
        self.task_engine.add_cpu_configuration(CpuConfiguration(
            cpu_config_320, onnx_model, cpu_post_processing
        ))
        self._fill_inputs()

    def _fill_inputs(self):
        datapath = Path("~/scheduler_inputs/Hand_Detection").expanduser()
        for file in datapath.iterdir():
            if file.is_file():
                image : np.ndarray | None = cv2.imread(file)
                if image is None:
                    raise ValueError("Could not load image:", file)
                self._inputs.append(image)

    def _generate_input_dictionary(self, input_array: MultidimensionalArray) -> Dict[str, MultidimensionalArray]:
        return {"input": input_array}

    def get_task_engine(self) -> TaskEngine:
        return self.task_engine

    @staticmethod
    def output_post_processing(outputs : torch.Tensor, job : TensorRtInferenceJob | OnnxInferenceJob):
        boxes = outputs[:, :4]
        scores = outputs[:, 4]
        # confidence
        boxes = boxes[scores > 0.25]
        scores = scores[scores > 0.25]
        # nms overlap is 0.5 since it was evaluated with AP@0.5
        keep = torch_nms(boxes, scores, iou_threshold=0.5)
        boxes = boxes[keep]
        scores = scores[keep]

        image_height, image_width = job.inputs.original_input[HandDetection.image_key].shape[:2]

        scale = torch.tensor([image_width, image_height, image_width, image_height], dtype=torch.float32, device=boxes.device)
        scaled_boxes = (boxes * scale).to(torch.int32)
        _ = torch.column_stack((scores, scaled_boxes))
    @staticmethod
    def gpu_post_processing(job : TensorRtInferenceJob):
        outputs = job.outputs["output"][0]
        HandDetection.output_post_processing(outputs, job)

    # adjusted from https://github.com/cansik/yolo-hand-detection
    @staticmethod
    def cpu_post_processing(job : OnnxInferenceJob):
        if job.status != OnnxInferenceJobStatus.FINISHED:
            return
        outputs = torch.as_tensor(job.outputs[0])[0]
        HandDetection.output_post_processing(outputs, job)
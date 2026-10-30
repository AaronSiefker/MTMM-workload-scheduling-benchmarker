from math import ceil
from pathlib import Path
from typing import Final, Dict, Callable, List

import cv2
import numpy as np
import torch
from torchvision.ops import nms

from scheduling.Scheduler import Scheduler
from task_processing.CpuConfiguration import CpuConfiguration
from task_processing.DefaultGpuConfiguration import DefaultGpuConfiguration
from task_processing.TaskConfiguration import TaskConfiguration
from task_processing.TaskEngine import TaskEngine
from task_processing.cpu_execution.OnnxInferenceJob import OnnxInferenceJob
from task_processing.cpu_execution.OnnxModel import OnnxModel
from task_processing.gpu_execution.TensorRtEngine import TensorRtEngine
from task_processing.gpu_execution.TensorRtInferenceJob import TensorRtInferenceJob
from task_processing.gpu_execution.TensorRtModel import TensorRtModel
from task_processing.input_preparation.ConfigurationInputConverter import ConfigurationInputConverter, \
    MultidimensionalArray
from tasks.InferenceTask import InferenceTask
from tasks.dependencies.Dependency import Dependency

# taken from https://github.com/biubug6/Pytorch_Retinaface with minor changes.
# Original license text:
# MIT License
#
# Copyright (c) 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}

# adjusted PriorBox from https://github.com/biubug6/Pytorch_Retinaface to work on gpu and cpu
class PriorBoxDevice:
    def __init__(self, cfg, image_size, device=None):
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.clip = cfg['clip']
        self.image_size = image_size
        self.device = device
        self.feature_maps = [[ceil(self.image_size[0] / step), ceil(self.image_size[1] / step)] for step in self.steps]

    def forward(self):
        im_height, im_width = self.image_size

        anchors_all = []

        for k, (f_h, f_w) in enumerate(self.feature_maps):

            step = self.steps[k]

            min_sizes = torch.tensor(self.min_sizes[k], dtype=torch.float32, device=self.device)

            y, x = torch.meshgrid(
                torch.arange(f_h, device=self.device),
                torch.arange(f_w, device=self.device),
                indexing='ij'
            )

            cx = (x.float() + 0.5) * step / im_width
            cy = (y.float() + 0.5) * step / im_height

            cx = cx.unsqueeze(-1)  # [H, W, 1]
            cy = cy.unsqueeze(-1)  # [H, W, 1]

            s_kx = (min_sizes / im_width).view(1, 1, -1)
            s_ky = (min_sizes / im_height).view(1, 1, -1)

            s_kx = s_kx.expand(f_h, f_w, -1)
            s_ky = s_ky.expand(f_h, f_w, -1)

            cx = cx.expand(-1, -1, len(min_sizes))
            cy = cy.expand(-1, -1, len(min_sizes))

            anchors = torch.stack([cx, cy, s_kx, s_ky], dim=-1)

            anchors_all.append(anchors.reshape(-1, 4))

        output = torch.cat(anchors_all, dim=0)

        if self.clip:
            output = output.clamp(0, 1)

        return output

# taken from https://github.com/biubug6/Pytorch_Retinaface
# Original license text:
# MIT License
#
# Copyright (c) 2019
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Original comment:
# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

class FaceDetection(InferenceTask):
    class FaceDetectionInputConverter(ConfigurationInputConverter):
        def __init__(self, image_key : str, final_size : int):
            self._image_key : Final[str] = image_key
            self._final_size : Final[int] = final_size

        def convert_input(self, model_input: Dict[str, MultidimensionalArray]) -> Dict[str, MultidimensionalArray]:
            image = np.asarray(model_input[self._image_key])
            if image.dtype != np.float32:
                image = image.astype(np.float32)
            height = int(image.shape[0])
            width = int(image.shape[1])
            if len(image.shape) == 4: # batched images
                if image.shape[0] > 1:
                    raise NotImplementedError("Did not implement convert_input for FaceDetectionInputConverter for batched images")
                height, width = image.shape[1:3]
                image = image[0]

            if height != self._final_size or width != self._final_size:
                image = cv2.resize(image, (self._final_size, self._final_size), interpolation=cv2.INTER_LINEAR)

            # do not need to letterbox here, since I letterboxed them in scheduler_inputs for maximum input size

            cv2.subtract(image, np.array([104, 117, 123], dtype=image.dtype), dst=image)
            image = image.transpose(2, 0, 1)
            image = np.expand_dims(image, axis=0)

            out = model_input.copy()
            out[self._image_key] = image
            return out

    image_key : Final[str] = "input"
    task_name : Final[str] = "Face Detection"
    def __init__(self, scheduler : Scheduler, relative_deadline, gpu_benchmark_results : Dict, cpu_benchmark_results : Dict,
                 num_streams : int,
                 dependencies : List[Dependency] | None = None):
        super().__init__(FaceDetection.task_name, scheduler, relative_deadline, dependencies)

        model_id : str = "mobilenetv1_retinaface"
        trt_engine_variant : str = "fp32"
        gpu_model_id : str = f"trt-{model_id}_{trt_engine_variant}"
        cpu_model_id : str = f"onnx-{model_id}_fp32"
        input_converter_size_1024 : ConfigurationInputConverter = FaceDetection.FaceDetectionInputConverter(FaceDetection.image_key, 1024)
        input_converter_size_832 : ConfigurationInputConverter = FaceDetection.FaceDetectionInputConverter(FaceDetection.image_key, 832)
        input_converter_size_640 : ConfigurationInputConverter = FaceDetection.FaceDetectionInputConverter(FaceDetection.image_key, 640)
        input_converter_size_576 : ConfigurationInputConverter = FaceDetection.FaceDetectionInputConverter(FaceDetection.image_key, 576)
        input_converter_size_512 : ConfigurationInputConverter = FaceDetection.FaceDetectionInputConverter(FaceDetection.image_key, 512)
        trt_engine_path : str = f"../TensorRT/Face_Detection/Retinaface_{trt_engine_variant}.engine"

        model_variant = "img_size=1024"
        gpu_config_1024 = TaskConfiguration.create(
            self.task, gpu_model_id, 0.232, model_variant, input_converter_size_1024,
            0.8807, gpu_benchmark_results, {"default": {1: 0.055374}}
        )
        model_variant = "img_size=832"
        gpu_config_832 = TaskConfiguration.create(
            self.task, gpu_model_id, 0.232, model_variant, input_converter_size_832,
            0.8669, gpu_benchmark_results, {"default": {1: 0.037096}}
        )
        model_variant = "img_size=640"
        gpu_config_640 = TaskConfiguration.create(
            self.task, gpu_model_id, 0.232, model_variant, input_converter_size_640,
            0.8303, gpu_benchmark_results, {"default": {1: 0.021598}}
        )
        model_variant = "img_size=576"
        gpu_config_576 = TaskConfiguration.create(
            self.task, gpu_model_id, 0.232, model_variant, input_converter_size_576,
            0.8110, gpu_benchmark_results, {"default": {1: 0.017751}}
        )
        model_variant = "img_size=512"
        gpu_config_512 = TaskConfiguration.create(
            self.task, gpu_model_id, 0.232, model_variant, input_converter_size_512,
            0.7839, gpu_benchmark_results, {"default": {1: 0.013918}}
        )
        onnx_model_path = "../OnnxModels/Face_Detection/PyTorch-RetinaFace_mobilenet_fp32.onnx"
        model_variant = "img_size=1024"
        cpu_config_1024 = TaskConfiguration.create(
            self.task, cpu_model_id, 0.0, model_variant, input_converter_size_1024,
            0.8807, cpu_benchmark_results, {}
        )
        model_variant = "img_size=832"
        cpu_config_832 = TaskConfiguration.create(
            self.task, cpu_model_id, 0.0, model_variant, input_converter_size_832,
            0.8669, cpu_benchmark_results, {}
        )
        model_variant = "img_size=640"
        cpu_config_640 = TaskConfiguration.create(
            self.task, cpu_model_id, 0.0, model_variant, input_converter_size_640,
            0.8303, cpu_benchmark_results, {}
        )
        model_variant = "img_size=576"
        cpu_config_576 = TaskConfiguration.create(
            self.task, cpu_model_id, 0.0, model_variant, input_converter_size_576,
            0.8110, cpu_benchmark_results, {}
        )
        model_variant = "img_size=512"
        cpu_config_512 = TaskConfiguration.create(
            self.task, cpu_model_id, 0.0, model_variant, input_converter_size_512,
            0.7839, cpu_benchmark_results, {}
        )

        trt_model = TensorRtModel(TensorRtEngine(trt_engine_path))
        onnx_model = OnnxModel(onnx_model_path)
        self.task_engine : Final[TaskEngine] = TaskEngine(
            self.task, DefaultGpuConfiguration(
                gpu_config_1024, trt_model, self.generate_gpu_post_processing(1024), num_streams
            ),
            CpuConfiguration(cpu_config_1024, onnx_model, self.generate_cpu_post_processing(1024))
        )
        self.task_engine.add_gpu_configuration(DefaultGpuConfiguration(
            gpu_config_832, trt_model, self.generate_gpu_post_processing(832), num_streams
        ))
        self.task_engine.add_gpu_configuration(DefaultGpuConfiguration(
            gpu_config_640, trt_model, self.generate_gpu_post_processing(640), num_streams
        ))
        self.task_engine.add_gpu_configuration(DefaultGpuConfiguration(
            gpu_config_576, trt_model, self.generate_gpu_post_processing(576), num_streams
        ))
        self.task_engine.add_gpu_configuration(DefaultGpuConfiguration(
            gpu_config_512, trt_model, self.generate_gpu_post_processing(512), num_streams
        ))

        self.task_engine.add_cpu_configuration(CpuConfiguration(
            cpu_config_832, onnx_model, self.generate_cpu_post_processing(832)
        ))
        self.task_engine.add_cpu_configuration(CpuConfiguration(
            cpu_config_640, onnx_model, self.generate_cpu_post_processing(640)
        ))
        self.task_engine.add_cpu_configuration(CpuConfiguration(
            cpu_config_576, onnx_model, self.generate_cpu_post_processing(576)
        ))
        self.task_engine.add_cpu_configuration(CpuConfiguration(
            cpu_config_512, onnx_model, self.generate_cpu_post_processing(512)
        ))
        self._fill_inputs()

    def _fill_inputs(self):
        datapath = Path("~/scheduler_inputs/Face_Detection").expanduser()
        for file in datapath.iterdir():
            if file.is_file():
                image : np.ndarray | None = cv2.imread(file, cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError("Could not load image:", file)
                self._inputs.append(image)


    def _generate_input_dictionary(self, input_array: MultidimensionalArray) -> Dict[str, MultidimensionalArray]:
        return {"input": input_array}

    def get_task_engine(self) -> TaskEngine:
        return self.task_engine

    # taken from https://github.com/biubug6/Pytorch_Retinaface with minor changes.
    # Original license text:
    # MIT License
    #
    # Copyright (c) 2019
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy
    # of this software and associated documentation files (the "Software"), to deal
    # in the Software without restriction, including without limitation the rights
    # to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    # copies of the Software, and to permit persons to whom the Software is
    # furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all
    # copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    # IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    # FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    # AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    # LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    # OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    # SOFTWARE.
    @staticmethod
    def post_processing(loc : torch.Tensor, conf : torch.Tensor, final_size : int, original_image : MultidimensionalArray,
                        device : torch.device):
        # assumes original image was loaded via cv2
        original_height, original_width = original_image.shape[:2]

        priorbox = PriorBoxDevice(cfg_mnet, image_size=(final_size, final_size), device=device)
        priors = priorbox.forward()

        priors = priors.to(device)
        prior_data = priors.data
        # variances taken from cfg_mnet in https://github.com/biubug6/Pytorch_Retinaface
        boxes = decode(loc.data.squeeze(0), prior_data, [0.1, 0.2])

        scale = final_size / max(original_height, original_width)
        boxes = boxes / scale

        scores = conf.squeeze(0)[:, 1]

        # ignore low scores
        # scores > confidence threshold
        inds = torch.where(scores > 0.4)[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # iou_threshold is pytorch retinaface default
        keep = nms(boxes, scores, iou_threshold=0.4)
        boxes = boxes[keep]
        scores = scores[keep]

        # keep top k
        _ = boxes[:50]
        _ = scores[:50]

    def generate_gpu_post_processing(self, final_size : int) -> Callable[[TensorRtInferenceJob], None]:
        def gpu_post_processing(job : TensorRtInferenceJob) -> None:
            with torch.cuda.stream(job.stream):
                loc = job.outputs["loc"]
                conf = job.outputs["conf"]
                image = job.inputs.original_input[FaceDetection.image_key]
                FaceDetection.post_processing(loc, conf, final_size, image, torch.device("cuda"))
        return self.finalize_gpu_post_processing(gpu_post_processing)

    def generate_cpu_post_processing(self, final_size : int) -> Callable[[OnnxInferenceJob], None]:
        def cpu_post_processing(job : OnnxInferenceJob):
            loc = torch.as_tensor(job.outputs[0])
            conf = torch.as_tensor(job.outputs[1])
            image = job.inputs.original_input[FaceDetection.image_key]
            FaceDetection.post_processing(loc, conf, final_size, image, torch.device("cpu"))
        return self.finalize_cpu_post_processing(cpu_post_processing)
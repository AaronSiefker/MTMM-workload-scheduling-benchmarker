import wave
from pathlib import Path
from typing import Final, Dict, List

import numpy as np
import torch
from torchaudio import transforms

from scheduling.Scheduler import Scheduler
from task_processing.CpuConfiguration import CpuConfiguration
from task_processing.EarlyExitGpuConfiguration import EarlyExitGpuConfiguration
from task_processing.TaskConfiguration import TaskConfiguration
from task_processing.TaskEngine import TaskEngine
from task_processing.cpu_execution.OnnxModel import OnnxModel
from task_processing.gpu_execution.TensorRtEngine import TensorRtEngine
from task_processing.gpu_execution.TensorRtModel import TensorRtModel
from task_processing.input_preparation.ComposedConverter import ComposedConverter
from task_processing.input_preparation.ConfigurationInputConverter import ConfigurationInputConverter, \
    MultidimensionalArray
from task_processing.input_preparation.ThresholdConverter import ThresholdConverter
from tasks.InferenceTask import InferenceTask
from tasks.dependencies.Dependency import Dependency


class KeywordSpotting(InferenceTask):
    class KwsInputConverter(ConfigurationInputConverter):
        def __init__(self, audio_key : str):
            self._audio_key : Final[str] = audio_key
            self._mel_transform = transforms.MelSpectrogram(
                sample_rate=16000,
                n_fft=400,
                hop_length=160,
                n_mels=40
            )
            self._db_transform = _db_transform = transforms.AmplitudeToDB()

        def convert_input(self, model_input: Dict[str, MultidimensionalArray]) -> Dict[str, MultidimensionalArray]:
            waveform = torch.as_tensor(model_input[self._audio_key])

            # average over channels
            # samples only have one channel
            #waveform = waveform.mean(dim=0, keepdim=True)

            features = self._mel_transform(waveform)
            features = self._db_transform(features)

            # Per-example normalization
            features = (features - features.mean()) / (features.std() + 1e-6)

            out = model_input.copy()
            out[self._audio_key] = features
            return out

    task_name : Final[str] = "Keyword Spotting"
    def __init__(self, scheduler : Scheduler, relative_deadline, gpu_benchmark_results : Dict, cpu_benchmark_results : Dict,
                 num_streams : int,
                 dependencies : List[Dependency] | None = None):
        super().__init__(KeywordSpotting.task_name, scheduler, relative_deadline, dependencies)
        self._default_early_exit_threshold = torch.Tensor([0.85])

        model_id = "early_exit_resnet"
        trt_engine_variant = "fp16"
        gpu_model_id = f"trt-{model_id}_{trt_engine_variant}"
        cpu_model_id = f"onnx-{model_id}_fp32"

        kws_input_converter : ConfigurationInputConverter = KeywordSpotting.KwsInputConverter("input")
        threshold_key = "threshold"
        input_converter_threshold_090 = ComposedConverter(
            [kws_input_converter, ThresholdConverter(threshold_key, 0.90)]
        )
        input_converter_threshold_065 = ComposedConverter(
            [kws_input_converter, ThresholdConverter(threshold_key, 0.65)]
        )
        input_converter_threshold_045 = ComposedConverter(
            [kws_input_converter, ThresholdConverter(threshold_key, 0.45)]
        )
        input_converter_threshold_035 = ComposedConverter(
            [kws_input_converter, ThresholdConverter(threshold_key, 0.35)]
        )
        trt_early_engine_path = f"../TensorRT/Keyword_Spotting/ResNetEarlyExit_EarlyBranch_{trt_engine_variant}.engine"
        trt_late_engine_path = f"../TensorRT/Keyword_Spotting/ResNetEarlyExit_LateBranch_{trt_engine_variant}.engine"
        model_variant = "threshold=0.95"
        energy_statistics = {"early_exit": {1: 0.000755}, "late_exit": {1: (0.000755 + 0.000485)}}
        gpu_config_090 = TaskConfiguration.create(
            self.task, gpu_model_id, 0.347, model_variant, input_converter_threshold_090,
            0.9540, gpu_benchmark_results,
            energy_statistics,
            false_positive_probability=0.022460776218001653
        )
        model_variant = "threshold=0.65"
        gpu_config_065 = TaskConfiguration.create(
            self.task, gpu_model_id, 0.347, model_variant, input_converter_threshold_065,
            0.9347, gpu_benchmark_results, energy_statistics,
            false_positive_probability=0.054616588419405324
        )
        model_variant = "threshold=0.45"
        gpu_config_045 = TaskConfiguration.create(
            self.task, gpu_model_id, 0.347, model_variant, input_converter_threshold_045,
            0.8843, gpu_benchmark_results, energy_statistics,
            false_positive_probability=0.11310156368604828
        )
        onnx_model_path = f"../OnnxModels/Keyword_Spotting/kws_widened_40_mels_30_fp32.onnx"
        cpu_config_090 = TaskConfiguration.create(
            self.task, cpu_model_id, 0.0, f"threshold=0.95", input_converter_threshold_090,
            0.9541, cpu_benchmark_results, {}, false_positive_probability=0.022457067371202115
        )
        cpu_config_065 = TaskConfiguration.create(
            self.task, cpu_model_id, 0.0, f"threshold=0.65", input_converter_threshold_065,
            0.9346, cpu_benchmark_results, {}, false_positive_probability=0.054625136954139925
        )
        cpu_config_035 = TaskConfiguration.create(
            self.task, cpu_model_id, 0.0, f"threshold=0.35", input_converter_threshold_035,
            0.8527, cpu_benchmark_results, {}, false_positive_probability=0.14772384684956286
        )
        early_trt_model = TensorRtModel(TensorRtEngine(trt_early_engine_path))
        late_trt_model = TensorRtModel(TensorRtEngine(trt_late_engine_path))
        onnx_model = OnnxModel(onnx_model_path)
        gpu_post_processing = self.finalize_gpu_post_processing(lambda job: None)
        cpu_post_processing = self.finalize_cpu_post_processing(lambda job: None)

        intermediate_features_key = "x"
        intermediate_prediction_key = "prediction"
        intermediate_confidence_key = "confidence"
        self.task_engine : Final[TaskEngine] = TaskEngine(self.task,
            EarlyExitGpuConfiguration(
                gpu_config_090, early_trt_model,
                intermediate_features_key=intermediate_features_key,
                intermediate_prediction_key=intermediate_prediction_key,
                intermediate_confidence_key=intermediate_confidence_key,
                input_threshold_key=threshold_key,
                late_trt_model=late_trt_model,
                post_processing_callback=gpu_post_processing,
                num_streams=num_streams
            ),
            CpuConfiguration(cpu_config_090, onnx_model, cpu_post_processing)
        )
        self.task_engine.add_gpu_configuration(
            EarlyExitGpuConfiguration(
                gpu_config_065, early_trt_model,
                intermediate_features_key=intermediate_features_key,
                intermediate_prediction_key=intermediate_prediction_key,
                intermediate_confidence_key=intermediate_confidence_key,
                input_threshold_key=threshold_key,
                late_trt_model=late_trt_model,
                post_processing_callback=gpu_post_processing,
                num_streams=num_streams
            )
        )
        self.task_engine.add_gpu_configuration(
            EarlyExitGpuConfiguration(
                gpu_config_045, early_trt_model,
                intermediate_features_key=intermediate_features_key,
                intermediate_prediction_key=intermediate_prediction_key,
                intermediate_confidence_key=intermediate_confidence_key,
                input_threshold_key=threshold_key,
                late_trt_model=late_trt_model,
                post_processing_callback=gpu_post_processing,
                num_streams=num_streams
            )
        )
        self.task_engine.add_cpu_configuration(
            CpuConfiguration(cpu_config_065, onnx_model, cpu_post_processing)
        )
        self.task_engine.add_cpu_configuration(
            CpuConfiguration(cpu_config_035, onnx_model, cpu_post_processing)
        )
        self._fill_inputs()

    def _fill_inputs(self):
        def load_wav(path):
            with wave.open(path, "rb") as wav_file:
                num_frames = wav_file.getnframes()
                audio_bytes = wav_file.readframes(num_frames)

            audio = np.frombuffer(audio_bytes, dtype=np.int16)
            audio = audio.astype(np.float32) / 32768.0
            waveform = np.expand_dims(audio, axis=0)

            return waveform

        datapath = Path("~/scheduler_inputs/Keyword_Spotting").expanduser()
        for file in datapath.iterdir():
            if file.is_file():
                # converting to string, since wave.open does not work on the path object
                audio_input = load_wav(str(file))
                self._inputs.append(np.array(audio_input))

    def _generate_input_dictionary(self, input_array: MultidimensionalArray) -> Dict[str, MultidimensionalArray]:
        return {"input": input_array, "threshold": self._default_early_exit_threshold}

    def get_task_engine(self) -> TaskEngine:
        return self.task_engine
from __future__ import annotations

import numpy as np
import onnxruntime
from onnxruntime import InferenceSession, RunOptions

from task_processing.InputContainer import InputContainer
from task_processing.TaskConfiguration import TaskConfiguration
from task_processing.cpu_execution.OnnxInferenceJob import OnnxInferenceJob, OnnxInferenceJobStatus


class OnnxModel:
    def __init__(self, onnx_filepath : str):
        self._session : InferenceSession = InferenceSession(onnx_filepath)

    def prepare_inference_job(self, request_id : str, associated_config : TaskConfiguration,
                              input_container : InputContainer[np.ndarray], request_time : float, deadline : float,
                              run_options : RunOptions) -> OnnxInferenceJob:
        return OnnxInferenceJob(request_id, associated_config, self, input_container, request_time, deadline,
                                run_options, OnnxInferenceJobStatus.PREPARED, [])

    def infer(self, inference_job : OnnxInferenceJob):
        if inference_job.onnx_model is not self:
            raise ValueError("Error, the provided inference job was not prepared using this OnnxModel")

        inference_job.status = OnnxInferenceJobStatus.IN_PROGRESS
        try:
            inference_job.outputs = self._session.run(None, inference_job.inputs.converted_input,
                                                      run_options=inference_job.run_options)
            inference_job.status = OnnxInferenceJobStatus.FINISHED
        # This gets thrown if job.terminate = True is set on another thread
        # reference couldn't resolve properly => ignored warning
        except onnxruntime.capi.onnxruntime_pybind11_state.Fail:
            inference_job.status = OnnxInferenceJobStatus.TERMINATED
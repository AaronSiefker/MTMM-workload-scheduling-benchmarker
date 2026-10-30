from __future__ import annotations

import torch

from task_processing.InputContainer import InputContainer
from task_processing.TaskConfiguration import TaskConfiguration
from task_processing.gpu_execution.TensorRtEngine import TensorRtEngine
import tensorrt as trt

from task_processing.gpu_execution.TensorRtInferenceJob import TensorRtInferenceJob


class TensorRtModel:
    def __init__(self, trt_engine : TensorRtEngine):
        self._trt_engine : TensorRtEngine = trt_engine
        self._context : trt.tensorrt.IExecutionContext = trt_engine.create_execution_context()

    @torch.no_grad()
    def prepare_inference_job(self, request_id : str, associated_config : TaskConfiguration,
                              input_container : InputContainer[torch.Tensor], batch_size : int, request_time : float, deadline : float,
                              stream: torch.cuda.Stream, stream_index : int) -> TensorRtInferenceJob:
        """
        Creates an inference job for the provided input and fully prepares self._context for
        the execution of said created inference job.
        NOTE: This method takes ownership of the provided inputs, if you require them after calling this method,
        copy them beforehand
        :param batch_size: The input batch size.
        :param stream_index: The cuda stream index in the strategy preparing this request.
        :param request_time: The time the request was generated.
        :param deadline: The deadline associated with the request.
        :param request_id: The id associated with the request.
        :param associated_config: The config for this inference.
        :param input_container: An InputContainer containing inference inputs in form of a dictionary mapping the input names of the trt engine
        this was created for to their concrete inputs in form of torch.Tensors.
        :param stream: The torch.cuda.Stream the inference job should use.
        :return: The created TensorRTInferenceJob, ready for execution
        """
        with torch.cuda.stream(stream):
            inputs = input_container.converted_input
            for input_name in self._trt_engine.input_names:
                if input_name not in inputs:
                    raise ValueError(f"Input tensor '{input_name}' does not exist")
                inputs[input_name] = inputs[input_name].to("cuda", non_blocking=True)
                self._context.set_input_shape(input_name, tuple(inputs[input_name].shape))
                self._context.set_tensor_address(input_name, inputs[input_name].data_ptr())

            outputs = {}
            for output_name in self._trt_engine.output_names:
                shape = tuple(self._context.get_tensor_shape(output_name))
                datatype = self._trt_engine.output_datatypes[output_name]
                output = torch.empty(size=shape, dtype=datatype, device="cuda")
                self._context.set_tensor_address(output_name, output.data_ptr())
                outputs[output_name] = output
            event = torch.cuda.Event()
            return TensorRtInferenceJob(request_id, associated_config, self, stream, stream_index,
                                        input_container, batch_size, request_time, deadline, event, outputs)

    def infer(self, inference_job : TensorRtInferenceJob) -> bool:
        """
        Schedules the provided job for inference on the cuda stream provided to self.prepare_inference_job.
        After scheduling, an event is recorded on said stream which can be used to determine
        if the job has finished execution.

        A ValueError is raised if the provided job was not prepared for this TensorRtModel

        :param inference_job: An inference job created using self.prepare_inference_job()
        :return: The return value of execute_async_v3. Note that this is already true, if the job could be
        scheduled successfully, it does not indicate whether the inference itself was successful
        """
        if inference_job.tensor_rt_model is not self:
            raise ValueError("The provided inference job was not prepared for this TensorRtModel")
        success = self._context.execute_async_v3(inference_job.stream.cuda_stream)
        inference_job.event.record(inference_job.stream)

        return success

    @staticmethod
    def copy_with_new_context(model : TensorRtModel) -> TensorRtModel:
        return TensorRtModel(model._trt_engine)
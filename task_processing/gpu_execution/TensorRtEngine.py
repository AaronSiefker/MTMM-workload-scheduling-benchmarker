import tensorrt as trt
import torch

from typing import Dict, List


class TensorRtEngine:
    # Ignored unresolved references of trt, since they aren't properly found when mapping packages into venv
    logger = trt.Logger(trt.Logger.WARNING)

    def __init__(self, engine_path: str):
        self._engine_path = engine_path
        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        runtime = trt.Runtime(TensorRtEngine.logger)
        self._engine = runtime.deserialize_cuda_engine(engine_bytes)

        input_names : List[str] = []
        output_names : List[str] = []

        self.output_datatypes : Dict[str, torch.dtype] = {}

        for i in range(self._engine.num_io_tensors):
            tensor_name = self._engine.get_tensor_name(i)
            mode = self._engine.get_tensor_mode(tensor_name)

            if mode == trt.TensorIOMode.INPUT:
                input_names.append(tensor_name)
            elif mode == trt.TensorIOMode.OUTPUT:
                output_names.append(tensor_name)

        if len(input_names) <= 0:
            raise RuntimeError(f"Error, the provided engine '{engine_path}' does not contain any input tensors")
        if len(output_names) <= 0:
            raise RuntimeError("Error, the provided engine '{engine_path}' does not contain any output tensors")

        self.input_names : tuple[str, ...] = tuple(input_names)
        self.output_names : tuple[str, ...] = tuple(output_names)

        for output_name in self.output_names:
            trt_to_torch = {
                trt.DataType.FLOAT: torch.float32,
                trt.DataType.HALF: torch.float16,
                trt.DataType.BF16: torch.bfloat16,
                trt.DataType.INT64: torch.int64,
                trt.DataType.INT32: torch.int32,
                trt.DataType.INT8: torch.int8,
                trt.DataType.BOOL: torch.bool,
            }
            datatype = trt_to_torch[self._engine.get_tensor_dtype(output_name)]
            self.output_datatypes[output_name] = datatype

    def create_execution_context(self) -> trt.tensorrt.IExecutionContext:
        return self._engine.create_execution_context()
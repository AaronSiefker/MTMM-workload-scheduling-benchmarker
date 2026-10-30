from dataclasses import dataclass
from typing import Dict, Generic, TypeVar

from task_processing.input_preparation.ConfigurationInputConverter import MultidimensionalArray

T = TypeVar("T", bound=MultidimensionalArray)
@dataclass(frozen=True)
class InputContainer(Generic[T]):
    original_input : Dict[str, MultidimensionalArray]
    converted_input : Dict[str, T]
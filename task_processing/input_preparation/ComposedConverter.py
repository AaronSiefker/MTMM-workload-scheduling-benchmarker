from typing import Dict, Sequence

from task_processing.input_preparation.ConfigurationInputConverter import ConfigurationInputConverter, \
    MultidimensionalArray


class ComposedConverter(ConfigurationInputConverter):

    def __init__(self, converters : Sequence[ConfigurationInputConverter]):
        self._converters = list(converters)

    def convert_input(self, model_input: Dict[str, MultidimensionalArray]) -> Dict[str, MultidimensionalArray]:
        if len(self._converters) == 0:
            return model_input.copy()
        out = model_input
        for converter in self._converters:
            out = converter.convert_input(out)
        return out
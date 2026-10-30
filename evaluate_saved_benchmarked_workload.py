import json
from pathlib import Path
from pprint import pprint

import numpy as np

from benchmark_workload import create_benchmarking_results, save_benchmark_results


def _lists_to_arrays(obj):
    if isinstance(obj, list):
        return np.array([_lists_to_arrays(x) for x in obj])
    elif isinstance(obj, dict):
        return {k: _lists_to_arrays(v) for k, v in obj.items()}
    return obj

def main():
    base_dir = Path("./energy_scheduling_results/considered/testing")
    benchmark_name = "default"
    benchmark_dir = base_dir / benchmark_name

    with open(benchmark_dir / "preparation_results.json", "r") as f:
        preparation_results = _lists_to_arrays(json.load(f))

    with open(benchmark_dir / "inference_results.json", "r") as f:
        inference_results = _lists_to_arrays(json.load(f))

    benchmarking_results = create_benchmarking_results(preparation_results, inference_results)
    pprint(benchmarking_results)
    # noinspection PyTypeChecker
    save_benchmark_results(benchmark_name, benchmarking_results, preparation_results, inference_results, exist_ok=True, results_directory=base_dir)

if __name__ == "__main__":
    main()
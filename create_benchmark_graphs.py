import json
import os
from pathlib import Path

import numpy as np
import plt


def main():
    results_directory = Path("./benchmarking_results")
    benchmark_name = Path("fixed_freq/default-gpu_only-SCHED_FIFO")
    concrete_results_directory = results_directory / benchmark_name

    with open(concrete_results_directory / "inference_results.json", "r") as f:
        inference_results = json.load(f)

    for task in inference_results:
        for model_description in inference_results[task]:
            current_results = inference_results[task][model_description]
            flattened_results = [x for sublist in current_results for x in sublist]
            percentile_98th = np.percentile(flattened_results, q=98)
            # filtering out obscene outliers to make graph readable and removing slots for executions that didn't happen
            flattened_results = [result for result in flattened_results if result <= percentile_98th and result != 0.0]
            inference_results[task][model_description] = flattened_results

    # for task in inference_results:
    #     if task != "Face Detection" and task != "Hand Detection" and task != "Image Classification":
    #         continue
    #     plt.figure(1)
    #     for model_description in inference_results[task]:
    #         model_name, input_description, batch_size = model_description.split("@")
    #         flattened_results = inference_results[task][model_description]
    #         print(task)
    #         print(min(flattened_results))
    #         if input_description.split("=")[0] == "img_size":
    #             input_description = f"{input_description.split("=")[1]}×{input_description.split("=")[1]}"
    #         plt.hist(flattened_results, bins=50, alpha=0.7, label=input_description)
    #         plt.xlabel("Inference Time (ms)")
    #         plt.ylabel("Count")
    #     plt.title(task)
    #     if task != "Image Classification":
    #         plt.legend()
    #
    #     if not os.path.exists("images"):
    #         os.mkdir("images")
    #
    #     plt.savefig(Path("images") / f"{task}.svg")
    #     plt.close()

    keyword_spotting_results = inference_results["Keyword Spotting"]

    plt.figure(1)
    plt.xlabel("Inference Time (ms)")
    plt.ylabel("Count")
    plt.title("Keyword Spotting")
    for model_description in keyword_spotting_results:
        model_name, input_description, batch_size = model_description.split("@")
        flattened_results = keyword_spotting_results[model_description]
        print(input_description)
        plt.hist(flattened_results, bins=50, alpha=0.7, label=input_description)

    plt.legend()
    plt.savefig(Path("images") / f"Keyword Spotting - all thresholds.svg")
    plt.close()

    # gesture_classification_results = inference_results["Gesture Classification"]
    # for model_description in gesture_classification_results:
    #     model_name, input_description, batch_size = model_description.split("@")
    #     batch_size = int(batch_size.split("=")[1])
    #     plt.figure(batch_size)
    #
    #     input_description = f"{input_description.split("=")[1]}×{input_description.split("=")[1]}"
    #     flattened_results = gesture_classification_results[model_description]
    #     plt.hist(flattened_results, bins=50, alpha=0.7, label=input_description)
    #     plt.xlabel("Inference Time (ms)")
    #     plt.ylabel("Count")
    #
    # plt.figure(1)
    # plt.legend()
    # plt.title(f"Gesture Classification, batch_size=1")
    # plt.savefig(Path("images") / "Gesture Classification batch_size=1.svg")
    # plt.close()
    # plt.figure(2)
    # plt.legend()
    # plt.title(f"Gesture Classification, batch_size=2")
    # plt.savefig(Path("images") / "Gesture Classification batch_size=2.svg")
    # plt.close()
    # plt.figure(3)
    # plt.legend()
    # plt.title(f"Gesture Classification, batch_size=3")
    # plt.savefig(Path("images") / "Gesture Classification batch_size=3.svg")
    # plt.close()
    # plt.figure(4)
    # plt.legend()
    # plt.title(f"Gesture Classification, batch_size=4")
    # plt.savefig(Path("images") / "Gesture Classification batch_size=4.svg")
    # plt.close()

if __name__ == "__main__":
    main()
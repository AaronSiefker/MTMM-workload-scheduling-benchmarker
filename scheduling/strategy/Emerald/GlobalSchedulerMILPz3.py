import json

from z3 import *

from scheduling.Scheduler import NullScheduler
from task_processing.GpuConfiguration import GpuConfiguration
from tasks.FaceDetection import FaceDetection
from tasks.GestureClassification import GestureClassification
from tasks.HandDetection import HandDetection
from tasks.ImageClassification import ImageClassification
from tasks.KeywordSpotting import KeywordSpotting


def _constrain_0_or_1_and_one_selected(optimize, cx):
    for c in cx:
        optimize.add(c >= 0)
        optimize.add(c <= 1)
    optimize.add(z3.Sum(cx) == 1)

def _get_inference_time(gpu_config : GpuConfiguration):
    inference_times_by_batch_size = gpu_config.task_configuration.static_98th_percentile_inference_time
    # for Default Workload only relevant for Gesture Classification, where batch_size=4 isn't much slower than batch_size=1
    max_batch_size = max(inference_times_by_batch_size)
    return z3.RealVal(inference_times_by_batch_size[max_batch_size])


# noinspection PyPep8Naming
def _solve_default_workload(num_streams : int, is_normal_period : bool):
    optimize = Optimize()
    period_length = 1.0 / 30 if is_normal_period else 1.0 / 35
    m = 1.0 / (2 * period_length)
    kws_1_deadline = period_length / 2
    m_kws_1 = 1.0 / (2 * kws_1_deadline)

    if num_streams == 1:
        used_gpu_benchmark = "/home/aaron/jetson-benchmarking_results/fixed_freq/default-gpu_only-SCHED_FIFO.json"
    elif num_streams == 2:
        used_gpu_benchmark = "/home/aaron/jetson-benchmarking_results/fixed_freq/default-gpu_only-background_gpu_benchmark-SCHED_FIFO.json"
    else:
        raise ValueError("Error, currently only supports 1 or 2 streams")

    with open(used_gpu_benchmark) as f:
        gpu_benchmark_results = json.load(f)
    used_cpu_benchmark = "benchmarking_results/fixed_freq/default.json"
    with open(used_cpu_benchmark) as f:
        cpu_benchmark_results = json.load(f)

    null_scheduler = NullScheduler()
    face_detection_configs = (
        FaceDetection(null_scheduler, 0, gpu_benchmark_results, cpu_benchmark_results, num_streams).task_engine.gpu_configurations)
    hand_detection_configs = (
        HandDetection(null_scheduler, 0, gpu_benchmark_results, cpu_benchmark_results, num_streams).task_engine.gpu_configurations)
    gesture_classification_configs = (
        GestureClassification(null_scheduler, 0, gpu_benchmark_results, cpu_benchmark_results, num_streams).task_engine.gpu_configurations)
    keyword_spotting_configs = (
        KeywordSpotting(null_scheduler, 0, gpu_benchmark_results, cpu_benchmark_results, num_streams).task_engine.gpu_configurations)
    image_classification_configs = (
        ImageClassification(null_scheduler, 0, gpu_benchmark_results, cpu_benchmark_results, num_streams).task_engine.gpu_configurations)

    # model scales
    # Face Detection
    c0 = [Int(f"c0{l}") for l in range(5)]
    _constrain_0_or_1_and_one_selected(optimize, c0)
    # Hand Detection
    c1 = [Int(f"c1{l}") for l in range(3)]
    _constrain_0_or_1_and_one_selected(optimize, c1)
    # Gesture Classification
    c2 = [Int(f"c2{l}") for l in range(3)]
    _constrain_0_or_1_and_one_selected(optimize, c2)

    # KWS 1
    c3 = [Int(f"c3{l}") for l in range(3)]
    _constrain_0_or_1_and_one_selected(optimize, c3)
    # KWS 2
    c4 = [Int(f"c4{l}") for l in range(3)]
    _constrain_0_or_1_and_one_selected(optimize, c4)

    # Image Classification 1
    c5 = [Int(f"c5{l}") for l in range(1)]
    _constrain_0_or_1_and_one_selected(optimize, c5)

    # Image Classification 2
    c6 = [Int(f"c6{l}") for l in range(1)]
    _constrain_0_or_1_and_one_selected(optimize, c6)

    # batch assignment variables (constant batch sized => equal to assignment variables for me)
    b = [[Int(f"b{i}{j}") for j in range(num_streams)] for i in range(7)]
    for task_processor_assignment_variables in b:
        _constrain_0_or_1_and_one_selected(optimize, task_processor_assignment_variables)

    # start times
    s = [[Real(f"s{i}{j}") for j in range(num_streams)] for i in range(7)]
    # finish times
    f = [[Real(f"f{i}{j}") for j in range(num_streams)] for i in range(7)]
    # start times constraints - immediately available
    for i in (0, 1, 3, 6):
        for j in range(num_streams):
            optimize.add(s[i][j] >= 0)
    for j1 in range(num_streams):
        for j2 in range(num_streams):
            # start time - Gesture Classification
            optimize.add(s[2][j1] >= f[1][j2])
    # start time - KWS 2
    for j in range(num_streams):
        optimize.add(s[4][j] >= (period_length / 2)) # second KWS becomes available after half the period
    # start time - Image Classification 1
    for j1 in range(num_streams):
        for j2 in range(num_streams):
            optimize.add(s[5][j1] >= f[3][j2])

    # execution times - Face Detection
    E0 = [Real(f"E0{j}") for j in range(num_streams)]
    for j in range(num_streams):
        optimize.add(E0[j] == (b[0][j] * (c0[0] * _get_inference_time(face_detection_configs[0]) +
                                          c0[1] * _get_inference_time(face_detection_configs[1]) +
                                          c0[2] * _get_inference_time(face_detection_configs[2]) +
                                          c0[3] * _get_inference_time(face_detection_configs[3]) +
                                          c0[4] * _get_inference_time(face_detection_configs[4])
                                          )
                               )
                     )
    # execution times - Hand Detection
    E1 = [Real(f"E1{j}") for j in range(num_streams)]
    for j in range(num_streams):
        optimize.add(E1[j] == (b[1][j] * (c1[0] * _get_inference_time(hand_detection_configs[0]) +
                                          c1[1] * _get_inference_time(hand_detection_configs[1]) +
                                          c1[2] * _get_inference_time(hand_detection_configs[2])
                                          )
                               )
                     )
    # execution times - Gesture Classification
    E2 = [Real(f"E2{j}") for j in range(num_streams)]
    for j in range(num_streams):
        optimize.add(E2[j] == (b[2][j] * (c2[0] * _get_inference_time(gesture_classification_configs[0]) +
                                          c2[1] * _get_inference_time(gesture_classification_configs[1]) +
                                          c2[2] * _get_inference_time(gesture_classification_configs[2])
                                          )
                               )
                     )
    # execution times - KWS 1
    E3 = [Real(f"E3{j}") for j in range(num_streams)]
    for j in range(num_streams):
        optimize.add(E3[j] == (b[3][j] * (c3[0] * _get_inference_time(keyword_spotting_configs[0]) +
                                          c3[1] * _get_inference_time(keyword_spotting_configs[1]) +
                                          c3[2] * _get_inference_time(keyword_spotting_configs[2])
                                          )
                               )
                     )
    # execution times - KWS 2
    E4 = [Real(f"E4{j}") for j in range(num_streams)]
    for j in range(num_streams):
        optimize.add(E4[j] == (b[4][j] * (c4[0] * _get_inference_time(keyword_spotting_configs[0]) +
                                          c4[1] * _get_inference_time(keyword_spotting_configs[1]) +
                                          c4[2] * _get_inference_time(keyword_spotting_configs[2])
                                          )
                               )
                     )
    # execution times - Image Classification 1
    E5 = [Real(f"E5{j}") for j in range(num_streams)]
    for j in range(num_streams):
        optimize.add(E5[j] == b[5][j] * c5[0] * _get_inference_time(image_classification_configs[0]))
    # execution times - Image Classification 2
    E6 = [Real(f"E6{j}") for j in range(num_streams)]
    for j in range(num_streams):
        optimize.add(E6[j] == b[6][j] * c6[0] * _get_inference_time(image_classification_configs[0]))

    # constrain finish_time on start_time
    E = [E0, E1, E2, E3, E4, E5, E6]
    for i in range(7):
        for j in range(num_streams):
            optimize.add(f[i][j] == s[i][j] + E[i][j])

    # only one task at once per stream
    for j in range(num_streams):
        for i1 in range(7):
            for i2 in range(7):
                if i2 == i1:
                    continue
                optimize.add(z3.Or(s[i1][j] >= f[i2][j], s[i2][j] >= f[i1][j]))

    # absolute finish times
    tau = [Real(f"tau{i}") for i in range(7)]
    # limiting to max is of all finish_times is kinda useless, since finish_time may arbitrarily be higher on the other stream
    for j in range(num_streams):
        for i in range(7):
            optimize.add(tau[i] >= f[i][j])

    # Latency Utilities
    UL = [Real(f"UL{i}") for i in range(7)]
    # Latency Utilities - period end (no Hand Detection, implicitly included, and no KWS 1) (with U_max = 1)
    for i in (0, 2, 4, 5, 6): # We do not need 1, since Hand Detection is implicitly included in Gesture Classification
        optimize.add(UL[i] == z3.If(
            tau[i] <= period_length,
            1,
            z3.If(
                tau[i] <= (3 * period_length),
                1 - m * (tau[i] - period_length),
                0
            )
        ))
    # Latency Utility - KWS 1
    optimize.add(UL[3] == z3.If(
            tau[3] <= kws_1_deadline,
            1,
            z3.If(
                tau[3] <= (3 * kws_1_deadline),
                1 - m_kws_1 * (tau[3] - kws_1_deadline),
                0
            )
        ))
    UL_avg = Real("UL_avg")
    optimize.add(UL_avg == (UL[0] + UL[2] + UL[3] + UL[4] + UL[5] + UL[6]) / 6.0)

    # Accuracy Utilities
    UA = [Real(f"UA{i}") for i in range(7)]
    optimize.add(UA[0] == (c0[0] * face_detection_configs[0].task_configuration.quality +
             c0[1] * face_detection_configs[1].task_configuration.quality +
             c0[2] * face_detection_configs[2].task_configuration.quality +
             c0[3] * face_detection_configs[3].task_configuration.quality +
             c0[4] * face_detection_configs[4].task_configuration.quality))

    optimize.add(UA[1] == (c1[0] * hand_detection_configs[0].task_configuration.quality +
             c1[1] * hand_detection_configs[1].task_configuration.quality +
             c1[2] * hand_detection_configs[2].task_configuration.quality))

    optimize.add(UA[2] == (c2[0] * gesture_classification_configs[0].task_configuration.quality +
             c2[1] * gesture_classification_configs[1].task_configuration.quality +
             c2[2] * gesture_classification_configs[2].task_configuration.quality))

    optimize.add(UA[3] == (c3[0] * keyword_spotting_configs[0].task_configuration.quality +
             c3[1] * keyword_spotting_configs[1].task_configuration.quality +
             c3[2] * keyword_spotting_configs[2].task_configuration.quality))

    optimize.add(UA[4] == (c4[0] * keyword_spotting_configs[0].task_configuration.quality +
             c4[1] * keyword_spotting_configs[1].task_configuration.quality +
             c4[2] * keyword_spotting_configs[2].task_configuration.quality))

    optimize.add(UA[5] == c5[0] * image_classification_configs[0].task_configuration.quality)
    optimize.add(UA[6] == c6[0] * image_classification_configs[0].task_configuration.quality)

    UA_avg = Real("UA_avg")
    optimize.add(UA_avg == z3.Sum(UA) / 7.0)

    UA_max = Real("UA_max")
    optimize.add(UA_max == (face_detection_configs[0].task_configuration.quality +
              hand_detection_configs[0].task_configuration.quality +
              gesture_classification_configs[0].task_configuration.quality +
              keyword_spotting_configs[0].task_configuration.quality +
              keyword_spotting_configs[0].task_configuration.quality +
              image_classification_configs[0].task_configuration.quality +
              image_classification_configs[0].task_configuration.quality) / 7.0)

    UC = Real("UC")
    # alpha=beta=0.5, UL_max=1 as well => do not need to divide it
    optimize.add(UC == ( z3.RealVal(0.5) * (UA_avg / UA_max) + z3.RealVal(0.5) * UL_avg))
    optimize.add(UC <= 1)

    optimize.maximize(UC)

    print("starting optimize.check")
    if optimize.check() == sat:
        model = optimize.model()

        def z3_to_float(value):
            return float(value.numerator_as_long()) / float(value.denominator_as_long())

        print("found solution")
        print("UC =", z3_to_float(model[UC]))
        print("UA_avg =", z3_to_float(model[UA_avg]))
        print("UA_max =", z3_to_float(model[UA_max]))
        print("UL_avg =", z3_to_float(model[UL_avg]))
        print()
        print("UA0 =", z3_to_float(model[UA[0]]))
        print("UA1 =", z3_to_float(model[UA[1]]))
        print("UA2 =", z3_to_float(model[UA[2]]))
        print("UA3 =", z3_to_float(model[UA[3]]))
        print("UA4 =", z3_to_float(model[UA[4]]))
        print("UA5 =", z3_to_float(model[UA[5]]))
        print("UA6 =", z3_to_float(model[UA[6]]))
        print("\n\n\n\n")

        for var in sorted(model, key=lambda model_value: model_value.name()):
            val = model[var]
            if z3.is_rational_value(val):
                val = float(val.numerator_as_long()) / float(val.denominator_as_long())
            print(var, val)

def main():
    _solve_default_workload(1, True)

if __name__ == "__main__":
    main()
import json

import gurobipy as gp
from gurobipy import GRB

from scheduling.Scheduler import NullScheduler
from scheduling.strategy.Emerald.EmeraldHelper import get_emerald_inference_time
from tasks.FaceDetection import FaceDetection
from tasks.GestureClassification import GestureClassification
from tasks.HandDetection import HandDetection
from tasks.ImageClassification import ImageClassification
from tasks.KeywordSpotting import KeywordSpotting


def _constrain_one_binary_variable_selected(model, variables):
    model.addConstr(sum(variables) == 1)

# noinspection PyPep8Naming
def _solve_default_workload(num_streams : int, is_normal_period : bool, is_mean_inference_times : bool):
    model = gp.Model()
    BIG_M = 1

    period_length = 1.0 / 30 if is_normal_period else 1.0 / 35
    # m = 1.0 / (2 * period_length)
    # kws_1_deadline = period_length / 2
    # m_kws_1 = 1.0 / (2 * kws_1_deadline)

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
    c0 = [model.addVar(vtype=GRB.BINARY, name=f"c0{l}") for l in range(5)]
    _constrain_one_binary_variable_selected(model, c0)
    # Hand Detection
    c1 = [model.addVar(vtype=GRB.BINARY, name=f"c1{l}") for l in range(3)]
    _constrain_one_binary_variable_selected(model, c1)
    # Gesture Classification
    c2 = [model.addVar(vtype=GRB.BINARY, name=f"c2{l}") for l in range(3)]
    _constrain_one_binary_variable_selected(model, c2)

    # KWS 1
    c3 = [model.addVar(vtype=GRB.BINARY, name=f"c3{l}") for l in range(3)]
    _constrain_one_binary_variable_selected(model, c3)
    # KWS 2
    c4 = [model.addVar(vtype=GRB.BINARY, name=f"c4{l}") for l in range(3)]
    _constrain_one_binary_variable_selected(model, c4)

    # Image Classification 1
    c5 = [model.addVar(vtype=GRB.BINARY, name=f"c5{l}") for l in range(1)]
    _constrain_one_binary_variable_selected(model, c5)
    # Image Classification 2
    c6 = [model.addVar(vtype=GRB.BINARY, name=f"c6{l}") for l in range(1)]
    _constrain_one_binary_variable_selected(model, c6)

    # batch assignment variables (constant batch sized => equal to assignment variables for me)
    b = [[model.addVar(vtype=GRB.BINARY, name=f"b{i}{j}") for j in range(num_streams)] for i in range(7)]
    for task_processor_assignment_variables in b:
        _constrain_one_binary_variable_selected(model, task_processor_assignment_variables)

    # start times
    s = [[model.addVar(vtype=GRB.CONTINUOUS, name=f"s{i}{j}") for j in range(num_streams)] for i in range(7)]
    # finish times
    f = [[model.addVar(vtype=GRB.CONTINUOUS, name=f"f{i}{j}") for j in range(num_streams)] for i in range(7)]
    # start times constraints - immediately available
    for i in (0, 1, 3, 6):
        for j in range(num_streams):
            model.addConstr(s[i][j] >= 0)
    for j1 in range(num_streams):
        for j2 in range(num_streams):
            # start time - Gesture Classification
            model.addConstr(s[2][j1] >= f[1][j2])
    for j in range(num_streams):
        model.addConstr(s[4][j] >= (period_length / 2)) # second KWS becomes available after half the period
    # start time - Image Classification 1
    for j1 in range(num_streams):
        for j2 in range(num_streams):
            model.addConstr(s[5][j1] >= f[3][j2])

    # execution times - Face Detection
    E0 = [model.addVar(vtype=GRB.CONTINUOUS, name=f"E0{j}") for j in range(num_streams)]
    for j in range(num_streams):
        model.addConstr(E0[j] == (b[0][j] * (c0[0] * get_emerald_inference_time(face_detection_configs[0], is_mean_inference_times) +
                                             c0[1] * get_emerald_inference_time(face_detection_configs[1], is_mean_inference_times) +
                                             c0[2] * get_emerald_inference_time(face_detection_configs[2], is_mean_inference_times) +
                                             c0[3] * get_emerald_inference_time(face_detection_configs[3], is_mean_inference_times) +
                                             c0[4] * get_emerald_inference_time(face_detection_configs[4], is_mean_inference_times)
                                             )
                               )
                     )
    # execution times - Hand Detection
    E1 = [model.addVar(vtype=GRB.CONTINUOUS, name=f"E1{j}") for j in range(num_streams)]
    for j in range(num_streams):
        model.addConstr(E1[j] == (b[1][j] * (c1[0] * get_emerald_inference_time(hand_detection_configs[0], is_mean_inference_times) +
                                             c1[1] * get_emerald_inference_time(hand_detection_configs[1], is_mean_inference_times) +
                                             c1[2] * get_emerald_inference_time(hand_detection_configs[2], is_mean_inference_times)
                                             )
                               )
                     )
    # execution times - Gesture Classification
    E2 = [model.addVar(vtype=GRB.CONTINUOUS, name=f"E2{j}") for j in range(num_streams)]
    for j in range(num_streams):
        model.addConstr(E2[j] == (b[2][j] * (c2[0] * get_emerald_inference_time(gesture_classification_configs[0], is_mean_inference_times) +
                                             c2[1] * get_emerald_inference_time(gesture_classification_configs[1], is_mean_inference_times) +
                                             c2[2] * get_emerald_inference_time(gesture_classification_configs[2], is_mean_inference_times)
                                             )
                               )
                     )
    # execution times - KWS 1
    E3 = [model.addVar(vtype=GRB.CONTINUOUS, name=f"E3{j}") for j in range(num_streams)]
    for j in range(num_streams):
        model.addConstr(E3[j] == (b[3][j] * (c3[0] * get_emerald_inference_time(keyword_spotting_configs[0], is_mean_inference_times) +
                                             c3[1] * get_emerald_inference_time(keyword_spotting_configs[1], is_mean_inference_times) +
                                             c3[2] * get_emerald_inference_time(keyword_spotting_configs[2], is_mean_inference_times)
                                             )
                               )
                     )
    # execution times - KWS 2
    E4 = [model.addVar(vtype=GRB.CONTINUOUS, name=f"E4{j}") for j in range(num_streams)]
    for j in range(num_streams):
        model.addConstr(E4[j] == (b[4][j] * (c4[0] * get_emerald_inference_time(keyword_spotting_configs[0], is_mean_inference_times) +
                                             c4[1] * get_emerald_inference_time(keyword_spotting_configs[1], is_mean_inference_times) +
                                             c4[2] * get_emerald_inference_time(keyword_spotting_configs[2], is_mean_inference_times)
                                             )
                               )
                     )
    # execution times - Image Classification 1
    E5 = [model.addVar(vtype=GRB.CONTINUOUS, name=f"E5{j}") for j in range(num_streams)]
    for j in range(num_streams):
        model.addConstr(E5[j] == b[5][j] * c5[0] * get_emerald_inference_time(image_classification_configs[0], is_mean_inference_times))
    # execution times - Image Classification 2
    E6 = [model.addVar(vtype=GRB.CONTINUOUS, name=f"E6{j}") for j in range(num_streams)]
    for j in range(num_streams):
        model.addConstr(E6[j] == b[6][j] * c6[0] * get_emerald_inference_time(image_classification_configs[0], is_mean_inference_times))

    # constrain finish_time on start_time
    E = [E0, E1, E2, E3, E4, E5, E6]
    for i in range(7):
        for j in range(num_streams):
            model.addConstr(f[i][j] == s[i][j] + E[i][j])

    # only one task at once per stream
    for j in range(num_streams):
        for i1 in range(7):
            for i2 in range(7):
                if i2 == i1:
                    continue
                # linearizing boolean or
                b = model.addVar(vtype=GRB.BINARY, name=f"order_proc_{j}_tasks_{i1}{i2}")
                model.addConstr(s[i1][j] >= f[i2][j] - BIG_M * (1 - b))
                model.addConstr(s[i2][j] >= f[i1][j] - BIG_M * b)

    # absolute finish time of all tasks: MY NOTES EXPLAIN WHY I CANT HAVE INDIVIDUAL LATENCY UTILITIES
    tau = model.addVar(vtype=GRB.CONTINUOUS, name="tau")
    for j in range(num_streams):
        for i in range(7):
            model.addConstr(tau >= f[i][j])

    # Latency Utility based on final finish time of all tasks just like EMERALD; MY NOTES EXPLAIN WHY I CANT HAVE INDIVIDUAL LATENCY UTILITIES
    UL = model.addVar(vtype=GRB.CONTINUOUS, name="UL")
    # xpts: the points it should linearly interpolate between (last value is an upper bound for tau)
    # ypts: the values it should linearly interpolate in between consecutive xpts
    model.addGenConstrPWL(tau, UL, [0, period_length, 3 * period_length, 1], [1, 1, 0, 0]) # gurobi can interpolate correctly by itself

    # Accuracy Utilities
    UA = [model.addVar(vtype=GRB.CONTINUOUS, name=f"UA{i}") for i in range(7)]
    model.addConstr(UA[0] == (c0[0] * face_detection_configs[0].task_configuration.quality +
             c0[1] * face_detection_configs[1].task_configuration.quality +
             c0[2] * face_detection_configs[2].task_configuration.quality +
             c0[3] * face_detection_configs[3].task_configuration.quality +
             c0[4] * face_detection_configs[4].task_configuration.quality))

    model.addConstr(UA[1] == (c1[0] * hand_detection_configs[0].task_configuration.quality +
             c1[1] * hand_detection_configs[1].task_configuration.quality +
             c1[2] * hand_detection_configs[2].task_configuration.quality))

    model.addConstr(UA[2] == (c2[0] * gesture_classification_configs[0].task_configuration.quality +
             c2[1] * gesture_classification_configs[1].task_configuration.quality +
             c2[2] * gesture_classification_configs[2].task_configuration.quality))

    model.addConstr(UA[3] == (c3[0] * keyword_spotting_configs[0].task_configuration.quality +
             c3[1] * keyword_spotting_configs[1].task_configuration.quality +
             c3[2] * keyword_spotting_configs[2].task_configuration.quality))

    model.addConstr(UA[4] == (c4[0] * keyword_spotting_configs[0].task_configuration.quality +
             c4[1] * keyword_spotting_configs[1].task_configuration.quality +
             c4[2] * keyword_spotting_configs[2].task_configuration.quality))

    model.addConstr(UA[5] == c5[0] * image_classification_configs[0].task_configuration.quality)
    model.addConstr(UA[6] == c6[0] * image_classification_configs[0].task_configuration.quality)

    UA_avg = model.addVar(vtype=GRB.CONTINUOUS, name="UA_avg")
    model.addConstr(UA_avg == sum(UA) / 7.0)

    UA_max = (face_detection_configs[0].task_configuration.quality +
              hand_detection_configs[0].task_configuration.quality +
              gesture_classification_configs[0].task_configuration.quality +
              keyword_spotting_configs[0].task_configuration.quality +
              keyword_spotting_configs[0].task_configuration.quality +
              image_classification_configs[0].task_configuration.quality +
              image_classification_configs[0].task_configuration.quality) / 7.0

    UC = model.addVar(vtype=GRB.CONTINUOUS, name="UC")
    model.addConstr(UC == ( 0.5 * (UA_avg / UA_max) + 0.5 * UL))
    model.addConstr(UC <= 1)

    model.setObjectiveN(UC, index=0, priority=num_streams + 1, weight=GRB.MAXIMIZE)
    index = 1
    for j in range(num_streams - 1, -1, -1):
        model.setObjectiveN(f[3][j], index=index, priority=j + 1, weight=GRB.MINIMIZE)
        index += 1
    # minimize tau so that tau is equal to latest finish time in overview
    model.setObjectiveN(tau, index=index, priority=0, weight=GRB.MINIMIZE)

    model.optimize()

    if model.Status == GRB.OPTIMAL:
        print("found solution")

        for var in sorted((v for v in model.getVars() if "order" not in v.VarName), key=lambda v: v.VarName):
            print(var.VarName, var.X)
    else:
        print("failed :(")

def main():
    _solve_default_workload(2, False, False)

if __name__ == "__main__":
    main()
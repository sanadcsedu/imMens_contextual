# New iteration of our MDP which has the [temporal, scatterplot and carrier] x [sensemaking, question] as states
# [observation, generalization, explanation and steer] as actions
import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd
import random
import sys
from read_data import read_data
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import adaptive_epsilon
# from mortal_bandit import mortal_bandit
from greedy_policy import Greedy

class Win_Stay_Lose_Shift():
    def __init__(self, vizs):
        self.vizs = vizs

    def run(self, data, threshold):
        epoch = 10
        accu_list = []
        for thres in threshold:
            sz = len(data)
            s_idx = int(thres * sz)
            accu = 0
            # pdb.set_trace()
            for e in range(epoch):
                prev_arm = random.choice(self.vizs)
                prev_reward = 0
                for idx in range(0, s_idx):
                    # 0: index, 1: action, 2: visualization, 3: high_level_state, 4: reward 
                    if data[idx][4] >= prev_reward:
                        pred_arm = prev_arm
                    else:
                        pred_arm = random.choice(self.vizs)
                    # pdb.set_trace()
                    if pred_arm == data[idx][2]:
                        prev_arm = pred_arm

                cnt = 0
                denom = 0
                prev_reward = 0
                for idx in range(s_idx + 1, sz):
                    denom += 1
                    if data[idx][4] >= prev_reward:
                        pred_arm = prev_arm
                    else:
                        pred_arm = random.choice(self.vizs)
                    # pdb.set_trace()
                    if pred_arm == data[idx][2]:
                        cnt += 1  # win
                    prev_arm = pred_arm

                accu += (cnt / denom)
            accu_list.append(round(accu / epoch, 2))
        return accu_list

#Going to use the function below, to AGGREGATE the [accuracy vs %training data] for all users
if __name__ == "__main__":
    # vizs = ['bar-5', 'hist-2', 'hist-3', 'hist-4', 'geo-0-1'] #Brightkite
    vizs = ['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1'] #FAA 
    obj = read_data(vizs)
    data, uname = obj.get_files()
    # pdb.set_trace()
    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    wsls = Win_Stay_Lose_Shift(vizs)
    greedy = Greedy(vizs)
    e_greedy_variations = adaptive_epsilon.adaptive_epsilon(vizs)
    # mortal = mortal_bandit(vizs)

    algos = []
    wsls_result = 9*[0]
    greedy_result = 9*[0]
    decay_result = 9*[0]
    egreedy_result = 9*[0]
    adaptive_result = 9*[0]
    # mortal_result = 9*[0]
    entries = 0
    for idx2, d in enumerate(data):

        # if uname[idx2] == 'p14' or uname[idx2] == 'p15':
        #     continue
        entries += 1
        temp = wsls.run(d, threshold)
        idx = 0
        # pdb.set_trace()
        # print(temp)
        for idx in range(len(temp)):
            wsls_result[idx] += temp[idx]
        # print(temp)
        # pdb.set_trace()
        temp = greedy.run(d, threshold)
        idx = 0
        # print(temp)
        for idx in range(len(temp)):
            greedy_result[idx] += temp[idx]
        # print(greedy_result)
        temp = e_greedy_variations.run_MAB(d, threshold, True)  # E-Greedy with decay
        idx = 0
        # print(temp)
        for idx in range(len(temp)):
            decay_result[idx] += temp[idx]
        temp = e_greedy_variations.run_MAB(d, threshold, False)  # E-Greedy
        idx = 0
        # print(temp)
        for idx in range(len(temp)):
            egreedy_result[idx] += temp[idx]
        # temp = e_greedy_variations.run_MAB_adaptive(d, threshold) # adaptive E-Greedy
        idx = 0
        # print(temp)
        for idx in range(len(temp)):
            adaptive_result[idx] += temp[idx]
        # ----
        print(uname[idx2])
        # temp = mortal.run_stochastic_early_stop(uname[idx2], threshold, d)
        # idx = 0
        # for idx in range(len(temp)):
        #     mortal_result[idx] += temp[idx]
        # idx2 += 1

    accu_list = []
    algos = []
    accu_list.append(wsls_result)
    algos.append('Win-Stay-Lose-Shift')
    accu_list.append(greedy_result)
    algos.append('Greedy')
    accu_list.append(decay_result)
    algos.append('E-Greedy with E-Decay')
    accu_list.append(egreedy_result)
    algos.append('E-Greedy')
    accu_list.append(adaptive_result)
    algos.append('Adaptive E-Greedy')
    # accu_list.append(mortal_result)
    # algos.append('Mortal Bandit')

    #Plotting
    for idx in range(len(accu_list)):
        # pdb.set_trace()
        for idx2 in range(9):
            accu_list[idx][idx2] = round(accu_list[idx][idx2] / entries, 2)
        print(algos[idx], accu_list[idx])

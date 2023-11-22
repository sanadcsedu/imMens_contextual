import os
import fnmatch
import pdb
from collections import defaultdict
import glob
import pandas as pd
import random
import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import adaptive_epsilon

class Greedy:
    def __init__(self, vizs):
        self.vizs = vizs
        self.arms_idx = defaultdict()
        idx = 0
        for v in self.vizs:
            self.arms_idx[v] = idx
            idx += 1
        self.arms = np.full(len(self.vizs), 1, dtype='float')
        # print(self.arms)

    def run(self, data, threshold):
        epoch = 10
        accu_list = []
        for thres in threshold: #Running for all thresholds

            #Training Module
            sz = len(data)
            s_idx = int(thres * sz) # Used for partitioning the dataset

            for idx in range(0, s_idx):
                # 0: index, 1: action, 2: visualization, 3: high_level_state, 4: reward 
                self.arms[self.arms_idx[data[idx][2]]] += data[idx][4]
            sum = self.arms.sum()
            self.arms /= sum
            # pdb.set_trace()
            arg_max = np.argmax(self.arms) #Finds the arm that yielded the maximum reward
            arg_value = self.arms[arg_max] # Probability
            accu = 0

            #Testing module [testing for 10 times and then averaging the test accuracy]
            for e in range(epoch):
                cnt = 0
                ###### Change #######
                sum = self.arms.sum()
                self.arms /= sum
                ####################
                cur_arm = self.vizs[arg_max]  # Always picking the best action based on past experience.
                denom = 0
                for idx in range(s_idx, sz):
                    denom += 1
                    if data[idx][2] == cur_arm:
                        cnt += 1
                    self.arms[self.arms_idx[data[idx][2]]] += data[idx][4]
                accu += (cnt / denom)
            accu_list.append(round(accu / epoch, 2))

        return accu_list

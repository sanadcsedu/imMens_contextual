import pdb
import random
import sys
sys.path.insert(0, "D:\\imMens Learning\\stationarity_test")
from mann_whitney_v3 import integrate
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import math
import numpy as np

class adaptive_epsilon:
    def __init__(self, vizs, epsilon_min=0.01, epsilon_decay=0.995):
        self.vizs = vizs
        idx = 0
        self.arms = defaultdict()
        for v in vizs:
            self.arms[v] = idx
            idx += 1
        # print(self.arms)
        self.num_arms = len(vizs)
        self.counts = [0 for _ in range(len(vizs))]
        self.q_values = [0.0 for _ in range(len(vizs))]
        # self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = 1.0

        self.k = 0  # Keeps track of how many times the algorithm performs exploration
        self.l = 0  # maximum number of time exploration can continue
        self.f = 0  # Regularization Parameter
        self.max_prev = 0
        self.max_cur = 0
        self.cnt = 0

############ E-Greedy Update and Select, Contains Adaptive E-Greedy in the form of E-Decay ###############
    def reset(self, vizs, eps, epsilon_decay):
        self.counts = [0 for _ in range(len(vizs))]
        self.q_values = [0.0 for _ in range(len(vizs))]
        # self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon = eps

    def update(self, chosen_arm, reward, adaptive=True):
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.q_values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.q_values[chosen_arm] = new_value

        # Update epsilon
        if adaptive:
            self.epsilon = self.epsilon * self.epsilon_decay
            # self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def argmax(self):
        top_value = float("-inf")
        ties = []
        for i in range(len(self.q_values)):
            if self.q_values[i] > top_value:
                top_value = self.q_values[i]
                ties = []
            if self.q_values[i] == top_value:
                ties.append(i)
        return np.random.choice(ties)

    def select_arm(self, epsilon):
        if random.random() > epsilon:
            # return "argmax", self.argmax()
            # print("Argmax: {} {}".format(self.q_values.index(max(self.q_values)), self.q_values))
            return "argmax", self.q_values.index(max(self.q_values))
        else:
            return "random", random.randrange(self.num_arms)

    def run_MAB(self, data, threshold, version):
        #Training the E-Greedy model from the data
        accu_list = []
        epsilons = [0.001, 0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
        runs = 10
        #Finding the best Epsilon value using the training data and then testing the performance of E-Greedy on the testing data.
        for thres in threshold: #Running for all thresholds
            #Training Module
            sz = len(data)
            s_idx = int(thres * sz)  # Used for partitioning the dataset
            # pdb.set_trace()
            max_accu = -1
            best_eps = None
            for eps in epsilons: #Finding the best eps using the training data
                avg_accu = 0
                # print("############# {} ############".format(eps))
                self.reset(self.vizs, eps, 0.9)
                for r in range(runs):
                    num = denom = 0
                    for idx in range(0, s_idx):
                        denom += 1
                        pred, arm = self.select_arm(eps)
                        # pdb.set_trace()
                        if self.arms[data[idx][2]] == arm:
                            num += 1
                        self.update(self.arms[data[idx][2]], data[idx][4], version)

                    cur_accu = num / denom
                    avg_accu += cur_accu
                avg_accu /= runs
                if avg_accu > max_accu:
                    max_accu = avg_accu
                    best_eps = eps
            #Now use the best epsilion using the testing data
            avg_accu = 0
            
            self.reset(self.vizs, best_eps, 0.9)
            for r in range(runs):
                # Now that we've the best eps, train the q_values again upto the testing point.
                for _ in range(50):
                    for idx in range(0, s_idx):
                        self.update(self.arms[data[idx][2]], data[idx][4], version)
                num = denom = 0
                # Here we do the testing: we calculate accuracy by measuring precision
                for idx in range(s_idx, sz):
                    denom += 1
                    pred, arm = self.select_arm(best_eps)
                    if self.arms[data[idx][2]] == arm:
                        num += 1
                    self.update(self.arms[data[idx][2]], data[idx][4], version)
                avg_accu += num / denom
                # pdb.set_trace()
            accu_list.append(round(avg_accu / runs, 2))
        return accu_list


############ Adaptive E-Greedy implementation from paper with two hyper-parameters l and f ###############
    def reset_adaptive(self, vizs, l, f):
        self.counts = [0 for _ in range(len(vizs))]
        self.q_values = [0.0 for _ in range(len(vizs))]
        self.epsilon = 0.5
        self.k = 0  # Keeps track of how many times the algorithm performs exploration
        self.l = l  # maximum number of time exploration can continue
        self.f = f  # Regularization Parameter
        self.max_prev = 0
        self.max_cur = 0
        self.cnt = 0

    def change_epsilon(self, delta):
        if delta > 0:
            self.epsilon = 1.0 / (1.0 + math.exp(-2 * delta))
            self.epsilon -= 0.5
            # self.epsilon /= 2
        else:
            self.epsilon = 0.5

    def select_arm_adaptive(self):
        if random.random() > self.epsilon:
            return self.q_values.index(max(self.q_values))
        else:
            self.k += 1
            if self.k == self.l and self.cnt > 0:
                self.max_cur /= self.cnt
                delta = (self.max_cur - self.max_prev) * self.f
                self.change_epsilon(delta)
                self.max_prev = self.max_cur
                self.max_cur = 0
                self.cnt = 0
                self.k = 0

            return random.randrange(self.num_arms)

    def update_adaptive(self, chosen_arm, reward):
        self.max_cur += reward
        self.cnt += 1

        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        value = self.q_values[chosen_arm]
        new_value = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.q_values[chosen_arm] = new_value

    def run_MAB_adaptive(self, data, threshold):
        #Training the E-Greedy model from the data
        accu_list = []
        #Finding the best Epsilon value using the training data and then testing the performance of E-Greedy on the testing data.
        for thres in threshold: #Running for all thresholds
            #Training Module
            sz = len(data)
            s_idx = int(thres * sz)  # Used for partitioning the dataset
            # pdb.set_trace()
            max_accu = -1
            # best_eps = None
            best_l = None
            best_f = None
            # for eps in epsilons: #Finding the best eps using the training data
            runs = 25
            for l in range(1, 50):
                for f in range(1, 25):
                    avg_accu = 0
                    self.reset_adaptive(self.vizs, l, f)
                    for r in range(runs):
                        num = denom = 0
                        for idx in range(0, s_idx):
                            denom += 1
                            arm = self.select_arm_adaptive()
                            self.update_adaptive(self.arms[data[idx][2]], data[idx][4])

                        cur_accu = num / denom
                        avg_accu += cur_accu
                    avg_accu /= runs
                    if avg_accu > max_accu:
                        max_accu = avg_accu
                        best_f = f
                        best_l = l
                #Now use the best epsilion using the testing data
            avg_accu = 0
            self.reset_adaptive(self.vizs, best_l, best_f)
            for r in range(runs):
                # Now that we've the best eps, train the q_values again upto the testing point.
                for _ in range(50):
                    for idx in range(0, s_idx):
                        arm = self.select_arm_adaptive()
                        self.update_adaptive(self.arms[data[idx][2]], data[idx][4])
                # Here we do the testing
                num = denom = 0
                for idx in range(s_idx, sz):
                    denom += 1
                    arm = self.select_arm_adaptive()
                    if self.arms[data[idx][2]] == arm:
                        num += 1
                    self.update_adaptive(self.arms[data[idx][2]], data[idx][4])

                avg_accu += num / denom
            accu_list.append(round(avg_accu / runs, 2))
        return accu_list


if __name__ == "__main__":
    obj = integrate()
    data, uname = obj.get_files()
    mab = adaptive_epsilon(['bar-4', 'bar-2', 'hist-3', 'scatterplot-0-1'])
    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    idx = 0
    for users_data in data:
        decay = mab.run_MAB(users_data, threshold, True) #Adaptive E-Greedy with decay
        egreedy = mab.run_MAB(users_data, threshold, False) #E-Greedy
        adaptive = mab.run_MAB_adaptive(users_data, threshold)
        print(uname[idx])
        idx += 1
        print(decay)
        print(egreedy)
        print(adaptive)

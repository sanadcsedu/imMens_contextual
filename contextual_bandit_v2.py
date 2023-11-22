import vowpalwabbit
import pandas as pd
import environment5
# import plotting
from collections import Counter,defaultdict
import json
from read_data import read_data
from pathlib import Path
import glob
from tqdm import tqdm 
import os 
import pdb 
import random
# import matplotlib.pyplot as plt
import itertools
import numpy as np

class contextual_bandit:
    def __init__(self):
        # self.train_data = []
        pass 
    
    # This function modifies (context, action, cost, probability) to VW friendly format
    def to_vw_example_format(self, context, actions, cb_label=None):
        if cb_label is not None:
            chosen_action, cost, prob = cb_label
        example_string = ""
        example_string += "shared |Raw raw_action={} State state={}\n".format(
            context["raw_action"], context["state"]
        )
        for action in actions:
            if cb_label is not None and action == chosen_action:
                example_string += "0:{}:{} ".format(cost, prob)
            example_string += "|Action vis={} \n".format(action)
        # Strip the last newline
        return example_string[:-1]

    def sample_custom_pmf(self, pmf):
        total = sum(pmf)
        scale = 1 / total
        pmf = [x * scale for x in pmf]
        draw = random.random()
        sum_prob = 0.0
        for index, prob in enumerate(pmf):
            sum_prob += prob
            if sum_prob > draw:
                return index, prob
    
    def get_action(self, vw, context, actions):
        vw_text_example = self.to_vw_example_format(context, actions)
        pmf = vw.predict(vw_text_example)
        chosen_action_index, prob = self.sample_custom_pmf(pmf)
        return actions[chosen_action_index], prob

    def run_simulation_multiple_cost_functions(self, vw, data, threshold, actions, do_learn=True):
        epoch = 10
        #training phase
        for _ in range(epoch):
            num_iterations = int(len(data) * threshold) 
            start_counter = 0
            end_counter = start_counter + num_iterations
            for i in range(start_counter, end_counter):
                # 0: index, 1: action, 2: visualization, 3: high_level_state, 4: reward 
                # 1. in each simulation choose a user
                raw_action = data[i][1]
                # 2. choose time of day for a given user
                high_level_state = data[i][3]
                # Construct context based on chosen user and time of day
                context = {"raw_action": raw_action, "state": high_level_state}

                # 3. Use the get_action function we defined earlier
                action, prob = self.get_action(vw, context, actions)
                # pdb.set_trace()
                if(action == data[i][2]):
                    # ctr.append(1)
                    cost = -1
                else:
                    # ctr.append(0)
                    cost = 0

                # 4. Get cost of the action we chose
                # cost = cost_function(context, action)
                
                if do_learn:
                    # 5. Inform VW of what happened so we can learn from it
                    # vw_format = vw.parse(
                    #     self.to_vw_example_format(context, actions, (action, cost, prob)),
                    #     vowpalwabbit.LabelType.CONTEXTUAL_BANDIT,)
                    # vw.learn(vw_format)

                    vw_format = vw.parse(
                        self.to_vw_example_format(context, actions, (data[i][2], -10*data[i][4], prob)),
                        vowpalwabbit.LabelType.CONTEXTUAL_BANDIT,)
                    vw.learn(vw_format)

        #testing phase
        ctr = []
        start_counter = int(len(data) * threshold) 
        end_counter = len(data)
        for i in range(start_counter, end_counter):
            
            raw_action = data[i][1]
            
            high_level_state = data[i][3]
            
            context = {"raw_action": raw_action, "state": high_level_state}

            action, prob = self.get_action(vw, context, actions)
            # print("Picked: {}   Ground: {} state: {} Reward: {}".format(action, data[i][2], data[i][3], data[i][4]))
            if(action == data[i][2]):
                ctr.append(1)
                cost = -1
            else:
                ctr.append(0)
                cost = 0
            
            if do_learn:
                # 5. Inform VW of what happened so we can learn from it
                # vw_format = vw.parse(
                #     self.to_vw_example_format(context, actions, (action, cost, prob)),
                #     vowpalwabbit.LabelType.CONTEXTUAL_BANDIT,)
                # vw.learn(vw_format)

                vw_format = vw.parse(
                    self.to_vw_example_format(context, actions, (data[i][2], -10*data[i][4], prob)),
                    vowpalwabbit.LabelType.CONTEXTUAL_BANDIT,)
                vw.learn(vw_format)
        return np.mean(ctr)

    def get_data_into_pandas(self, data, thres):
        train_data = pd.DataFrame(columns=['action', 'cost', 'probability', 'feature1', 'feature2'])
        action_space = {'pan': 1, 'zoom': 2, 'brush': 3, 'range select': 4}
        vis_space = {'bar-4': 1, 'bar-2': 2, 'scatterplot-0-1': 3, 'hist-3': 4}
        
        l = int(len(data) * thres)
        for idx in range(0, l):
            # 0: index, 1: action, 2: visualization, 3: high_level_state, 4: reward 
            # pdb.set_trace()
            # print(data[idx])
            train_data = pd.concat([train_data, pd.DataFrame({
                # 'action': [vis_space[data[idx][2]]],
                'label' : [data[idx][2]],
                'cost': [data[idx][4]],
                'probability': [0.25],
                'feature1': [data[idx][3]],
                # 'feature2': [action_space[data[idx][1]]],
                'feature2': [data[idx][1]],
            })], ignore_index=False)

        train_df = pd.DataFrame(train_data)

        # # Add index to data frame
        train_df["index"] = range(1, len(train_df) + 1)
        train_df = train_df.set_index("index")
        return train_df
    
    def get_data_into_pandas_test(self, data, thres):
    
        test_data = pd.DataFrame(columns=['action', 'cost', 'probability','feature1', 'feature2'])
        action_space = {'pan': 1, 'zoom': 2, 'brush': 3, 'range select': 4}
        vis_space = {'bar-4': 1, 'bar-2': 2, 'scatterplot-0-1': 3, 'hist-3': 4}

        l = int(len(data) * thres)
        for idx in range(l, len(data)):
            # 0: index, 1: action, 2: visualization, 3: high_level_state, 4: reward 
            test_data = pd.concat([test_data, pd.DataFrame({
                # 'action': [vis_space[data[idx][2]]],
                'action' : [data[idx][2]],
                'cost': [data[idx][4]],
                'probability': [0.25],
                'feature1': [data[idx][3]],
                # 'feature2': [action_space[data[idx][1]]],
                'feature2': [data[idx][1]],
            })], ignore_index=False)

        test_df = pd.DataFrame(test_data)
        # Add index to data frame
        test_df["index"] = range(1, len(test_df) + 1)
        test_df = test_df.set_index("index")
        return test_df

    def run(self, train_df, test_df):
        vw = vowpalwabbit.Workspace("--cb_explore 4", quiet=False)
        for i in train_df.index:
            # pdb.set_trace()
            action = train_df.loc[i, "action"]
            cost = train_df.loc[i, "cost"]
            probability = train_df.loc[i, "probability"]
            feature1 = train_df.loc[i, "feature1"]
            feature2 = train_df.loc[i, "feature2"]
            # feature3 = train_df.loc[i, "feature3"]

            # Construct the example in the required vw format.
            learn_example = (str(action) + ":" + str(cost)  + " | " + str(feature1) + " " + str(feature2))
            # print(learn_example)
            # Here we do the actual learning.
            vw.learn(learn_example)

        num = 0
        denom = 0
        for j in test_df.index:
            feature1 = test_df.loc[j, "feature1"]
            feature2 = test_df.loc[j, "feature2"]

            test_example = "| " + str(feature1) + " " + str(feature2)
            choice = vw.predict(test_example)
            print(j, choice, test_df.loc[j, 'action'])
            if(choice == test_df.loc[j, 'action']):
                num += 1
            denom += 1
        return round(num / denom, 2)

def run_cb(data):
    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # context = {"user": "Tom", "time_of_day": "morning"}
    actions = ['bar-4', 'bar-2', 'scatterplot-0-1', 'hist-3']
    
    accuracy = []
    for thres in threshold:
        cb = contextual_bandit()
        epsilon = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        _max = -1
        for eps in epsilon:
            command = "--cb_explore_adf -q RA SA RS --quiet --epsilon " + str(eps)
            vw = vowpalwabbit.Workspace(command)
            ctr = cb.run_simulation_multiple_cost_functions(vw, data, thres, actions)
            _max = max(_max, ctr)
        accuracy.append(_max)    
        # pdb.set_trace()
    return accuracy

if __name__ == "__main__":     
    env = read_data(['bar-4', 'bar-2', 'scatterplot-0-1', 'hist-3'])
    accu = np.zeros(9, dtype=float)
    for raw_fname in env.raw_files:
        user = Path(raw_fname).stem.split('-')[0]
        excel_fname = [string for string in env.excel_files if user in string][0]
        data = env.merge(raw_fname, excel_fname)
        accu = np.add(accu, run_cb(data))
        print(user, accu)
        # print(accu)
    accu /= 8
    print(np.round(accu, decimals=2))

    # eps 0.1 [0.68 0.66 0.64 0.62 0.6  0.62 0.62 0.59 0.57]
    # eps 0.2 [0.68 0.66 0.66 0.62 0.63 0.58 0.62 0.62 0.61]
#This version of the Mann Whitney test checks stationarity of the actions based on recieved rewards
import csv
import pdb
import glob
import random

import pandas as pd
import os
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
from scipy.special import rel_entr
from scipy.stats import mannwhitneyu


class integrate:
    def __init__(self, vizs):
        if len(vizs) == 5:
            self.raw_files = glob.glob("/Users/sanadsaha92/Desktop/Research_Experiments/imMens-Interactions/stationarity_test/KLDivergenceTest/RawInteractions/*-reformed.csv")
            self.excel_files = glob.glob("/Users/sanadsaha92/Desktop/Research_Experiments/imMens-Interactions/stationarity_test/KLDivergenceTest/FeedbackLog/*annot.xlsx")
        else:
            self.raw_files = glob.glob("/Users/sanadsaha92/Desktop/Research_Experiments/imMens-Interactions/stationarity_test/KLDivergenceTest/RawInteractions/faa/*.csv")
            self.excel_files = glob.glob("/Users/sanadsaha92/Desktop/Research_Experiments/imMens-Interactions/stationarity_test/KLDivergenceTest/FeedbackLog/faa/*.xlsx")

        self.path = '/Users/sanadsaha92/Desktop/Research_Experiments/imMens-Interactions/stationarity_test/KLDivergenceTest/Merged'
        self.vizs = vizs
        self.cum_rewards = defaultdict(list)

    def debug(self, user, raw_data, feedback_data):
        print(user)
        print("#######RAW#########")
        for idx in range(len(raw_data)):
            print(idx, raw_data[idx][0], raw_data[idx][1], raw_data[idx][2])
        print("#######EXCEL#########")
        for idx in range(len(feedback_data)):
            print(idx, feedback_data[idx][0], feedback_data[idx][2], feedback_data[idx][3])

        print(user)
        for v in self.cum_rewards:
            print(v)
            for keys in self.cum_rewards[v]:
                print(keys)

    def get_files(self):
        ret = []
        uname = []
        for raw_fname in self.raw_files:
            merged = []
            user = Path(raw_fname).stem.split('-')[0]
            # if user in for_now:
            #     continue
            excel_fname = [string for string in self.excel_files if user in string][0]
            self.cum_rewards.clear()
            merged = self.merge(user, raw_fname, excel_fname)
            ret.append(merged)
            uname.append(user)
        return ret, uname

    def excel_to_memory(self, df):
        data = []
        for index, row in df.iterrows():
            mm = row['time'].minute
            ss = row['time'].second
            seconds = mm * 60 + ss
            if row['State'] == "None": #When reading from excel do not consider states = None
                continue
            data.append([seconds, row['proposition'], row['Reward'], row['State'], row['Subtask']])
        return data

    def raw_to_memory(self, csv_reader):
        next(csv_reader)
        data = []
        for lines in csv_reader:
            time = lines[0].split(":")
            mm = int(time[1])
            ss = int(time[2])
            seconds = mm * 60 + ss
            data.append([seconds, lines[1], lines[2]])
        return data

    def get_cur_viz(self, cur_time, raw_data):
        for idx in range(len(raw_data) - 1):
            if raw_data[idx][0] <= cur_time <= raw_data[idx + 1][0]:
                return raw_data[idx][2]
        return raw_data[idx - 1][2]

    def find_runtime(self, raw_data, feedback_data):
        l1 = len(raw_data)
        l2 = len(feedback_data)
        _min = min(int(raw_data[l1 - 1][0]), int(feedback_data[l2 - 1][0]))
        return _min

    #Used for merging the raw interaction (reformed) files with the Excel feedback files
    #we are also going to use this function to calculate the cumulative rewards (probability distribution)
    def merge(self, user, raw_fname, excel_fname):
        raw_interaction = open(raw_fname, 'r')
        csv_reader = csv.reader(raw_interaction)
        raw_data = self.raw_to_memory(csv_reader)
        raw_interaction.close()

        df_excel = pd.read_excel(excel_fname, sheet_name="Sheet3 (2)", usecols="A:G")
        feedback_data = self.excel_to_memory(df_excel)

        holder = []
        for idx in range(len(feedback_data)):
            holder.append(
                (idx, self.get_cur_viz(feedback_data[idx][0], raw_data), feedback_data[idx][2], feedback_data[idx][4]))
            # print(holder[idx])
        # pdb.set_trace()
        for idx in range(len(feedback_data)):
            for v in self.vizs:
                if v == holder[idx][1]:
                    self.cum_rewards[v].append(holder[idx][2])
                else:
                    self.cum_rewards[v].append(0)
                # pdb.set_trace()
        # self.stationarity_test1(user)
        return holder

        # cur_subtask = 1
        # for idx in range(len(feedback_data)):
        #     if holder[idx][3] == cur_subtask:
        #         for v in self.vizs:
        #             if v == holder[idx][1]:
        #                 self.cum_rewards[v].append(holder[idx][2])
        #             else:
        #                 self.cum_rewards[v].append(0)
        #     else:
        #         # pdb.set_trace()
        #         for v in self.vizs:
        #             for idx2 in range(1, len(self.cum_rewards[v])):
        #                 self.cum_rewards[v][idx2] += self.cum_rewards[v][idx2 - 1]
        #         self.stationarity_test1(user, cur_subtask)
        #         self.cum_rewards.clear()
        #         cur_subtask = holder[idx][3]
        #         idx -= 1
        # self.stationarity_test1(user)

    # This one checks the 50-50 split
    def stationarity_test1(self, user):
        print("##### USER {} #######".format(user))
        for v in self.vizs:
            w1 = []
            w2 = []
            denom = 0
            numer = 0
            for idx in range(len(self.cum_rewards[v])):
                numer += self.cum_rewards[v][idx]
                for v2 in self.vizs:
                    denom += self.cum_rewards[v2][idx]
                if denom == 0:
                    continue
                if idx < (len(self.cum_rewards[v]) / 2):
                    w1.append(round(numer / denom, 2))
                else:
                    w2.append(round(numer / denom, 2))
            print(v, end=" ")
            try:
                results = mannwhitneyu(w1, w2)
                # print(results)
                if results.pvalue < 0.05:
                    print("True", end=" ")
                else:
                    print("False", end=" ")
            except ValueError as ve:
                print("NED", end=" ")
            print()

    #Based on frequency of picking up a visualization (In this function we're not considering the reward)
    def merge2(self, user, raw_fname, excel_fname):
        raw_interaction = open(raw_fname, 'r')
        csv_reader = csv.reader(raw_interaction)
        raw_data = self.raw_to_memory(csv_reader)
        raw_interaction.close()

        df_excel = pd.read_excel(excel_fname, sheet_name= "Sheet3 (2)", usecols="A:G")
        feedback_data = self.excel_to_memory(df_excel)

        holder = []
        for idx in range(len(feedback_data)):
            holder.append((idx, self.get_cur_viz(feedback_data[idx][0], raw_data), feedback_data[idx][2], feedback_data[idx][4]))
            # print(holder[idx])

        cur_subtask = 1
        #This section spilts the rewards into windows (each subtask is a single window)
        # For each window we calculate accumulated reward for each visualization
        windows = []
        for idx in range(len(feedback_data)):
            if holder[idx][3] == cur_subtask:
                for v in self.vizs:
                    if v == holder[idx][1]:
                        self.cum_rewards[v].append(holder[idx][2]) #Adding 1 if we see the user picking this viz
                    else:
                        self.cum_rewards[v].append(0) #Adding 0 as the user is not picking this viz
            else:
                # pdb.set_trace()
                for v in self.vizs:
                    for idx2 in range(1, len(self.cum_rewards[v])):
                        self.cum_rewards[v][idx2] += self.cum_rewards[v][idx2 - 1]

                windows.append(self.cum_rewards.copy())
                # pdb.set_trace()

                # self.stationarity_test1(user, cur_subtask)
                self.cum_rewards.clear()
                # pdb.set_trace()

                cur_subtask = holder[idx][3]
                idx -= 1
        windows.append(self.cum_rewards)
        self.stationarity_test2(user, windows)

    #This one does stationarity checks for all windows (subtasks)
    def stationarity_test2(self, user, windows):
        # Generating the probability distribution from the windows
        for v in self.vizs:
            windows_prob = []
            for w in windows:
                w1 = []
                for idx in range(len(w[v])):
                    denom = 0
                    for v2 in self.vizs:
                        # pdb.set_trace()
                        denom += w[v2][idx]
                    if denom == 0:
                        continue
                    w1.append(round(w[v][idx] / denom, 2))
                windows_prob.append(w1)
            # pdb.set_trace()
            for idx in range(len(windows_prob)):
                print(idx+1, end= "------\n")
                for idx2 in range(idx+1, len(windows_prob)):
                    try:
                        results = mannwhitneyu(windows_prob[idx], windows_prob[idx2])
                        print(idx2 + 1, end=" ")
                        if results.pvalue < 0.05:
                            print("True", end=" ")
                        else:
                            print("False", end=" ")
                    except ValueError as ve:
                        print("NED", end=" ")
                    print()

    def plot_graph(self, user):
        for v in self.cum_rewards:
            x_axis = []
            y_axis = []
            for keys in self.cum_rewards[v]:
                x_axis.append(keys[0])
                y_axis.append(keys[1])

            plt.plot(x_axis, y_axis, label=v)
            plt.ylabel('Rewards')
            # plt.xticks([])
            plt.xlabel('time')
        plt.legend(loc='best')
        title = 'Cumulative Rewards for user: ' + user
        plt.title(title)
        plt.show()

    #I'm going to use the following function to merge
    def to_generate_tableau_plot(self, user, raw_fname, excel_fname):
        raw_interaction = open(raw_fname, 'r')
        csv_reader = csv.reader(raw_interaction)
        raw_data = self.raw_to_memory(csv_reader)
        raw_interaction.close()

        df_excel = pd.read_excel(excel_fname, sheet_name = "Sheet3 (2)", usecols="A:G")
        feedback_data = self.excel_to_memory(df_excel)

        holder = []
        for idx in range(len(feedback_data)):
            holder.append((feedback_data[idx][0], self.get_cur_viz(feedback_data[idx][0], raw_data), feedback_data[idx][2], feedback_data[idx][4]))
            # print(holder[idx])
        ret = []
        start_time = 0
        cur_viz = holder[0][1]
        sum_time = 0
        for row in holder:
            # pdb.set_trace()
            if cur_viz != row[1]:
                ret.append((cur_viz, row[0] - start_time))
                sum_time += row[0] - start_time
                start_time = row[0]
                cur_viz = row[1]

        if cur_viz == holder[len(holder) - 1]:
            ret.append((cur_viz, holder[len(holder) - 1] - start_time))
            sum_time += holder[len(holder) - 1] - start_time

        ret_list = []
        for row in ret:
            viz = row[0]
            time_seconds = int(row[1])
            time_mins = round(int(row[1]) / 60, 2)
            time_percent = round(time_seconds / sum_time, 2)
            ret_list.append([user, viz, time_seconds, time_mins, time_percent])
        return ret_list

    def get_file_tableau(self):

        uname = []
        with open('all_user_viz_time_track.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["User", "Visualization", "Time_Spent (Seconds)", "Time_Spent (Minutes)", "Percentage_time"])
            for raw_fname in self.raw_files:
                user = Path(raw_fname).stem.split('-')[0]
                uname.append(user)
                excel_fname = [string for string in self.excel_files if user in string][0]
                self.cum_rewards.clear()
                # merged = self.merge(user, raw_fname, excel_fname)
                ret = self.to_generate_tableau_plot(user, raw_fname, excel_fname)
                for r in ret:
                    writer.writerow(r)
        file.close()

if __name__ == "__main__":
    obj = integrate()
    obj.get_file_tableau()
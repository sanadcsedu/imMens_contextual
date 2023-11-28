#This version of the Mann Whitney test checks stationarity of the actions based on recieved rewards
import csv
import pdb
import glob
import random
import os
from pathlib import Path
import pandas as pd
import os
from collections import defaultdict

class read_data:
    def __init__(self, vizs):
        path = os.getcwd()
        if len(vizs) == 4: #Access the FAA dataset
            self.excel_files = glob.glob(path + '/FeedbackLog/*faa.xlsx')
            self.raw_files = glob.glob(path + '/RawInteractions/faa_data/*.csv')
        else: #Access the Brightkite dataset
            self.excel_files = glob.glob(path + '/FeedbackLog/*brightkite.xlsx')
            self.raw_files = glob.glob(path + '/RawInteractions/brightkite_data/*.csv')

    def get_files(self):
        ret = []
        uname = []
        for raw_fname in self.raw_files:
            user = Path(raw_fname).stem.split('-')[0]
            uname.append(user)
            excel_fname = [string for string in self.excel_files if user in string][0]
            merged = self.merge(raw_fname, excel_fname)
            ret.append(merged)
        return ret, uname

    def excel_to_memory(self, df):
        data = []
        for index, row in df.iterrows():
            mm = row['time'].minute
            ss = row['time'].second
            seconds = mm * 60 + ss
            if row['type'] == "interface" or row['type'] == 'simulation' or row['type'] == 'none':
                continue
            if row['type'] == 'recall':
                data.append([seconds, 'observation'])
            else:
                data.append([seconds, row['type']])
        return data

    def raw_to_memory(self, csv_reader):
        # data[0] = time, data[1] = action, data[2] = visualization
        next(csv_reader)
        data = []
        for lines in csv_reader:
            time = lines[1].split(":")
            mm = int(time[1])
            ss = int(time[2])
            seconds = mm * 60 + ss
            data.append([seconds, lines[2], lines[3]])
        return data

    #Used for merging the raw interaction (reformed) files with the Excel feedback files
    #we are also going to use this function to calculate the cumulative rewards (probability distribution)
    def merge(self, raw_fname, excel_fname):
        raw_interaction = open(raw_fname, 'r')
        csv_reader = csv.reader(raw_interaction)
        raw_data = self.raw_to_memory(csv_reader)
        # for idx, rows in enumerate(raw_data):
        #     print(idx, rows)
        # print("---------")
        raw_interaction.close()

        df_excel = pd.read_excel(excel_fname, sheet_name="Sheet3 (2)", usecols="A:G")
        feedback_data = self.excel_to_memory(df_excel)
        # for rows in feedback_data:
        #     print(rows)
        # print(raw_fname, len(raw_data))
        # print(excel_fname, len(feedback_data))
        holder = []
        idx = 0
        idx2 = 0
        for idx in range(len(feedback_data)):
            # pdb.set_trace()
            while idx2 < len(raw_data) and feedback_data[idx][0] >= raw_data[idx2][0] :
                # 0: index, 1: action, 2: visualization, 3: high_level_state, 4: reward
                if(feedback_data[idx][1] == 'observation'):
                    reward = 0.1
                else:
                    reward = 1
                holder.append([idx2, raw_data[idx2][1], raw_data[idx2][2], feedback_data[idx][1], reward])
                # print(holder[idx2])
                idx2 += 1
            if len(holder) > 1:
                holder[idx2 - 1][4] = 1
            idx += 1
        # for items in holder:
        #     print(items)
        return holder

if __name__ == "__main__":
    obj = read_data()
    print(obj.raw_files)
    print(obj.excel_files)
    obj.get_files()

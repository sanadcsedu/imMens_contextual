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

class contextual_bandit:
    def __inti__(self):
        # self.train_data = []
        pass 

    def get_data_into_pandas(self, data, thres):
        train_data = pd.DataFrame(columns=['action', 'cost', 'probability', 'feature1', 'feature2'])
        action_space = {'pan': 1, 'zoom': 2, 'brush': 3, 'range select': 4}
        vis_space = {'bar-4': 1, 'bar-2': 2, 'scatterplot-0-1': 3, 'hist-3': 4}
        
        l = int(len(data) * thres)
        for idx in range(0, l):
            # 0: index, 1: action, 2: visualization, 3: high_level_state, 4: reward 
            train_data = pd.concat([train_data, pd.DataFrame({
                'action': [vis_space[data[idx][2]]],
                'cost': data[idx][4],
                'probability': 0.25,
                'feature1': data[idx][3],
                # 'feature2': [action_space[data[idx][1]]],
                'feature2': data[idx][1],
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
                'action': [vis_space[data[idx][2]]],
                'cost': data[idx][4],
                'probability': 0.25,
                'feature1': data[idx][3],
                # 'feature2': [action_space[data[idx][1]]],
                'feature2': data[idx][1],
            })], ignore_index=False)

        test_df = pd.DataFrame(test_data)
        # Add index to data frame
        test_df["index"] = range(1, len(test_df) + 1)
        test_df = test_df.set_index("index")
        return test_df

    def train(self, vw, train_df):
        # train_df = self.get_data_into_pandas()
        for i in train_df.index:
            action = train_df.loc[i, "action"]
            cost = train_df.loc[i, "cost"]
            probability = train_df.loc[i, "probability"]
            feature1 = train_df.loc[i, "feature1"]
            feature2 = train_df.loc[i, "feature2"]
            # feature3 = train_df.loc[i, "feature3"]

            # Construct the example in the required vw format.
            learn_example = (str(action) + " | " + str(feature1) + " " + str(feature2))
            # Here we do the actual learning.
            vw.learn(learn_example)

        return vw
     
    def test(self, vw, test_df):
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
    vw = vowpalwabbit.Workspace("--cb_explore 4", quiet=True)
    threshold = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    for thres in threshold:
        cb = contextual_bandit()
        df_train = cb.get_data_into_pandas(data, thres)
        # pdb.set_trace()
        vw = cb.train(vw, df_train)
        df_test = cb.get_data_into_pandas_test(data, thres)
        # pdb.set_trace()
        accu = cb.test(vw, df_test)
        print("Threshold {:.1f} Testing Accuracy {:.2f}", thres, accu)
    
if __name__ == "__main__":     
    env = read_data()
    
    for raw_fname in env.raw_files:
        user = Path(raw_fname).stem.split('-')[0]
        excel_fname = [string for string in env.excel_files if user in string][0]
        data = env.merge(raw_fname, excel_fname)
        print(user, len(data))
        run_cb(data)
        break
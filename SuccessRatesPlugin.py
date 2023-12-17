import numpy as np
import pandas as pd

def compute_success_rate(method, ascending, df, all_targets, target2):
    top1 = 0
    top10 = 0
    top25 = 0
    top100= 0
    top200 = 0
    for target in all_targets:
        target_df = df[df[target2]==target]
        target_df = target_df.sort_values(by=method, ascending=ascending).reset_index(drop=True)
        if 1 in list(target_df[:1]['label']):
             top1+=1
        if 1 in list(target_df[:10]['label']):
            top10+=1
        if 1 in list(target_df[:25]['label']):
            top25+=1
        if 1 in list(target_df[:100]['label']):
             top100+=1
        if 1 in list(target_df[:200]['label']):
            top200+=1
    all_success_rate = [top1, top10, top25, top100, top200]
    #all_success_rate = [top10, top25, top200]

    all_success_rate = [int(x*100/len(all_targets)) for x in all_success_rate]
#     print(method)
#     print(all_success_rate)
    #print(f"Top1: {all_success_rate[0]} Top10: {all_success_rate[1]}; Top25: {all_success_rate[[2]]}; Top100: {all_success_rate[3]}; Top200: {all_success_rate[4]}")
    return all_success_rate


#compute_success_rate('PISToN', True)
import PyPluMA
import PyIO
import pickle

class SuccessRatesPlugin:
    def input(self, inputfile):
       self.parameters = PyIO.readParameters(inputfile)
    def run(self):
        pass
    def output(self, outputfile):
       inpickle = open(PyPluMA.prefix()+"/"+self.parameters["pickle"], "rb")
       df = pickle.load(inpickle)
       target = self.parameters['target']
       all_targets = list(df[target].unique())
       methods = PyIO.readSequential(PyPluMA.prefix()+"/"+self.parameters["methods"])
       #methods = ['PISToN', 'dMaSIF', 'iSCORE', 'DeepRank_GNN', 'DeepRank', 'HADDOCK', 'GNN_DOVE', 'DOVE']
       all_scores = []
       for method in methods:
          if method=='DeepRank_GNN' or method=='GNN_DOVE' or method=='DOVE' or method=='dMaSIF':
            all_scores.append([method]+compute_success_rate(method, False, df, all_targets, target))
          else:
            all_scores.append([method]+compute_success_rate(method, True, df, all_targets, target))

       df_top = pd.DataFrame(all_scores, columns=['model','top1', 'top10', 'top25', 'top100', 'top200'])
       #df_top
       print(df_top)

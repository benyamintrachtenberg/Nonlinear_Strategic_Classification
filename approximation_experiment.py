import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from math import comb

import pickle
import os
import warnings
import multiprocessing as mp
import concurrent.futures
import argparse
import time
from traceback import print_exc
import datetime

class best_acc:
      def __init__(self, n=100, n_features=2, class_separation=1, random_state=0, budget=2):
        self.n = n
        self.class_separation = class_separation
        self.random_state = random_state
        self.budget = budget
        
        self.linear_param = None
        self.strat_pos = None
        
        ### Generate Data ###
        self.X, self.Y = make_blobs(n_samples=n, centers=[[0,0],[class_separation,0]], n_features=n_features, random_state=random_state,shuffle=False)

      def plot_data(self, directory):
        fig, ax = plt.subplots(1, 1)
        scatter = ax.scatter(self.X[:,0], self.X[:,1], c=self.Y)

        # produce a legend with the unique colors from the scatter
        legend1 = ax.legend(*scatter.legend_elements())
        ax.add_artist(legend1)

        if self.linear_param is not None:
            x = [-1, self.class_separation+1]
            y = [x[0]*self.linear_param[0] + self.linear_param[1], x[1]*self.linear_param[0] + self.linear_param[1]]
            ax.plot(x,y)
        
        if self.strat_pos is not None:
            for pt in self.strat_pos:
              c = plt.Circle(pt, self.budget, alpha=0.2, ec='k')
              ax.add_artist( c )

        ax.set_xlim(min(self.X[:,0]) -1, max(self.X[:,0])+1 )
        ax.set_ylim(min(self.X[:,1]) -1, max(self.X[:,1])+1 )
        
        dir = directory + '/plots'
        os.makedirs(dir, exist_ok=True)
        plt.savefig(dir + f'/b={self.budget}_cs={self.class_separation}_rs={self.random_state}.png')
        plt.show()

      def best_linear_acc(self,):
        best_acc = 0 
        self.linear_param = [0,0]

        for a, x1 in enumerate(self.X):
          for b, x2 in enumerate(self.X):
            if a==b: continue
            slope = (x2[1]-x1[1]) / (x2[0]-x1[0])
            intercept = x1[1] - slope * x1[0]

            count=0
            for i, x in enumerate(self.X):
              pred = int( (x[1] - (slope*x[0]+intercept)) >=0 )

              if i==a or i==b: count +=1
              else: count += pred==self.Y[i]
          

            acc = count/self.n
            if acc > best_acc: 
              best_acc = acc
              self.linear_param = [slope, intercept]
        
        return best_acc
      
      def get_options(self, grid_step):
        max_x = max(self.X[:,0]) + self.budget
        max_y = max(self.X[:,1]) + self.budget
        min_x = min(self.X[:,0]) - self.budget
        min_y = min(self.X[:,1]) - self.budget
        options = []
        centers = []

        for x in np.arange(min_x, max_x, grid_step):
          for y in np.arange(min_y, max_y, grid_step):
            pos = []
            added_pos = False
            for i, pt in enumerate(self.X):
              if (pt[0]-x)**2 + (pt[1]-y)**2 <= self.budget**2:
                pos.append(i)
                if self.Y[i] == 1: added_pos= True
            
            if added_pos and pos not in options: 
              options.append(pos)
              centers.append((x,y))
        
        return options, centers

      def get_acc(self,options, subset, centers):
        i = len(subset)
        num = self.n//2
        pos = np.zeros(num)
        c_used=[]

        for j, o in enumerate(options):
          go_next = False
          for n in o: 
            if self.Y[n]==0 and n not in subset: 
              go_next = True
              break
          if go_next: continue

          changed = False
          for n in o: 
            if self.Y[n]==1 and pos[n-num]!=1: 
              pos[n-num] = 1
              changed=True
          
          if changed:c_used.append(centers[j])

        return num - i + sum(pos), c_used
      
      def next_subset(self, subset, max_idx):
        if subset[-1] < max_idx: 
          subset[-1] += 1
        else:
          subset[:-1] = self.next_subset(subset[:-1], max_idx-1)
          subset[-1] = subset[-2]+1
        return subset

      def best_strat_acc(self,grid_step=0.1):
        options, centers = self.get_options(grid_step = grid_step)
        num = self.n//2
        best_count = 0

        #pbar = tqdm(range(self.n//2), leave=False)
        #for i in pbar:
        #  pbar.set_description(f'Best Acc = {best_count/self.n}')
        for i in range(self.n//2):
          if best_count >= self.n - i: break
          subset = [j for j in range(i)]
          count, c_used = self.get_acc(options, subset, centers)

          if count > best_count: 
            best_count = count
            self.strat_pos = c_used

          for j in range(comb(num, i) - 1):
            subset = self.next_subset(subset, num-1)
            count, c_used = self.get_acc(options, subset, centers)

            if count > best_count: 
              best_count = count
              self.strat_pos = c_used
            
        return best_count/self.n


### Function to Run Single Experiment Instance ###      
def run(args):
    n, n_features, class_separation, random_state, budget, grid_step, directory = args
    test = best_acc(n=n, n_features=n_features, class_separation=class_separation, random_state=random_state, budget=budget)
    lin_acc = test.best_linear_acc()
    strat_acc = test.best_strat_acc(grid_step=grid_step)
    test.plot_data(directory)
    
    return class_separation, random_state, budget, lin_acc, strat_acc
    
    
if __name__ == '__main__':
    ### Fixed Parameters ###
    n_features=2
    grid_step=0.01
    max_workers = 100
    
    ### Experiment Hyperparameters ###  
    n = 50
    budget=[0.5,1,2]
    times = 20
    
    min_cs = 0
    max_cs = 5
    step_cs = 0.1
    class_separation = np.arange(min_cs, max_cs, step_cs)
    
    
    directory = f'results/data_exp/n={n}_times={times}/budget={budget}'
    os.makedirs(directory, exist_ok=True)
    
    lin_acc = {b: np.zeros((len(class_separation), times)) for b in budget}
    strat_acc = {b: np.zeros((len(class_separation), times)) for b in budget}
    
    ### Run Experiment Over Multiple CPUs ###
    args = [(n, n_features, cs, rs, b, grid_step, directory) for cs in class_separation for rs in range(times) for b in budget]
    with concurrent.futures.ProcessPoolExecutor(max_workers = max_workers) as executer:
        results = executer.map(run, args)

    for result in results:
        cs, rs, b, l_acc, s_acc = result
        cs_idx = int(round((cs-min_cs)/step_cs))
        lin_acc[b][cs_idx][rs] = l_acc
        strat_acc[b][cs_idx][rs] = s_acc

    avg_lin_acc = {b: np.mean(lin_acc[b], 1) for b in budget}
    avg_strat_acc = {b: np.mean(strat_acc[b], 1) for b in budget}
    
    ### Save Data ###
    with open(directory + '/lin_acc.pkl', 'wb') as f:
          pickle.dump(lin_acc, f)   
    with open(directory + '/strat_acc.pkl', 'wb') as f:
          pickle.dump(strat_acc, f)    
    with open(directory + '/avg_lin_acc.pkl', 'wb') as f:
          pickle.dump(avg_lin_acc, f)   
    with open(directory + '/avg_strat_acc.pkl', 'wb') as f:
          pickle.dump(avg_strat_acc, f)   


    ### Plot And Save Accuracy Plots ###
    for b in avg_strat_acc.keys():
        plt.plot(class_separation, avg_strat_acc[b])
    
    plt.plot(class_separation, avg_lin_acc[list(avg_lin_acc.keys())[0]])
    plt.legend(list(avg_strat_acc.keys()) + ["Linear"])
    plt.xlabel("Class Separation")
    plt.ylabel("Accuracy")
    plt.savefig(directory + '/acc_plot.png')
    plt.show()
        
        
        
    
    

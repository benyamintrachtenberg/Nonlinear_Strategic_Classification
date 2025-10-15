import numpy as np
import pandas as pd
pd.set_option('display.width', 140)
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import pickle
import os
from sklearn import svm

from sklearn.inspection import DecisionBoundaryDisplay
import warnings
import multiprocessing as mp
import concurrent.futures

import argparse
import time
from traceback import print_exc
import datetime

class multinomial_overfit:
      def __init__(self, degree=2, max=10, num_points=10, C=100, grid_step=0.01, max_init=10, max_iter=1000000, scale=10, epsilon=0.001, budget=2, multi_core=False):
        self.max = max
        self.max_init = max_init
        self.degree = degree
        self.num_points = num_points
        self.C = C
        self.scale = scale
        self.epsilon = epsilon
        self.grid_step = grid_step
        self.max_iter= int(max_iter)
        self.budget = budget
        self.multi_core = multi_core
        self.outerLoopMultiprocess=False
        self.generate_classifier()

      def generate_classifier(self):
        ### Point to overfit to ###
        self.poly_pts = np.random.uniform(-self.max_init, self.max_init, (self.num_points,2))
        self.poly_pts_labels = self.scale - 2 * self.scale * np.random.random_sample(size=self.num_points)
        
        ### Force at least one point to be in each class so that not all points [-100,100]^2 have the same label ###
        assert self.num_points >= 2
        self.poly_pts_labels[0] = -1* abs(self.poly_pts_labels[0])
        self.poly_pts_labels[1] = abs(self.poly_pts_labels[1])
        
        ### Generate classifier ###
        clf = svm.SVR(kernel="poly", degree=self.degree, C=self.C, coef0=1, max_iter=self.max_iter, epsilon=self.epsilon)
        self.classifier = clf.fit(self.poly_pts, self.poly_pts_labels)

      def eval(self, X):
         return self.classifier.predict(X)

      def make_grid(self):
        a, b = np.mgrid[-self.max:self.max+self.grid_step:self.grid_step, -self.max:self.max+self.grid_step:self.grid_step]
        a = a.reshape(-1,1)
        b = b.reshape(-1,1)
        self.num_rows = np.sqrt(len(a)).astype(int)
        return np.hstack((a,b))

      def generate_data(self, num=1000, fast=0, random=False, strat=True):
          self.data = self.make_grid()
          self.labels = self.label_data(self.data)
          self.num_data = len(self.labels)
          if strat: self.strat_labels = self.grid_strat_label_data(self.data)

      def label_data(self, X):
         y = self.eval(X)
         labelling = np.array(y>= 0)
         return labelling.astype(int)

      def grid_strat_label_data(self, data):
        c = self.budget
        num_rows = self.num_rows
        num_change = np.floor(c/self.grid_step).astype(int)
        labels = copy.deepcopy(self.labels).reshape(num_rows, num_rows)

        can_move = np.zeros_like(labels)
        for i in range(num_rows):
          for j in range(num_rows):
            if labels[i,j]==1:
              if i>0 and j>0 and i<num_rows-1 and j<num_rows-1 and labels[i-1,j]==1 and labels[i+1,j]==1 and labels[i,j-1]==1 and labels[i,j+1]==1 :
                continue
              else:
                for x in range(max(0,i-num_change), min(num_rows, i+num_change+1)):
                  x_shift = i-x
                  max_y_shift = np.floor(np.sqrt((c/self.grid_step)**2-x_shift**2)).astype(int)
                  for y in range(max(0,j-max_y_shift), min(num_rows, j+max_y_shift+1)):
                    can_move[x,y]=1

        return ((labels + can_move)>0).astype(int).reshape(-1)

      def sparse_subset(self, sparseness=0.9, border_sparseness=0, show=False, strat=True):
        num_rows = self.num_rows
        use=np.zeros_like(self.labels).astype(bool)
        rands = np.random.rand(self.num_data)

        labels = copy.deepcopy(self.labels).reshape(num_rows, num_rows)
        if strat: strat_labels = copy.deepcopy(self.strat_labels).reshape(num_rows, num_rows)

        idx=0
        for i in range(num_rows):
          for j in range(num_rows):
            if strat:
              if i==0 or j==0 or i==num_rows-1 or j==num_rows-1:
                use[idx]= (rands[idx] > sparseness)
              elif strat_labels[i,j]==1 and strat_labels[i-1,j]==1 and strat_labels[i+1,j]==1 and strat_labels[i,j-1]==1 and strat_labels[i,j+1]==1:
                use[idx]= (rands[idx] > sparseness)
              elif strat_labels[i,j]==0 and strat_labels[i-1,j]==0 and strat_labels[i+1,j]==0 and strat_labels[i,j-1]==0 and strat_labels[i,j+1]==0:
                use[idx]= (rands[idx] > sparseness)
              else:
                use[idx]= (rands[idx] > border_sparseness)
            else:
              if i==0 or j==0 or i==num_rows-1 or j==num_rows-1:
                use[idx]= (rands[idx] > sparseness)
              elif labels[i,j]==1 and labels[i-1,j]==1 and labels[i+1,j]==1 and labels[i,j-1]==1 and labels[i,j+1]==1:
                use[idx]= (rands[idx] > sparseness)
              elif labels[i,j]==0 and labels[i-1,j]==0 and labels[i+1,j]==0 and labels[i,j-1]==0 and labels[i,j+1]==0:
                use[idx]= (rands[idx] > sparseness)
              else:
                use[idx]= (rands[idx] > border_sparseness)

            idx+=1

        if strat:
          self.sparse_strat_labels = copy.deepcopy(self.strat_labels[use])
          self.sparse_strat_data = copy.deepcopy(self.data[use,:])
        else:
          self.sparse_data = copy.deepcopy(self.data[use,:])
          self.sparse_labels = copy.deepcopy(self.labels[use])

        if show: self.plot_boundary(title="{}% Sparse Data".format(100*sparseness), sparse=True)
      
      
########################### For Final Experiments ###########################
      def get_strat_acc(self, d):
        clf = svm.SVC(kernel='poly', degree=d, C=self.C, coef0=1, max_iter=self.max_iter, verbose=False)
        deg_d_classifier = clf.fit(self.sparse_strat_data, self.sparse_strat_labels)
        return deg_d_classifier.score(self.sparse_strat_data, self.sparse_strat_labels)

      def reg_fit_bad(self, real_degree, reg_thr):
        if all(self.labels) or all(1-self.labels): 
          print(f'Degree {real_degree}, seed {self.seed}. all labels are {int(all(self.labels))}.')
          return 0, True
        if all(self.sparse_labels) or all(1-self.sparse_labels): 
          print(f'Degree {real_degree}, seed {self.seed}. all sparse labels are {int(all(self.sparse_labels))}. regular labels are {100-100*np.mean(self.labels)}% 0 and {100*np.mean(self.labels)}% 1')
          return 0, True
        clf = svm.SVC(kernel='poly', degree=real_degree, C=self.C, coef0=1, max_iter=self.max_iter, verbose=False)
        deg_d_classifier = clf.fit(self.sparse_data, self.sparse_labels)
        acc = deg_d_classifier.score(self.sparse_data, self.sparse_labels)
        return acc, acc < reg_thr
            
      def run_experiment_iter(self, args):
        real_degree, budget, reg_thr, strat_thr, num, strat, sparseness, border_sparseness, seed = args
        self.degree = real_degree
        self.budget = budget
        self.seed = seed
        np.random.seed(seed)
        successive_fits = {}
        
        ### Set initial number of points to d+2 choose 2 ###
        self.num_points = int((real_degree+2)*(real_degree+1)/2)
        
        ### Generate classifier. Regen if poor ###
        with warnings.catch_warnings(record=True) as warning_list:
          self.generate_classifier()
          self.generate_data(num=num, strat=True)
          self.sparse_subset(sparseness=sparseness, border_sparseness=border_sparseness, strat=False)
          reg_acc, regen = self.reg_fit_bad(real_degree, reg_thr)

          regen_count = 0
          while(all(self.labels) or all(1-self.labels) or regen):
              self.generate_classifier()
              self.generate_data(num=num, strat=True)
              self.sparse_subset(sparseness=sparseness, border_sparseness=border_sparseness, strat=False)
              reg_acc, regen = self.reg_fit_bad(real_degree, reg_thr)
              regen_count += 1
          
          self.sparse_subset(sparseness=sparseness, border_sparseness=border_sparseness, strat=True)
          
          if strat_thr == 0:
            strat_thr = reg_acc
          successive_fits["reg"] = reg_acc
          successive_fits["thr"] = strat_thr
          
          try: successive_fits["d-1"] = self.get_strat_acc(real_degree-1)
          except Exception as e: 
                print(f'Exception happened for d-1={real_degree} seed={seed} budget={budget} d={real_degree}' )
                print ('type is:', e.__class__.__name__)
                print_exc()
    
          
          ### Find degree d that best fits strategic data ###
          d = real_degree
          try:
            acc = self.get_strat_acc(d)
            successive_fits[d] = acc
          except Exception as e: 
                print(f'Exception happened for real_degree={real_degree} seed={seed} budget={budget} d={d}' )
                print ('type is:', e.__class__.__name__)
                print_exc()
                return budget, real_degree, seed, 0, successive_fits, len(warning_list), regen_count
            
          if acc >= strat_thr:
            while acc >= strat_thr and d > 0:
              d -= 1
              acc = self.get_strat_acc(d)
              successive_fits[d] = acc
            if acc < strat_thr: d +=1
          else:
            while acc < strat_thr and d <self.degree_fit_upper_limit:
              d += 1
              try : 
                acc = self.get_strat_acc(d)
                successive_fits[d] = acc
              except Exception as e: 
                print(f'Exception happened for real_degree={real_degree} seed={seed} budget={budget} d={d}' )
                print ('type is:', e.__class__.__name__)
                print_exc()
                break
              
            if d==self.degree_fit_upper_limit: print(f'Hit strategic fit upper limit of {self.degree_fit_upper_limit}  for real_degree={real_degree} seed={seed} budget={budget}' )
          
          return budget, real_degree, seed, d, successive_fits, len(warning_list), regen_count
        
      def run_experiment(self, times=1, min_degree=1, max_degree=10, budget=[2], reg_thr=0, strat_thr=0, start_time=time.time(), job_id="Current Job"):
        # basic parameters
        self.max = 100
        self.max_init = 90
        self.grid_step = 2
        self.num = 10000
        sparseness = 1
        border_sparseness = 0
        max_workers = 100 
        
        self.multi_core = True
        self.outerLoopMultiprocess=True
        strat = True
        self.degree_fit_upper_limit = 30
        
        # hyperparameters
        self.max_iter = int(1e9)
        self.epsilon = 0.0001
        self.C = 1000
        self.num_points = 100

        # reg options: constant, constant based on deg, avg fit of degree
        # strat options: constant, constant based on deg, reg fit, avg fit of degree
        if reg_thr[0] == 0:
          reg_thresholds = [0.993,  0.986,  0.990,  0.985,  0.984,  0.981,  0.974,  0.962,  0.948, 0.922]
        elif len(reg_thr) == 1:
          reg_thresholds = reg_thr + np.zeros(max_degree-min_degree+1)
          
        if strat_thr[0] == -1:
          strat_thresholds = [0.993,  0.986,  0.990,  0.985,  0.984,  0.981,  0.974,  0.962,  0.948, 0.922]
        if strat_thr[0] == -2:
          strat_thresholds = [0.943,  0.936,  0.940,  0.935,  0.934,  0.931,  0.924,  0.912,  0.898, 0.872]
        elif len(strat_thr) == 1:
          strat_thresholds = strat_thr + np.zeros(max_degree-min_degree+1)
          
        
        fits = {b:np.zeros((max_degree-min_degree+1, times)) for b in budget}
        warns = {b:np.zeros(max_degree-min_degree+1) for b in budget}
        regens = {b:np.zeros(max_degree-min_degree+1) for b in budget}
        successive_degree_fits = {b:{d:{} for d in range(min_degree, max_degree+1)} for b in budget}
        
        ### Run each instance in parallel ###
        args = [(d, b, reg_thresholds[d-min_degree], strat_thresholds[d-min_degree], self.num, strat, sparseness, border_sparseness, t) for d in range(min_degree,max_degree+1) for t in range(times) for b in budget]
        with concurrent.futures.ProcessPoolExecutor(max_workers = max_workers) as executer:
          results = executer.map(self.run_experiment_iter, args)

          for result in results:
            b, real_deg, t, fit_deg, successive_fits, warn, regen = result
            fits[b][real_deg-min_degree, t] = fit_deg
            successive_degree_fits [b][real_deg][t] = successive_fits
            warns[b][real_deg-min_degree] += warn
            regens[b][real_deg-min_degree] += regen

        avg_fits = {b: np.mean(fits[b], 1) for b in budget}
        median_fits = {b: np.median(fits[b], 1) for b in budget}
        
        
        ### Save Results ###
        if self.max_iter==1e9: self.directory = f'results/reg_thr={reg_thr}_strat_thr={strat_thr}/1e9_times={times}_range={min_degree}:{max_degree}_budget={budget}'
        else: self.directory = f'results/reg_thr={reg_thr}_strat_thr={strat_thr}/times={times}_range={min_degree}:{max_degree}_budget={budget}'
        
        os.makedirs(self.directory, exist_ok=True)
        with open(self.directory + '/all_fits.pkl', 'wb') as f:
          pickle.dump(fits, f)     
        with open(self.directory + '/avg_fits.pkl', 'wb') as f:
          pickle.dump(avg_fits, f)  
        with open(self.directory + '/median_fits.pkl', 'wb') as f:
          pickle.dump(median_fits, f)  
        with open(self.directory + '/successive_fits.pkl', 'wb') as f:
          pickle.dump(successive_degree_fits, f)  
          
        self.print_params(times=times, min_degree=min_degree, max_degree=max_degree, budget=budget, reg_thr=reg_thr, strat_thr=strat_thr, start_time=start_time, job_id=job_id, warns = warns, regens=regens)    
        self.plot_results(times=times, min_degree=min_degree, max_degree=max_degree, budget=budget, reg_thr=reg_thr, strat_thr=strat_thr)
        
      def plot_results(self, times=1, min_degree=1, max_degree=10, reg_thr=0, strat_thr=0, budget =[2]):
        with open(self.directory + '/avg_fits.pkl', 'rb') as f:
          avg_fits = pickle.load(f)
          
        with open(self.directory + '/median_fits.pkl', 'rb') as f:
          median_fits = pickle.load(f)
        
        true_fits = range(min_degree, max_degree+1)

        for b in avg_fits.keys():
          plt.plot(true_fits, avg_fits[b], marker="o")

        plt.plot(true_fits, true_fits, "-")
        plt.legend(list(avg_fits.keys()) + ["Baseline"])
        plt.title(f'reg_thr={reg_thr} strat_thr={strat_thr} times={times} range={min_degree}:{max_degree} budget={budget}')
        plt.xlabel("Real Degree")
        plt.ylabel("Strategic Degree")
        plt.savefig(self.directory + '/fits_plot.png')
        plt.show()
        
        for b in median_fits.keys():
          plt.plot(true_fits, avg_fits[b], marker="o")

        plt.plot(true_fits, true_fits, "-")
        plt.legend(list(median_fits.keys()) + ["Baseline"])
        plt.title(f'reg_thr={reg_thr} strat_thr={strat_thr} times={times} range={min_degree}:{max_degree} budget={budget}')
        plt.xlabel("Real Degree")
        plt.ylabel("Strategic Degree")
        plt.savefig(self.directory + '/median_fits_plot.png')
        plt.show()

      def print_params(self, times, min_degree, max_degree, budget, reg_thr, strat_thr,  start_time, job_id, warns, regens):
        total_time = time.time() - start_time
        hours = total_time // 3600
        minutes = (total_time - 3600*hours) // 60
        seconds = total_time - 3600*hours - 60*minutes
        
        if reg_thr == 0: reg_threshold = "average fit accuracy of degree"
        else: reg_threshold = reg_thr
        if strat_thr ==0: strat_threshold = "regular fit accuracy"
        else: strat_threshold = strat_thr
        
        f = open(self.directory + f'/{job_id}.txt', "w")
        f.write(f' JOB ID             : {job_id} \n TOTAL TIME         : {hours} hours {minutes} minutes {seconds} seconds \n TIMES RUN          : {times} \n MAX ITER           : {self.max_iter} \n MIN DEGREE         : {min_degree} \n MAX DEGREE         : {max_degree} \n BUDGET             : {budget} \n REGULAR THRESHOLD  : {reg_threshold} \n STRATEGIC THRESHOLD: {strat_threshold} \n WARNINGS           : {warns} \n REGENS             : {regens}')
        f.close()

####################### Get Data For Single Instance #######################
      def instance_iter(self, args):
        real_degree, budget, reg_thr, num, strat, sparseness, border_sparseness, seed = args
        self.degree = real_degree
        self.budget = budget
        self.seed = seed
        np.random.seed(seed)
        
        ### Set initial number of points to d+2 choose 2 ###
        self.num_points = int((real_degree+2)*(real_degree+1)/2)

        ### Generate classifier. Regen if poor ###
        with warnings.catch_warnings(record=True) as warning_list:
          self.generate_classifier()
          self.generate_data(num=num, strat=True)
          self.sparse_subset(sparseness=sparseness, border_sparseness=border_sparseness, strat=False)
          reg_acc, regen = self.reg_fit_bad(real_degree, reg_thr)

          while(all(self.labels) or all(1-self.labels) or regen):
              self.generate_classifier()
              self.generate_data(num=num, strat=True)
              self.sparse_subset(sparseness=sparseness, border_sparseness=border_sparseness, strat=False)
              reg_acc, regen = self.reg_fit_bad(real_degree, reg_thr)
        
        self.sparse_subset(sparseness=sparseness, border_sparseness=border_sparseness, strat=True)
        
        return budget, self.labels, self.strat_labels, self.sparse_data, self.sparse_labels, self.sparse_strat_data, self.sparse_strat_labels

      def one_instance(self, deg=10, budget=[2], seed=3, reg_thr=0.0):
        # basic parameters
        self.max = 100
        self.max_init = 90
        self.grid_step = 2
        self.num = 10000
        sparseness = 1
        border_sparseness = 0
        max_workers = 100
        
        self.multi_core = True
        self.outerLoopMultiprocess=True
        strat = True
        self.degree_fit_upper_limit = 50
        
        # hyperparameters
        self.max_iter = int(1e9)
        self.epsilon = 0.0001
        self.C = 1000
        self.num_points = 100

        # reg options: constant, constant based on deg, avg fit of degree
        # strat options: constant, constant based on deg, reg fit, avg fit of degree
        f = [0.993,  0.986,  0.990,  0.985,  0.984,  0.981,  0.974,  0.962,  0.948, 0.922]
        
        if reg_thr == 0:
          reg_thr = f[deg-1]
        
        ### Fet results and save them ###
        args = [(deg, b, reg_thr, self.num, strat, sparseness, border_sparseness, seed)  for b in budget]
        dir = f'results/instances/deg={deg}_seed={seed}'
        os.makedirs(dir, exist_ok=True)
              
        with concurrent.futures.ProcessPoolExecutor(max_workers = max_workers) as executer:
          results = executer.map(self.instance_iter, args)

          for result in results:
            b, l, s, sd, sl, ssd, ssl = result
          
            with open(dir + f'/reg_labels.pkl', 'wb') as f:
              pickle.dump(l, f)   
            with open(dir + f'/b={b}_strat_labels.pkl', 'wb') as f:
              pickle.dump(s, f)  
            with open(dir + f'/b={b}_sparse_data.pkl', 'wb') as f:
              pickle.dump(sd, f)  
            with open(dir + f'/b={b}_reg_sparse_labels.pkl', 'wb') as f:
              pickle.dump(sl, f)   
            with open(dir + f'/b={b}_strat_sparse_data.pkl', 'wb') as f:
              pickle.dump(ssd , f)   
            with open(dir + f'/b={b}_strat_sparse_labels.pkl', 'wb') as f:
              pickle.dump(ssl, f)    
          
        print(f'finished {deg}, {seed}')

      
      
################### Testing And Hyperparameter Selection ###################
      def fit_single_degree(self, arg):
          degree = arg[0]
          labels = arg[1]
          clf = svm.SVC(kernel='poly', degree=degree, C=self.C, coef0=1, max_iter=self.max_iter, verbose=False)
          deg_d_classifier = clf.fit(self.sparse_data, labels)
          return deg_d_classifier.score(self.sparse_data, labels)
      
      def fit_degrees(self, strat=True, min_degree=1, max_degree=10):
        if self.multi_core==False or self.outerLoopMultiprocess==True:
          reg_fit = strat_fit = np.zeros(max_degree-min_degree+1)
          for d in tqdm(range(min_degree, max_degree+1), leave=False):
            reg_fit[d-min_degree] = self.fit_single_degree((d, self.sparse_labels))
          
          if strat:
            for d in tqdm(range(min_degree, max_degree+1), leave=False):
              strat_fit[d-min_degree] = self.fit_single_degree((d,self.sparse_strat_labels))
        
        else:
          if strat: 
            args = [(i, self.sparse_labels) for i in range(min_degree, max_degree+1)] +  [(i, self.sparse_strat_labels) for i in range(min_degree, max_degree+1)]
          else:
            args = [(i, self.sparse_labels) for i in range(min_degree, max_degree+1)]
            strat_fit = np.zeros(max_degree-min_degree+1)
            
          pool = mp.Pool()
          fit = np.array(pool.map(self.fit_single_degree, args))
          pool.close()
          reg_fit = fit[:max_degree-min_degree+1]
          if strat: strat_fit = fit[max_degree-min_degree+1:]
        return reg_fit, strat_fit

      def get_fits(self, min_degree=1, max_degree=10, sparseness=0.9, border_sparseness=0, show=False, strat=True):
        self.sparse_subset(sparseness=sparseness, border_sparseness=border_sparseness, show=show, strat=strat)
        return self.fit_degrees(strat=strat, min_degree=min_degree, max_degree=max_degree)

      def print_fits(self, min_degree=1, max_degree=10, sparseness=0.9, border_sparseness=0):
        reg_fits, strat_fits = self.get_fits(min_degree=min_degree, max_degree=max_degree, sparseness=sparseness, border_sparseness=border_sparseness, show=True)
        text = "\n Polynomial Degree:"
        for i in range(min_degree, max_degree+1): text+= "    " + str(i) + " "
        print(text)
        print("Regular   Fit Acc:", np.round(reg_fits,3))
        print("Strategic Fit Acc:", np.round(strat_fits,3))

      def plot_boundary(self, title="", sparse=False):
         if sparse:
            data = self.sparse_data
            labels = self.sparse_labels
            strat_labels = self.sparse_strat_labels
         else:
            data = self.data
            labels = self.labels
            strat_labels = self.strat_labels

         colormap = np.array(['red', 'orange', 'blue'])
         color = labels + strat_labels

         plt.scatter(data[:,0], data[:,1], s=10, c = colormap[color], marker="o")

         plt.xlim((-self.max, self.max))
         plt.ylim((-self.max, self.max))
         plt.title(title)
         plt.show()

      def fits_once(self, args):
        d, num, strat, sparseness, border_sparseness, seed = args
        self.degree = d
        if seed != None: np.random.seed(seed)
        with warnings.catch_warnings(record=True) as warning_list:
          self.generate_classifier()
          self.generate_data(num=num, fast=100, random=False, strat=strat)
          while(all(self.labels) or all(1-self.labels)):
              self.generate_classifier()
              self.generate_data(num=num, fast=100, random=False, strat=strat)
          reg_fits, strat_fits = self.get_fits(min_degree=d-1, max_degree=d,sparseness=sparseness, border_sparseness=border_sparseness, strat=strat, show=False)
        
        out = np.array([
          int(reg_fits[1] > 0.9),
          int(reg_fits[1] > 0.95),
          int(reg_fits[1] > 0.99),
          int(reg_fits[1] > 0.95 and reg_fits[0] < 0.95),
          int(reg_fits[1] > 0.95 and reg_fits[0] < 0.9),
          int(reg_fits[1] > 0.99 and reg_fits[0] < 0.99),
          int(reg_fits[1] > 0.99 and reg_fits[0] < 0.95),
          reg_fits[0],
          reg_fits[1],
          reg_fits[1] - reg_fits[0],
          len(warning_list)
        ])
      
        return (d, out)
                
      def fits_well(self, num_points=[100], C=[10], min_degree=1, max_degree=10, num=10000, times=5, strat=False, sparseness=0.95, border_sparseness=0,):
        for c in C:
          self.C = c
          for n in num_points:
            self.num_points = n
            out = np.zeros((10, max_degree-min_degree+1))
            num_warn = 0

            if self.multi_core==False:
              for d in range(min_degree,max_degree+1):
                self.degree = d
                for t in range(times):
                  o = self.fits_once((d, num, strat, sparseness, border_sparseness, None))[1]
                  out[:,d-min_degree] += o[:10]
                  num_warn += o[10]
            else:
              args = [(d, num, strat, sparseness, border_sparseness, d+t*max_degree) for d in range(min_degree,max_degree+1) for t in range(times)]
              self.outerLoopMultiprocess=True
              
              with concurrent.futures.ProcessPoolExecutor() as executer:
                results = executer.map(self.fits_once, args)
                for result in results:
                  d,o = result
                  out[:,d-min_degree] += o[:10]
                  num_warn += o[10]    
                  
            out = np.round(out/times,3)
            print()
            print("####### C={}   num_points={}   max_iter={}   eps={} (Fit didn't converge {} times)#######".format(c,n, self.max_iter, self.epsilon, num_warn))
            print()
            self.display_table(out, min_degree, max_degree)

      def display_table(self, out, min, max):
        idx = ["d > 0.9:", "d > 0.95:", "d > 0.99:", "d > 0.95 and d-1 < 0.95:", "d > 0.95 and d-1 < 0.9:", "d > 0.99 and d-1 < 0.99:", "d > 0.99 and d-1 < 0.95:", "Average d-1 fit acc:", "Average d fit acc:", "Average difference:"]
        data = {d: out[:,d-min] for d in range(min, max+1)}
        
        df = pd.DataFrame(data, index=idx)
        print(df)
        
      def num_pts_ablation(self, num_points=[100], C=[10], min_degree=1, max_degree=10, num=10000, times=5, strat=False, sparseness=0.95, border_sparseness=0,):
          for d in range(min_degree,max_degree+1):
            print()
            print("####### Degree {} Polynomial #######".format(d))
            self.degree = d
            text = ""
            for i in range(max(1, d-2), d+3): text+= "   " + str(i) + "  "
            for c in C:
              self.C = c
              for n in num_points:
                self.num_points = n
                r = np.zeros(d+3 - max(1, d-2))
                s = np.zeros(d+3 - max(1, d-2))
                count=0
                for t in range(times):
                  self.generate_classifier()
                  self.generate_data(num=num, fast=100, random=False, strat=strat)

                  while(all(self.labels) or all(1-self.labels)):
                    count+=1
                    self.generate_classifier()
                    self.generate_data(num=num, fast=100, random=False, strat=strat)
                  #self.plot_orig()
                  #self.plot_boundary()
                  reg_fits, strat_fits = self.get_fits(min_degree=max(1, d-2), max_degree=d+2,sparseness=sparseness, border_sparseness=border_sparseness, strat=strat)
                  r += reg_fits
                  s += strat_fits
                print()
                print("C={}  num_points={} (needed to regen {} times)".format(c, n, count))
                print(text)
                prRed(np.round(r/times,3), d+3-max(1, d-2))
                prRed(np.round(s/times,3), d+3-max(1, d-2))

      def plot_orig(self, n=1, support_vectors=True):
        ### https://scikit-learn.org/stable/auto_examples/svm/plot_svm_kernels.html ###
        cmap = matplotlib.colors.ListedColormap(['red', 'blue'])

        if n==1:
            _, axs = plt.subplots()
            expanded_x = np.concatenate((self.poly_pts,[[self.max,self.max],[-self.max,-self.max]]))
            common_params = {"estimator": self.classifier, "X": expanded_x, "ax": axs}
            DecisionBoundaryDisplay.from_estimator( **common_params, response_method="predict", plot_method="pcolormesh", alpha=0.2, cmap=cmap)
            #DecisionBoundaryDisplay.from_estimator( **common_params, response_method="decision_function", plot_method="contour", levels=[0], colors=["k"], linestyles=["-"],)
            DecisionBoundaryDisplay.from_estimator( **common_params, response_method="predict", plot_method="contour", levels=[0], colors=["k"], linestyles=["-"],)
            if support_vectors:
              axs.scatter( self.classifier.support_vectors_[:, 0], self.classifier.support_vectors_[:, 1],
                              s=150, facecolors="none", edgecolors="k",)
            scatter = axs.scatter(self.poly_pts[:,0], self.poly_pts[:,1],  s=10, c=self.poly_pts_labels, cmap=cmap, marker = "o")
            plt.show()
        else:
            _, axs = plt.subplots(n//5, 5, figsize=(15, 0.6*n))
            x_min, x_max, y_min, y_max = -self.max, self.max, -self.max, self.max

            for i in range(n//5):
              for j in range(5):
                self.generate_classifier()
                expanded_x = np.concatenate((self.poly_pts,[[self.max,self.max],[-self.max,-self.max]]))

                # Plot decision boundary and margins
                common_params = {"estimator": self.classifier, "X": expanded_x, "ax": axs[i,j]}

                DecisionBoundaryDisplay.from_estimator( **common_params, response_method="predict", plot_method="pcolormesh", alpha=0.2, cmap=cmap)
                DecisionBoundaryDisplay.from_estimator( **common_params, response_method="decision_function", plot_method="contour",levels=[0], colors=["k"], linestyles=["-"],)
                #DecisionBoundaryDisplay.from_estimator( **common_params, response_method="predict", plot_method="contour", levels=[0], colors=["k"], linestyles=["-"],)

                if support_vectors:
                    # Plot bigger circles around samples that serve as support vectors
                    axs[i,j].scatter( self.classifier.support_vectors_[:, 0], self.classifier.support_vectors_[:, 1],
                                s=150, facecolors="none", edgecolors="k",)

                # Plot samples by color and add legend
                scatter = axs[i,j].scatter(self.poly_pts[:,0], self.poly_pts[:,1],  s=10, c=self.poly_pts_labels, cmap=cmap, marker = "o")
                #axs[i,j].legend(*scatter.legend_elements(), loc="upper right", title="Classes")
                #axs[i,j].set(xlim=(x_min, x_max), ylim=(y_min, y_max))
            plt.show()
      

if __name__ == '__main__':
    ### Parse Script Arguments ###
    parser = argparse.ArgumentParser()

    parser.add_argument("--times", type=int)
    parser.add_argument("--min", type=int)
    parser.add_argument("--max", type=int)
    parser.add_argument("--bud")
    parser.add_argument("--reg_thr", nargs="*", type=float)
    parser.add_argument("--strat_thr", nargs="*", type=float)
    parser.add_argument("--j", type=int)

    args = parser.parse_args()

    times = args.times
    min_degree = args.min
    max_degree = args.max
    budget = [int(b) for b in args.bud.split(sep=",")]
    reg_thr = args.reg_thr
    strat_thr = args.strat_thr
    job_id = args.j
    
    start_time = time.time()
    
    ### Run Experiment ###
    mult_over = multinomial_overfit()
    mult_over.run_experiment(times=times, min_degree=min_degree, max_degree=max_degree, budget=budget, reg_thr=reg_thr, strat_thr=strat_thr, start_time=start_time, job_id=job_id)
    
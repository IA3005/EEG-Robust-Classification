#Required libraries
import mne
import numpy as np
import matplotlib.pyplot as plt
import scipy
import os
from collections import OrderedDict
import seaborn as sns
import pandas as pd
import gzip
from scipy.signal import filtfilt, butter
import pickle
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
from pyriemann.tangentspace import TangentSpace
from estimation import covariances
#from riemannian_geometry import mean_riemann, distance_riemann
from itertools import combinations,product
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.covariance import shrunk_covariance


          

def SsvepLoading(data_path):
    """
    Inputs: 
        data_path: path for the ssvep recordings
        
    Outputs:
        subj_list : a list of subjects ["subject1",...]
        records : a dictionnary of subjects and their associated sessions
    """
    subj_list = os.listdir(data_path)
    records = {s: [] for s in range(len(subj_list))}
    for s in range(len(subj_list)):
        subj = subj_list[s]
        record_all = os.listdir(data_path+subj+'/')
        n = len(record_all)//4 #number of records of a given subject
        for i in range(n):
            records[s].append(record_all[i*4][:28])
    return subj_list,records
 



def filter_bandpass(signal, lowcut, highcut, fs, order=4, filttype='forward-backward'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    if filttype == 'forward':
        filtered = lfilter(b, a, signal, axis=-1)
    elif filttype == 'forward-backward':
        filtered = filtfilt(b, a, signal, axis=-1)
    else:
        raise ValueError("Unknown filttype:", filttype)    
    return filtered    
    
    
    
class TrialsBuilding():
    
    
    def __init__(self,data_path,records,subj_list,subject,nb_classes):
        self.records = records
        self.data_path = data_path
        self.subj_list = subj_list
        self.subject = subject
        self.nb_classes = nb_classes
        assert 0<= self.subject <len(self.subj_list),"The selected subject does not exist in the dataset"
        self.nb_total_sessions = len(self.records[self.subject])
        #experimental setup
        self.tmin=2
        self.tmax=5
        self.sfreq = 256
        self.freq_band=0.1
        self.frequencies= [13,17,21]
        if self.nb_classes==4:
            self.event_code = [33024,33025,33026,33027]
            self.names=['resting','stim13','stim21','stim17']
        else:
            self.event_code = [33025,33026,33027]
            self.names=['stim13','stim21','stim17']
        self.channels = np.array(['Oz','O1','O2','PO3','POz','PO7','PO8','PO4'])
        
        
    def load_single_session(self,subject,session):
        chosen_subject = self.subj_list[subject]
        fname = chosen_subject+'/'+self.records[subject][session]
        with gzip.open(self.data_path + fname + '.pz', 'rb') as f:
            o = pickle.load(f, encoding='latin1')
        raw_signal = o['raw_signal'].T
        event_pos = o['event_pos'].reshape((o['event_pos'].shape[0]))
        event_type = o['event_type'].reshape((o['event_type'].shape[0]))
        return raw_signal,event_pos,event_type
    
    
    def make_extended_trials_single_session(self,raw_signal,event_pos,event_type):
        ext_signal = np.empty_like(raw_signal[0,:])    
        for f in self.frequencies:
            ext_signal = np.vstack((ext_signal, filter_bandpass(raw_signal, f-self.freq_band,
                                                                     f+self.freq_band, fs=self.sfreq)))
        ext_signal = ext_signal[1:,:]
        
        ext_trials = list()
        for i in range(1,len(event_type)):
            boolean = (event_type[i] == 32779) and (event_type[i-1] in self.event_code) # start of a trial
            if boolean: 
                t = event_pos[i]
                start = t + self.tmin*self.sfreq
                stop  = t + self.tmax*self.sfreq
                ext_trials.append(ext_signal[:, start:stop])
        ext_trials = np.array(ext_trials)
        ext_trials = ext_trials - np.tile(ext_trials.mean(axis=2).reshape(ext_trials.shape[0], 
                                    ext_trials.shape[1], 1), (1, 1, ext_trials.shape[2]))
        return ext_trials
    
        
    def make_labels_single_session(self,event_type):
        labels = []
        n_events = len(self.event_code)
        for e in event_type:
            for i in range(n_events):
                if e==self.event_code[i]:
                    labels.append(i)
        return labels
    
    
    def extended_trials_and_labels_all_sessions(self):
        ext_trials_all_sessions = []
        labels_all_sessions = [] #list of length 32*total_nb_sessions
        
        for session in range(len(self.records[self.subject])):
            raw_signal,event_pos,event_type = self.load_single_session(self.subject,session)
            ext_trial_single = self.make_extended_trials_single_session(raw_signal,event_pos,event_type)
            labels = self.make_labels_single_session(event_type)
            ext_trials_all_sessions.append(ext_trial_single)
            labels_all_sessions.extend(labels) 
        
        n_trials_per_session,dim1,dim2 = ext_trial_single.shape #(32,24,768) or (24,24,768)
        extended_trials = np.zeros((self.nb_total_sessions*n_trials_per_session,dim1,dim2)) 
        #shape : (32*total_nb_sessions,24,768) if nb_clases=4 
        #shape : (24*total_nb_sessions,24,768) if nb_clases=3
        
        for i in range(len(ext_trials_all_sessions)):
            extended_trials[n_trials_per_session*i:n_trials_per_session*(i+1),:,:]= ext_trials_all_sessions[i]
        
        ch = "There is a problem of shapes : "+str(len(labels_all_sessions))+" != "+str(extended_trials.shape[0])
        assert len(labels_all_sessions)==extended_trials.shape[0],ch
        
        return extended_trials,labels_all_sessions    



class Covariances():
    
    def __init__(self,estimator,estimator_params=None):
        self.estimator = estimator
        self.estimator_params = estimator_params
        
    def transform(self,extended_trials):
        cov_ext_trials = covariances(extended_trials, self.estimator,self.estimator_params)
        return cov_ext_trials
    
    
class Classify():

    def __init__(self,method,covs,labels,nb_trains,nb_classes):
        self.covs = covs #all covs , for test and trainn
        self.labels = labels
        self.method = method
        assert method in ["MDM","TangentSpace"]
        self.nb_classes = nb_classes
        self.nb_total_trials = covs.shape[0]
        self.nb_total_sessions = self.nb_total_trials//(8*self.nb_classes) 
        self.nb_trains = nb_trains
        #if nb_trains=0, we choose a session and we split into train and test sets
        #if nb_trains=nb_total_sessions, we take nb_total_sessions-1 and split the last session into train and test
        assert 0<= self.nb_trains <= self.nb_total_sessions,"Make sure that the nbr of training sets is <= to the nber of sessions"
        
        
    def classifier(self, x_train, y_train ):
        if self.method=="MDM":
            return MDM(x_train,y_train,self.nb_classes)
        if self.method=="TangentSpace":
            return Tangent_Space(x_train,y_train,self.nb_classes)
            
           
    def split(self,session=0,train_per_class=7,max_folds=10):
        trains_idx , tests_idx = [],[]
        sessions = list(range(self.nb_total_sessions))
        
        if (1<=self.nb_trains<self.nb_total_sessions):
            test_sessions_idx = list(combinations(sessions,self.nb_total_sessions-self.nb_trains))
            for i in range(len(test_sessions_idx)):
                test_idx , train_idx = [],[]
                test_session_idx = test_sessions_idx[i]
                for j in test_session_idx:
                    test_idx.extend(list(range(j*8*self.nb_classes,(j+1)*8*self.nb_classes)))
                for k in range(len(self.labels)):
                    if not(k in test_idx):
                        train_idx.append(k)
                trains_idx.append(train_idx)
                tests_idx.append(test_idx)
            
              
        if self.nb_trains==0:
            assert 0<=session<self.nb_total_sessions#session to split
            #balanced cross_validation
            classes_idx = { i :[] for i in range(self.nb_classes)}   
            labels_session = self.labels[session*8*self.nb_classes:(session+1)*8*self.nb_classes]
            for j in range(len(labels_session)) : 
                classes_idx[labels_session[j]].append(j)
            for i in range(max_folds):
                train_idx, test_idx  = [],[]
                boolean = True
                while boolean:
                    for c in self.nb_classes:
                        test_idx.append(list(np.random.choice(classes_idx[c],size=8-train_per_class,replace=False)))
                    boolean = test_idx in tests_idx
                for j in range(self.nb_total_trials):
                    if not(j in test_idx):
                        train_idx.append(j)
                trains_idx.append(train_idx)
                tests_idx.append(test_idx)
         
        
        if self.nb_trains==self.nb_total_sessions:
            assert 0<=session<self.nb_total_sessions#session to split
            test_sessions_idx = list(combinations(sessions,self.nb_total_sessions-self.nb_trains))
            for i in range(len(test_sessions_idx)):
                test_idx , train_idx = [],[]
                test_session_idx = test_sessions_idx[i]
                for j in test_session_idx:
                    test_idx.extend(list(range(j*8*self.nb_classes,(j+1)*8*self.nb_classes)))
                for k in range(len(labels)):
                    if not(k in test_idx):
                        train_idx.append(k)
                trains_idx.append(train_idx)
                tests_idx.append(test_idx)
            train_idx, test_idx  = [],[]
            boolean = True
            while boolean:
                for c in self.nb_classes:
                    test_idx.append(list(np.random.choice(classes_idx[c],size=8-train_per_class,replace=False)))
                boolean = test_idx in tests_idx
                for j in range(self.nb_total_trials):
                    if not(j in test_idx):
                        train_idx.append(j)
            for i in range(self.nb_trains):
                trains_idx[i].extend(train_idx)
                tests_idx[i].extend(test_idx)
             
        return trains_idx,tests_idx
    
    
    def listing(self,covs,labels,indices):
        x,y= [],[]
        for i in indices:
            x.append(covs[i,:,:])
            y.append(labels[i])
        return x,y
        
        
    def score(self, predicted_labels,true_labels):
        assert len(predicted_labels)==len(true_labels),"True and predicted labels' lists haven't the same length"
        acc = 0
        for j in range(len(predicted_labels)):
            if predicted_labels[j]==true_labels[j]:
                acc +=1
        return acc/len(predicted_labels)
    
    
    def accuracies(self):
        trains_idx,tests_idx = self.split()
        accuracies_train = []
        accuracies_test  = []
        
        for i in range(len(trains_idx)):
            train_idx , test_idx = trains_idx[i],tests_idx[i]
            covs_train, true_labels_train = self.listing(self.covs,self.labels,train_idx)
            classifier = self.classifier(covs_train, true_labels_train)
                
            predicted_labels_train = classifier.predict(covs_train)
            accuracy_train  = self.score(predicted_labels_train,true_labels_train)
            accuracies_train.append(accuracy_train)
            
            covs_test, true_labels_test = self.listing(self.covs,self.labels,test_idx)
            predicted_labels_test  = classifier.predict(covs_test)
            accuracy_test  = self.score(predicted_labels_test,true_labels_test)
            accuracies_test.append(accuracy_test)
         
        return np.asarray(accuracies_train),np.asarray(accuracies_test)
   
    
        
class MDM():
    
    def __init__(self,x_train,y_train,nb_classes):
        self.nb_classes = nb_classes
        self.cov_centers = self.MassCenters(x_train,y_train)
        
    def MassCenters(self,x_train,y_train):
        cov_centers = np.empty((self.nb_classes, x_train[0].shape[1], x_train[0].shape[1]))
        x_train_bis = np.empty((len(x_train),x_train[0].shape[1], x_train[0].shape[1]))
        for i in range(len(x_train)):
            x_train_bis[i,:,:] = x_train[i]
        classes = list(range(self.nb_classes))
        y_train_bis = np.asarray(y_train)
        for i, l in enumerate(classes):
            cov_centers[i, :, :] = mean_riemann(x_train_bis[y_train_bis==l,:,:])
        return cov_centers
    
    def argmin_distance(self,sample):
        min_dist = np.inf
        for i in range(self.nb_classes):
            dist = distance_riemann(sample, self.cov_centers[i])
            if min_dist > dist:
                min_dist = dist
                idx = i
        return idx
            
    def predict(self,x): 
        prediction = []
        for sample in x:
            predicted_label = self.argmin_distance(sample)
            prediction.append(predicted_label)
        return prediction  


class Tangent_Space():
                              
    def __init__(self,x_train,y_train,nb_classes):
        self.nb_classes = nb_classes
        self.tangent = TangentSpace(metric='riemann')
        self.clf = self.ProjectedClassifier(x_train,y_train)
       
    
    def project(self,covs_list):
        n_samples,size = len(covs_list),covs_list[0].shape[0]
        covs_array = np.zeros((n_samples,size,size))
        for i in range(n_samples):
            covs_array[i,:,:] = covs_list[i]
        proj_covs= self.tangent.fit_transform(covs_array)
        return proj_covs
    
    def ProjectedClassifier(self,x_train,y_train):
        x_train_proj = self.project(x_train)
        clf = LogisticRegression(random_state=0).fit(x_train_proj,y_train)
        return clf
    
    def predict(self,x):
        x_proj = self.project(x)
        return self.clf.predict(x_proj)
        


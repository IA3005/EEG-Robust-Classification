#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from sklearn.model_selection import KFold

class SSVEP():
    
    def __init__(self,data_path,selection):
        """
        Inputs: 
            data_path: path for the ssvep recordings
            selection : dict , keys are selected subjects and values are selected sessions for each selected subject
        """
        self.data_path = data_path
        self.selection = selection
        
        #Parameters: experimental setup
        self.tmin = 2
        self.tmax = 5
        self.sfreq = 256
        self.freq_band = 0.1
        self.frequencies = [13,17,21]
        self.nb_classes  = len(self.frequencies)+1
        self.event_code  = [33024,33025,33026,33027]
        self.channels   = np.array(['Oz','O1','O2','PO3','POz','PO7','PO8','PO4'])
       
        #organize data : into a dict
        self.subj_list = os.listdir(self.data_path)
        self.records = {k: [] for k in self.subj_list}
        for subj in self.subj_list:
            record_all = os.listdir(data_path+subj+'/')
            n = len(record_all)//4 #number of records of a given subject
            for i in range(n):
                self.records[subj].append(record_all[i*4][:28])
        
        #verify the format of the entries
        assert self.verify_selection_format(self.selection)==True, "Please verify the entries in your dictionnary"
        
        #number of selected recordings
        n = 0
        for subj in selection.keys():
            n += len(selection[subj])
        self.nb_selected_recordings = n
        
        
    def verify_selection_format(self,selection):
        for subj in selection.keys():
            if not(subj in range(len(self.subj_list))):
                return False
        for subj in selection.keys():
            for session in selection[subj]:
                if not(session in range(len(self.records[self.subj_list[subj]]))):
                    return False
        return True
        
    def load_data_single_recording(self,subject,session):
        chosen_subject = list(self.records.keys())[subject]
        fname = chosen_subject+'/'+self.records[chosen_subject][session]
        with gzip.open(self.data_path + fname + '.pz', 'rb') as f:
            o = pickle.load(f, encoding='latin1')
        raw_signal = o['raw_signal'].T
        event_pos = o['event_pos'].reshape((o['event_pos'].shape[0]))
        event_type = o['event_type'].reshape((o['event_type'].shape[0]))
        return raw_signal,event_pos,event_type
    
    
    def load_data_for_selection(self):
        selected_raw_signal= []
        selected_event_pos = []
        selected_event_type= []
        for subj in self.selection.keys():
            if len(self.selection[subj]) != 0:
                for session in range(len(self.selection[subj])):
                    raw_signal,event_pos,event_type = self.load_data_single_recording(subj,session)
                    selected_raw_signal.append(raw_signal)
                    selected_event_pos.append(event_pos)
                    selected_event_type.append(event_type)
        return selected_raw_signal, selected_event_pos, selected_event_type
    
    
    #The Butterworth filter : band-pass filter, flat in the passband , the passband is concentrated on 
    def filter_bandpass(self,signal, fmin, fmax, fs, order=4, filttype='forward-backward'):
        nyq = 0.5 * fs
        low = fmin / nyq
        high = fmax / nyq
        b, a = butter(order, [low, high], btype='band')
        #filter tpe : forwaard-backward
        filtered = filtfilt(b, a, signal, axis=-1)  
        return filtered
    
    
    def make_extended_trials_single(self,raw_signal,event_pos,event_type):
        ext_signal = np.empty_like(raw_signal[0,:])    
        for f in self.frequencies:
            ext_signal = np.vstack((ext_signal, self.filter_bandpass(raw_signal, f-self.freq_band,
                                                                     f+self.freq_band, fs=self.sfreq)))
        ext_signal = ext_signal[1:,:]
        ext_trials = list()
        for e, t in zip(event_type, event_pos):
            if e == 32779: # start of a trial
                start = t + self.tmin*self.sfreq
                stop  = t + self.tmax*self.sfreq
                ext_trials.append(ext_signal[:, start:stop])
        ext_trials = np.array(ext_trials)
        ext_trials = ext_trials - np.tile(ext_trials.mean(axis=2).reshape(ext_trials.shape[0], 
                                    ext_trials.shape[1], 1), (1, 1, ext_trials.shape[2]))
        return ext_trials
    
    
    def make_labels_single(self,event_type):
        labels = []
        n_events = len(self.event_code)
        for e in event_type:
            for i in range(n_events):
                if e==self.event_code[i]:
                    labels.append(i)
        return labels


    def extended_trials_and_labels_selection(self):
        selected_raw_signal, selected_event_pos, selected_event_type = self.load_data_for_selection()
        ext_trials_list = []
        all_labels = [] #list of length 32*total_nb_sessions
        
        for i in range(len(selected_raw_signal)):
            #ext_trial_single: matrix of shape (32,24,768)
            ext_trial_single = self.make_extended_trials_single(selected_raw_signal[i], selected_event_pos[i],
                                                                selected_event_type[i])
            ext_trials_list.append(ext_trial_single)
            all_labels.extend(self.make_labels_single(selected_event_type[i])) 
        
        self.n_trials_per_session,dim1,dim2 = ext_trial_single.shape
        extended_trials = np.zeros((self.nb_selected_recordings*self.n_trials_per_session,dim1,dim2)) 
        #shape : (32*total_nb_sessions,24,768)
        
        for i in range(len(ext_trials_list)):
            extended_trials[self.n_trials_per_session*i:self.n_trials_per_session*(i+1),:,:]= ext_trials_list[i]
            

        return extended_trials,all_labels
    
    
    def build_covariances(self,extended_trials,estimator='scm'):
        cov_ext_trials = Covariances(estimator=estimator).transform(extended_trials)
        return cov_ext_trials
    
    def GeometricCenters(self,x_train,y_train):
        cov_centers = np.empty((self.nb_classes, x_train.shape[1], x_train.shape[1]))
        x_trains=[[] for i in range(self.nb_classes) ]
        for i in range(self.nb_classes):
            for j in range(x_train.shape[0]):
                if y_train[j]==i:
                    x_trains[i].append(x_train[j,:,:])
        for i in range(self.nb_classes):
            x_trains[i]=np.asarray(x_trains[i])
        for i in range(self.nb_classes):
            cov_centers[i, :, :] = mean_riemann(x_trains[i])
        return cov_centers


    def accuracy(self,x,y,cov_centers):
        classes=list(range(self.nb_classes))
        accuracies = list()
        for sample, true_label in zip(x, y):
            dist = [distance_riemann(sample, cov_centers[m]) for m in range(self.nb_classes)]
            if classes[np.array(dist).argmin()] == true_label:
                accuracies.append(1)
            else: accuracies.append(0)
        accuracy_ = 100.*np.array(accuracies).sum()/len(y)
        return accuracy_
    


    def CovGeoMDM(self,n_splits,shuffle):
        #with cross-validation
        extended_trials, labels = self.extended_trials_and_labels_selection()
        cov_ext_trials = self.build_covariances(extended_trials)
        
        kf = KFold(n_splits=n_splits, shuffle=shuffle)
        train_accuracy, test_accuracy = [], []
        for train_index , test_index in kf.split(labels):
            x_train,x_test,y_train,y_test = [],[],[],[]

            for i in train_index:
                x_train.append(cov_ext_trials[i,:,:])
                y_train.append(labels[i])

            for i in test_index:
                x_test.append(cov_ext_trials[i,:,:])
                y_test.append(labels[i])

            x_train = np.asarray(x_train)
            x_test  = np.asarray(x_test)

            cov_centers = self.GeometricCenters(x_train,y_train)

            train_accuracy.append(self.accuracy(x_train,y_train,cov_centers))
            test_accuracy.append(self.accuracy(x_test,y_test,cov_centers))

        train_accuracy = np.asarray(train_accuracy)
        test_accuracy = np.asarray(test_accuracy)

        return train_accuracy,test_accuracy


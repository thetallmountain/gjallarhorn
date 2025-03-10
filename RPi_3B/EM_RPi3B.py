# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 16:36:52 2023

@author: anonymous
"""

# LIBRARIES

from sklearn.metrics import *
import time
import datetime
import numpy as np
import shutil
import serial
import matplotlib.pyplot as plt
import csv
import pandas as pd
import pyvisa as visa
from lecroy3 import *
from numpy import loadtxt
from itertools import chain
from itertools import zip_longest
import seaborn as sns
import scipy.integrate as integrate
from sklearn.impute import SimpleImputer
import scipy.signal  
from scipy.stats import pearsonr, spearmanr
from numpy.random import normal
import random
import scipy.stats as stats
from numpy import exp
from numpy import savetxt
from scipy.fft import ifft, fft, rfft
from sklearn.neighbors import KernelDensity
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EllipticEnvelope
from sklearn.covariance import MinCovDet
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, HDBSCAN, OPTICS, MeanShift, estimate_bandwidth, AffinityPropagation
from sklearn.utils.multiclass import unique_labels
from s_dbw import S_Dbw
import paramiko
from scp import SCPClient
import os.path
from threading import *
import os
# import ffmpeg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # <--- This is important for 3d plotting 
from scipy.stats import norm

import tkinter as tk
import tkinter.ttk as ttk
from tkinter.filedialog import *
import tkinter.filedialog as filedialog
from tkinter import StringVar
from tkinter import IntVar
from tkinter import LabelFrame
from tkinter import Label
from tkinter import Button
from tkinter import messagebox
from PIL import Image, ImageTk

# Warning configuration
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

SHOW_GRAPH = False

device = 'RPi3B'
my_ip = "X.X.X.X"
n_fuzz=1

date = "2024_01_23_14o38"

# Define paths based on extraction
base_path = './results/' + device + "_EM_bugs_" + date + '/'
calibration_path = base_path
entries_path = base_path
results_entries_path = base_path
ffmpeg_path = base_path + 'ffmpeg_files/'
save_path = base_path + 'output_figs/'

# Initialize a dictionary for mapping
mapeo_strings = {}

# Class labels
mapeo_bugs = {
    './Bugs_RPi4/SUT0101.bin': 'E0101',
    './Bugs_RPi4/SUT0102.bin': 'E0102',
    './Bugs_RPi4/SUT0103.bin': 'E0103',
    './Bugs_RPi4/SUT0104.bin': 'E0104',
    './Bugs_RPi4/SUT0105.bin': 'E0105',
    './Bugs_RPi4/SUT0106.bin': 'E0106',
    './Bugs_RPi4/SUT0201.bin': 'E0201',
    './Bugs_RPi4/SUT0202.bin': 'E0202',
    './Bugs_RPi4/SUT0203.bin': 'E0203',
    './Bugs_RPi4/SUT0204.bin': 'E0204',
    './Bugs_RPi4/SUT0205.bin': 'E0205',
    './Bugs_RPi4/SUT0206.bin': 'E0206',
    './Bugs_RPi4/SUT0207.bin': 'E0207',
    './Bugs_RPi4/SUT0208.bin': 'E0208',
    './Bugs_RPi4/SUT0209.bin': 'E0209',
    #'./Bugs_RPi4/SUT0210.bin': 'E0210',
    './Bugs_RPi4/SUT00I.bin': 'SUT00I',
    './Bugs_RPi4/SUT00F.bin': 'SUT00F'
}

# FIXED VARIABLES FOR TESTING
n_files = 3
pca_plot = 0
clust = 0
silh = 0

imp_model = 6

# GLOBAL PARAMETERS
gauss_dict = dict([(1, 0.682), (2, 0.954)])
val=1
input_signal = np.array([])
outlied_signal = np.array([])
robust_samples = np.array([])
pca_samples = np.array([])
silh_score = []
calinski_score = []
davies_score = []
sdbw_score = []
variance_score = []
variance_coefficient = []
max_n_components_pca = []
labels_array = []

def get_class_name(obj):
    return type(obj).__name__

def outlier_detection(kind):
    
    global n_samples
    global n_files
    global imp_model
    global outlied_signal
    
    outlied_signal = [[[0 for z in range(np.shape(input_signal)[2])] for y in range(np.shape(input_signal)[1])] for x in range(np.shape(input_signal)[0])]
    n_it = np.shape(outlied_signal)[0]
    
    for i in range(n_it):
        quartile_threshold=50
        indice_25000_ceros = 0
            
        if i<2:
            ceros_por_fila = np.count_nonzero(input_signal[i] == 0, axis=1)
            # Find the index where the number of zeros per row equals 25000
            indice_25000_ceros = np.where(ceros_por_fila == n_samples)[0]            
            
        if i==1:
            quartile_threshold=5
        
        for j in range(np.shape(outlied_signal)[1]):
            
            if i==2 or (i<2 and j not in indice_25000_ceros):
    
                q1 = pd.DataFrame(input_signal[i][j]).quantile(0.25)[0]
                q3 = pd.DataFrame(input_signal[i][j]).quantile(0.75)[0]
                iqr = q3 - q1 # Interquartile range
                fence_low = q1 - (quartile_threshold*iqr)
                fence_high = q3 + (quartile_threshold*iqr)
                
                if imp_model == 2: # Mean imputation
                        outlied_signal[i][j] = input_signal[i][j].copy()
                        outlied_signal[i][j][(input_signal[i][j] <= fence_low)]=np.mean(outlied_signal[i][j])
                        outlied_signal[i][j][(input_signal[i][j] >= fence_high)]=np.mean(outlied_signal[i][j])       
                
                elif imp_model == 3: # Median imputation
                        outlied_signal[i][j] = input_signal[i][j].copy()
                        outlied_signal[i][j][(input_signal[i][j] <= fence_low)]=np.median(outlied_signal[i][j])
                        outlied_signal[i][j][(input_signal[i][j] >= fence_high)]=np.median(outlied_signal[i][j])      
                
                elif imp_model == 4: # LOCF
                        signal = pd.DataFrame(input_signal[i][j].copy().reshape(1,-1)).astype(float)
                        signal.T[(input_signal[i][j] <= fence_low)]=np.nan
                        signal.T[(input_signal[i][j] >= fence_high)]=np.nan
                        outlied_signal[i][j] = signal.T.fillna(method='bfill').T.to_numpy()
                    
                elif imp_model == 5: # NOCB
                        signal = pd.DataFrame(input_signal[i][j].copy().reshape(1,-1)).astype(float)
                        signal.T[(input_signal[i][j] <= fence_low)]=np.nan
                        signal.T[(input_signal[i][j] >= fence_high)]=np.nan
                        outlied_signal[i][j] = signal.T.fillna(method='ffill').T.to_numpy()
                       
                elif imp_model == 6: # Linear interpolation
                        signal = pd.DataFrame(input_signal[i][j].copy().reshape(1,-1)).astype(float)
                        signal.T[(input_signal[i][j] <= fence_low)]=np.nan
                        signal.T[(input_signal[i][j] >= fence_high)]=np.nan
                        outlied_signal[i][j] = signal.T.interpolate(method='linear').T.to_numpy()
    
                elif imp_model == 7: # Spline interpolation
                        signal = pd.DataFrame(input_signal[i][j].copy().reshape(1,-1)).astype(float)
                        signal.T[(input_signal[i][j] <= fence_low)]=np.nan
                        signal.T[(input_signal[i][j] >= fence_high)]=np.nan
                        outlied_signal[i][j] = signal.T.interpolate(method='spline').T.to_numpy()
                      
                else: # Zeroes imputation
                        outlied_signal[i][j] = input_signal[i][j].copy()
                        outlied_signal[i][j][(input_signal[i][j] <= fence_low)]=0
                        outlied_signal[i][j][(input_signal[i][j] >= fence_high)]=0  
                        
                outlied_signal[i][j] = outlied_signal[i][j].ravel()

        
    outlied_signal = [np.atleast_1d(np.asarray(x,dtype=np.int64)) for x in outlied_signal] 
           
def pca_technique_application(kind):
    
    global n_traces
    global robust_samples
    global pca_samples
    global input_signal
    global max_n_components_pca
    global save_path
    
    pca_samples = np.array([[0 for y in range(np.shape(outlied_signal)[2])] for x in range(np.shape(outlied_signal)[0]*np.shape(outlied_signal)[1])], dtype=object)
          
    # PCA TRAINING
    datos = pd.DataFrame(np.transpose(outlied_signal[2]))
    scaler = StandardScaler()
    data_rescaled = scaler.fit_transform(datos)
    pca = PCA().fit(data_rescaled)
    xi = np.arange(1, np.shape(outlied_signal)[1]+1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)        
    plt.rcParams["figure.figsize"] = (20,6)
    fig, ax = plt.subplots()
    plt.ylim(0.0,1.1)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Components')
    plt.xticks(np.arange(1, np.shape(outlied_signal)[1]+1, step=1), rotation=90) #change from 0-based array index to 1-based human-readable label
    plt.ylabel('Cumulative variance (%)')
    plt.title('The number of components needed to explain variance ({})' .format(kind))
    plt.axhline(y=0.99, color='r', linestyle='-')
    plt.text(0.5, 1, '99% cut-off threshold', color = 'red', fontsize=16)
    ax.grid(axis='x')
    plt.savefig(save_path + device + "_entries_list_" + date + "_PCA_{}" .format(kind))
    plt.show()
    # Calculate the slope at each point (numerical differentiation)
    slope = np.diff(y) / np.diff(xi)
    
    # Find the indices where the slope decreases sharply while respecting the previous element
    sharp_slope_indices = []
    for i in range(1, len(slope)):
        if np.abs(slope[i] - slope[i - 1]) < 1e-4:  # (Adjust the threshold as needed)
            sharp_slope_indices.append(i)
            break
    
    if len(sharp_slope_indices)>0:
        n_components = sharp_slope_indices[0]+1
        
    plt.figure(dpi=100)
    plt.plot(xi, np.abs(np.gradient(y/(xi/n_traces),xi)))
    plt.axvline(x=n_components, color='r', linestyle='-')
    plt.title('Curve meaning absolute value of gradient of explained variance/(n_element/n_traces), for {}' .format(kind))
    plt.text(n_components, 0.1, n_components, color = 'red', fontsize=16)
    plt.savefig(save_path + device + "_entries_list_" + date + "_PCA_gradient_curve_{}" .format(kind))
    plt.show()
                
    max_n_components_pca = sharp_slope_indices[0]+1
    reshaped_robust_samples = np.array(outlied_signal).reshape(np.shape(outlied_signal)[0]*np.shape(outlied_signal)[1],np.shape(outlied_signal)[2])
   
    # REMOVE NULL TRACES
    # Count the number of zeros per row
    ceros_por_fila = np.count_nonzero(reshaped_robust_samples == 0, axis=1)

    # Find the index where the number of zeros per row equals 25000
    indice_25000_ceros = np.where(ceros_por_fila == n_samples)[0]
    
    reshaped_robust_samples = np.delete(reshaped_robust_samples, indice_25000_ceros, axis=0)
    
    # PCA APPLICATION TO BUGS
    print("Using {} components in PCA" .format(max_n_components_pca))
    pca = PCA(n_components=np.where(y>=0.99)[0][0]+2*cal_n_traces, whiten=False).fit(reshaped_robust_samples)
    pca_samples = pca.transform(reshaped_robust_samples)
       
def clustering_procedure(kind):
    
    global entries_path
    global entries_list
    global pca_samples
    global input_signal
    global labels_array
    global silh_score
    global calinski_score
    global davies_score
    global sdbw_score
    global variance_score
    global variance_coefficient
    global NOEXEC_samples
    global EXEC_samples
    global n_tests
    global instances
    global distances
    global y_pred
    global data
    global date
    global gauss_dict
    global val
    global entries_list
    global mapeo_strings
    global cal_n_traces
    
    print("Performing clustering...")
    silh_score = [np.nan for x in range(n_traces)]
    calinski_score = np.array([np.nan for x in range(n_traces)])
    davies_score = [np.nan for x in range(n_traces)]
    sdbw_score = [np.nan for x in range(n_traces)]
    labels_array = [[np.nan for y in range(2*cal_n_traces+n_traces)] for x in range(n_traces)]
    
    for index in range(1,n_traces+1):
    
        data = np.abs(rfft(pca_samples[:(2*cal_n_traces+index)]))
    
        instances = np.shape(data)[0]
        neighbors = NearestNeighbors(n_neighbors=n_tests).fit(data)
        distances, indices = neighbors.kneighbors(data)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        eps = distances[instances - n_tests - 1]
    
        if n_tests == 0:
            eps = distances[round(instances * 0.99)]
        
        # Clustering algorithms
        # db = DBSCAN(eps=eps, min_samples=1)            
        # db = HDBSCAN(cluster_selection_epsilon=eps, min_samples=1)
        # y_pred = db.fit_predict(data)            
        # bwth = estimate_bandwidth(data)
        # db = MeanShift(bandwidth=bwth)
        # y_pred = db.fit_predict(data)            
        db = AffinityPropagation()
        y_pred = db.fit_predict(data)
        
        labels_array[index-1] = db.labels_
            
    
    unique, counts = numpy.unique(np.array(entries_list), return_counts=True)
    print("REAL ENTRIES:", dict(zip(unique, counts)))
                           
    print("FINAL RESULTS")
    
    bugs_entries_list = [mapeo_bugs[valor] for valor in entries_list]
    etiquetas_unicas_true = np.unique(bugs_entries_list)
    etiquetas_unicas_pred = np.unique(y_pred[2 * cal_n_traces:])
    
    confus_matrix = np.zeros((len(etiquetas_unicas_true), len(etiquetas_unicas_pred)))
    for true_label, pred_label in zip(bugs_entries_list, y_pred[2 * cal_n_traces:]):
        confus_matrix[np.where(etiquetas_unicas_true == true_label)[0][0],
                      np.where(etiquetas_unicas_pred == pred_label)[0][0]] += 1
    
    y_pred_changed = np.copy(y_pred).astype(str)
    for i in range(len(etiquetas_unicas_true)):
        label = np.argmax(confus_matrix[i])
        rep = len(np.where(confus_matrix[i] == np.max(confus_matrix[i]))[0])
        for label in range(rep):
            index = np.where(confus_matrix[i] == np.max(confus_matrix[i]))[0][label]
            if isinstance(etiquetas_unicas_pred[index], np.int64):
                y_pred_changed[y_pred == etiquetas_unicas_pred[index]] = etiquetas_unicas_true[i]
    
    # Change labels not in bugs_entries_list to "UAN"
    for i in range(len(y_pred_changed)):
        if y_pred_changed[i] not in bugs_entries_list:
            y_pred_changed[i] = "UAN"
    
    etiquetas_unicas_pred = np.unique(y_pred_changed[2 * cal_n_traces:])
    valores_nuevos = np.setdiff1d(etiquetas_unicas_true, etiquetas_unicas_pred)
    etiquetas_unicas_pred = np.union1d(etiquetas_unicas_pred, valores_nuevos)
    etiquetas_unicas_pred = np.roll(etiquetas_unicas_pred, -np.where(etiquetas_unicas_pred == 'E0101')[0])
    
    confus_matrix = np.zeros((len(etiquetas_unicas_true), len(etiquetas_unicas_pred)))
    for true_label, pred_label in zip(bugs_entries_list, y_pred_changed[2 * cal_n_traces:]):
        confus_matrix[np.where(etiquetas_unicas_true == true_label)[0][0],
                      np.where(etiquetas_unicas_pred == pred_label)[0][0]] += 1
    
    plt.figure(figsize=(20, 12), dpi=100)
    plt.imshow(confus_matrix, cmap=plt.get_cmap('GnBu'), interpolation='nearest')
    for i in range(len(etiquetas_unicas_true)):
        for j in range(len(etiquetas_unicas_pred)):
            plt.text(j, i, str(int(confus_matrix[i, j])), ha="center", va="center", color="black", fontsize=22)
    plt.title('Confusion Matrix (EM), {}, Frequency Domain in {}'.format(get_class_name(db), device), fontsize=25)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=22)
    plt.xticks(np.arange(len(etiquetas_unicas_pred)), etiquetas_unicas_pred, rotation=90)
    plt.yticks(np.arange(len(etiquetas_unicas_true)), etiquetas_unicas_true)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.xlabel('Assigned Labels', fontsize=22)
    plt.ylabel('Actual Labels', rotation=90, verticalalignment='center', fontsize=22)
    plt.savefig(save_path + device + "_entries_list_" + date + "_CONFUSION_MATRIX_TIMEDOMAIN_" + str(get_class_name(db)) + "_PWR.png", bbox_inches='tight', dpi=100)
    plt.show()
    
    fig = plt.figure(figsize=(16,16), dpi=100)
    ax = fig.add_subplot(projection='3d')
    middle_idx = int(np.shape(data)[1] / 2)  # Middle index of data's axis 1
    sc = ax.scatter(data[:2*cal_n_traces, middle_idx-1], 
                    data[:2*cal_n_traces, middle_idx], 
                    data[:2*cal_n_traces, middle_idx+1], 
                    zdir='y', c=y_pred[:2*cal_n_traces], s=25, cmap='viridis', marker="*", 
                    label=y_pred_changed[:2*cal_n_traces])
    sc = ax.scatter(data[2*cal_n_traces+1:, middle_idx-1], 
                    data[2*cal_n_traces+1:, middle_idx], 
                    data[2*cal_n_traces+1:, middle_idx+1], 
                    zdir='y', c=y_pred[2*cal_n_traces+1:], s=25, cmap='viridis', marker="o", 
                    label=y_pred_changed[2*cal_n_traces+1:])
    ax.set_xlabel('$X$', fontsize=20)
    ax.set_ylabel('$Y$', fontsize=20)
    ax.set_zlabel('$Z$', fontsize=20, rotation=0)
    ax.legend(*sc.legend_elements(), title='Clusters')
    plt.title("AffinityPropagation clustering considering {} elements ({})" .format(2*cal_n_traces+index, kind))
    plt.savefig(save_path + device + "_entries_list_" + date + "_AffinityPropagation_" + kind + ".png", bbox_inches='tight', dpi=100)
    plt.show()
    
def operation(kind):
    
    global calibration_path
    global entries_path
    global root
    global details
    global clust
    global silh
    global pca_plot
    global imp_model
    global ip_address
    global port_list
    global channel
    global n_traces
    global n_samples
    global command_OP
    global input_signal
    global date
    global command
    global device
      
    exec_name = [filename for filename in os.listdir(calibration_path) if filename.startswith(device + "_exec_")][0]
    noexec_name = [filename for filename in os.listdir(calibration_path) if filename.startswith(device + "_noexec")][0]
    fuzz_name = device + "_EM_bugs_" + date + ".csv"
    
    EM_signal = np.loadtxt(calibration_path + "/" + exec_name,delimiter=',')
    EM_cal_traces = np.shape(EM_signal)[0]
    n_samples = np.shape(EM_signal)[1]
    
    input_signal = np.array([[[0 for z in range(n_samples)] for y in range(n_traces)] for x in range(n_files)])
     
    input_signal[0][:EM_cal_traces] = np.abs(np.loadtxt(calibration_path + "/" + exec_name,delimiter=','))
    input_signal[1][:EM_cal_traces] = np.abs(np.loadtxt(calibration_path + "/" + noexec_name, delimiter=','))
    input_signal[2] = np.abs(np.loadtxt(results_entries_path + fuzz_name, delimiter=','))
    
    print("Entries signals files: ", fuzz_name)
        
    outlier_detection(kind)   
    pca_technique_application(kind)
    clustering_procedure(kind)
    return

def bugs_capture():
    
    global pinata_ip
    global date    
    global entries_path

    entries_list = list(map(str,random.choices(input_info, k=n_traces)))
    now = datetime.datetime.now()
    date_arr = [str(now.day), str(now.month), str(now.year), str(now.hour), str(now.minute)]
    date_arr = [x.zfill(2) for x in date_arr]
    date = date_arr[0] + "_" + date_arr[1] + "_" + date_arr[2] + "_" + date_arr[3] + "o" + date_arr[4]
    with open(entries_path + "entries_list_" + date + '.csv', 'w') as txt_file:
        for line in entries_list:
            txt_file.write(line + "\n")
    os.system(entries_command + str(n_traces) + " " + str("entries_list_" + date + '.csv'))
    
def calibration(ssh):
    global calibration_command
    global cal_n_traces
    global device
    
    # CALIBRATION SIGNALS CAPTURE
    cal_command = "python3.11 /home/anonymous/Project_Calibration_Clustering/device_default_calibration.py "
    command = "a1"
    calibration_command = command
    os.system(cal_command + command + " " + str(cal_n_traces) + " " + device)
    print(ssh_stdout.read().decode())
    print(time.time()-start, "seconds to get calibration samples")
    
def entries_list_creation(entries):
    global entries_list
    global list_nfs_path
    global date
    global device
    
    # Array to store 20 repetitions of each string
    entries_list = []

    # Repeat each string 20 times and add it to the new array
    for string in entries:
        repeated_string = [string] * 20
        entries_list.extend(repeated_string)
            
    now = datetime.datetime.now()
    date_arr = [str(now.day), str(now.month), str(now.year), str(now.hour), str(now.minute)]
    date_arr = [x.zfill(2) for x in date_arr]
    date = date_arr[2] + "_" + date_arr[1] + "_" + date_arr[0] + "_" + date_arr[3] + "o" + date_arr[4]

    with open(entries_path + device + "_entries_list_" + date + '.csv', 'w') as txt_file:
        for line in entries_list:
            txt_file.write(line + "\n")
            
    print("Entries list created: ", device, '_entries_list_', date, '.csv')

start1 = time.time()

start = time.time()
# FUZZING ENTRIES SIGNALS CAPTURE
entries_command = "python3.11 data_collection.py "
cal_n_traces = 100
# calibration()

kind = "EM"

for it in range(n_fuzz):
    it_start = time.time()
    print("ITERATION", it+1)
    # bugs_capture() # BUGS CAPTURING
    # entries = list(mapeo_bugs.keys())
    # entries_list_creation(entries)
    
    with open(entries_path + device + "_entries_list_" + date + '.csv') as file:
        entries_list = [line.rstrip() for line in file]
    
    n_traces = np.shape(entries_list)[0]
    n_tests = int(np.sqrt(n_traces))
    
    operation(kind) # DATA PROCESSING + CLUSTERING
    print(time.time()-it_start, "seconds to perform fuzzing iteration", it+1)           
    
print(time.time()-start, "seconds to perform operation")
print(time.time()-start1, "seconds to perform whole process") 
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 15:40:14 2023

@author: atenea
"""

# LIBRARIES

from sklearn.metrics import *
import time
import datetime
import numpy as np
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
from sklearn.neighbors import NearestNeighbors
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, HDBSCAN, MeanShift, estimate_bandwidth, OPTICS
from sklearn.utils.multiclass import unique_labels
from s_dbw import S_Dbw
from scipy.stats import norm
import paramiko
from scp import SCPClient
import os.path
from threading import *
import os

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from collections import Counter

import zipfile

# Warning configuration
# ==============================================================================
import warnings
warnings.filterwarnings('ignore')

SHOW_GRAPH = False

stm_ip = "X.X.X.X"  # Anonymized STM32-F429ZI's IP address
server_ip = "Y.Y.Y.Y"      # Anonymized server's IP address
n_fuzz=1

device="STM32-F429ZI"      # Anonymized device name
date = "2024_04_17_15o16"

def unzip_results():
    zip_filename = 'STM32-F429ZI_EM_bugs_2024_04_17_15o16.zip'
    zip_path = './results/' + zip_filename
    # Extract to folder with same name as zip (without .zip extension)
    extract_folder = './results/' + zip_filename.replace('.zip', '')
    
    # Create extraction directory if it doesn't exist
    os.makedirs(extract_folder, exist_ok=True)
    
    # Create output_figs directory inside the extraction folder
    output_figs_path = os.path.join(extract_folder, 'output_figs')
    os.makedirs(output_figs_path, exist_ok=True)
    
    print(f"Extracting {zip_path} to {extract_folder}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)
    print("Extraction completed successfully.")
    print(f"Created output directory: {output_figs_path}")

# Call the unzip function before any other processing
unzip_results()

# Define paths based on extraction
base_path = "./results/" + device + "_EM_bugs_" + date + '/'
EM_calibration_path = base_path
EM_entries_nfs_path = base_path
list_nfs_path = base_path + device
ffmpeg_path = base_path + 'ffmpeg_files/'
save_path = base_path + 'output_figs/'

# FIXED VARIABLES FOR TESTING
n_files = 3
pca_plot = 0
clust = 0
silh = 0

imp_model = 6

# GLOBAL PARAMETERS
gauss_dict = dict([(1, 0.682), (2, 0.954)])
val=1
silh_score = []
calinski_score = []
davies_score = []
sdbw_score = []
variance_score = []
variance_coefficient = []
max_n_components_pca = []
labels_array = []

# Initialize a dictionary for mapping
mapeo_strings = {}
mapeo_bugs = {}

# Class labels
mapeo_strings = {
    '00': 1,
    '02': 2,
    '03': 3,
    '04': 4,
    '05': 5,
    '06': 6,
    '07': 7,
    '08': 8,
    '09': 9,
    '0a': 10,
    '0b': 11,
    '0c': 12,
    '0d': 13,
    '0e': 14,
    '0f': 15,
    'a3': 16,
    'a1': 0,
    'a0': 17
}
    

mapeo_bugs = {
    '00': 'E0101',
    '02': 'E0102',
    '03': 'E0103',
    '04': 'E0104',
    '05': 'E0105',
    '06': 'E0106',
    '07': 'E0201',
    '08': 'E0202',
    '09': 'E0203',
    '0a': 'E0204',
    '0b': 'E0205',
    '0c': 'E0206',
    '0d': 'E0207',
    '0e': 'E0208',
    '0f': 'E0209',
    'a3': 'E0210',
    'a0': 'SUT00I',
    'a1': 'SUT00F'
}

def get_class_name(obj):
    return type(obj).__name__

def robust_covariance(signal): # STAGE 3
    detector = EllipticEnvelope(contamination=0.1, assume_centered=True)
    return detector.fit(signal).predict(signal)
    
def outlier_detection(input_signal, kind):
    
    global n_samples
    global n_files
    global imp_model
    # global fence_low
    # global fence_high
    
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
                
                elif imp_model == 4: # LOCF (Last Observation Carried Forward)
                        signal = pd.DataFrame(input_signal[i][j].copy().reshape(1,-1)).astype(float)
                        signal.T[(input_signal[i][j] <= fence_low)]=np.nan
                        signal.T[(input_signal[i][j] >= fence_high)]=np.nan
                        outlied_signal[i][j] = signal.T.fillna(method='bfill').T.to_numpy()
                    
                elif imp_model == 5: # NOCB (Next Observation Carried Backward)
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
    return outlied_signal
        
def robust_covariance_procedure(outlied_signal, kind):
    
    global n_files
    global n_traces
    global n_samples

    robust_samples = np.array([[[0 for z in range(np.shape(outlied_signal)[2])] for y in range(np.shape(outlied_signal)[1])] for x in range(np.shape(outlied_signal)[0])])
    
    for test in range(np.shape(outlied_signal)[0]): 
            
            # REMOVE NULL TRACES
            # Count the number of zeros per row
            ceros_por_fila = np.count_nonzero(outlied_signal[test] == 0, axis=1)

            # Find the index where the number of zeros per row equals 25000
            indice_25000_ceros = np.where(ceros_por_fila == n_samples)[0]
                
            for i in range(np.shape(outlied_signal)[1]):
                if ceros_por_fila[i] == 0:
                    continue
                elif test==2 or (test<2 and i not in indice_25000_ceros):
                    robust_samples[test][i] = robust_covariance(np.transpose(outlied_signal[test][i, np.newaxis])) # STAGE 3
    
    return robust_samples

def pca_technique_application(robust_samples, kind):
    
    global n_traces
    global n_samples
    global max_n_components_pca
    global save_path
    global device
    global date
    global pca_samples
    
    pca_samples = np.array([[0 for y in range(np.shape(robust_samples)[2])] for x in range(np.shape(robust_samples)[0]*np.shape(robust_samples)[1])], dtype=object)   
       
    # PCA TRAINING
    datos = pd.DataFrame(np.transpose(robust_samples[2]))
    scaler = StandardScaler()
    data_rescaled = scaler.fit_transform(datos)
    pca = PCA().fit(data_rescaled)
    xi = np.arange(1, np.shape(robust_samples)[1]+1, step=1)
    y = np.cumsum(pca.explained_variance_ratio_)   
    plt.rcParams["figure.figsize"] = (10,6)
    fig, ax = plt.subplots()
    plt.ylim(0.0,1.10)
    plt.plot(xi, y, marker='o', linestyle='--', color='b')
    plt.xlabel('Number of Components', fontsize=20)
    #plt.xticks(np.arange(1, np.shape(robust_samples)[1]+1, step=1), rotation=90, fontsize=5) #change from 0-based array index to 1-based human-readable label
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.ylabel('Cumulative variance (%)', fontsize=20)
    plt.title('The number of components needed to explain variance ({})' .format(kind), fontsize=20)
    plt.axhline(y=0.99, color='r', linestyle='-')
    plt.text(0.5, 1.02, '99% cut-off threshold', color = 'red', fontsize=16)
    ax.grid(axis='both')
    plt.savefig(save_path + device + "_entries_list_" + date + "_PCA_{}" .format(kind))
    plt.show()
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
    plt.rcParams["figure.figsize"] = (5,5)
    plt.plot(xi, np.abs(np.gradient(y/(xi/n_traces),xi)))
    plt.tick_params(axis='both', which='major', labelsize=15)
    plt.grid(axis='both')
    plt.xlabel('Number of Components', fontsize=15)
    plt.axvline(x=n_components, color='r', linestyle='-')
    plt.title('Curve meaning  | \u2207 explained variance/(n_element/n_traces) |,\n for {} measuring {}' .format(device, kind), fontsize=15)
    plt.text(n_components, 0.0, n_components, color = 'red', fontsize=15)
    plt.savefig(save_path + device + "_entries_list_" + date + "_PCA_gradient_curve_{}" .format(kind))
    plt.show()
                
    max_n_components_pca = sharp_slope_indices[0]+1
    reshaped_robust_samples = np.array(robust_samples).reshape(np.shape(robust_samples)[0]*np.shape(robust_samples)[1],np.shape(robust_samples)[2])
   
    # REMOVE NULL TRACES
    # Count the number of zeros per row
    ceros_por_fila = np.count_nonzero(reshaped_robust_samples == 0, axis=1)

    # Find the index where the number of zeros per row equals 25000
    indice_25000_ceros = np.where(ceros_por_fila == n_samples)[0]
    
    reshaped_robust_samples = np.delete(reshaped_robust_samples, indice_25000_ceros, axis=0)
    
    # PCA APPLICATION TO BUGS
    print("Using {} components in PCA" .format(max_n_components_pca))
    pca = PCA(n_components=max_n_components_pca, whiten=False).fit(reshaped_robust_samples)
    pca_samples = pca.transform(reshaped_robust_samples)
    
    return pca_samples
     
def clustering_procedure(pca_samples, kind):
    global entries_path
    global entries_list
    global n_tests
    global cal_n_traces
    global n_traces
    global y_pred
    global gauss_dict
    global mapeo_strings
    global mapeo_bugs
    global val
    global save_path
    global device
    global date
       
    print("Performing clustering...")
    silh_score = [np.nan for x in range(n_traces)]
    calinski_score = np.array([np.nan for x in range(n_traces)])
    davies_score = [np.nan for x in range(n_traces)]
    sdbw_score = [np.nan for x in range(n_traces)]
    labels_array = [[np.nan for y in range(2*cal_n_traces+n_traces)] for x in range(n_traces)]
    
    for index in range(1,n_traces+1):
    
        data = pca_samples[:(2*cal_n_traces+index)]
    
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
        db = HDBSCAN(cluster_selection_epsilon=eps, min_samples=1)        
        # db = MeanShift()        
        # db = OPTICS(min_samples=2, metric='euclidean')
        
        y_pred = db.fit_predict(data)
        labels_array[index-1] = db.labels_
       
        num = str(index-1)
        if index < 10:
            num = '0' + num
    
        x = np.linspace(2*cal_n_traces+1, 2*cal_n_traces+index, index)
            

    unique, counts = numpy.unique(y_pred, return_counts=True)
    print("DBSCAN CLUSTERS:", dict(zip(unique, counts)))
            
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
    plt.title('Confusion Matrix (EM), {}, Time Domain in {}'.format(get_class_name(db), device), fontsize=25)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=22)
    plt.xticks(np.arange(len(etiquetas_unicas_pred)), etiquetas_unicas_pred, rotation=90)
    plt.yticks(np.arange(len(etiquetas_unicas_true)), etiquetas_unicas_true)
    plt.tick_params(axis='both', which='major', labelsize=22)
    plt.xlabel('Assigned Labels', fontsize=22)
    plt.ylabel('Actual Labels', rotation=90, verticalalignment='center', fontsize=22)
    plt.savefig(save_path + device + "_entries_list_" + date + "_CONFUSION_MATRIX_TIMEDOMAIN_" + str(get_class_name(db)) + "_PWR.png", bbox_inches='tight', dpi=100)
    plt.show()
          
    # Define the categorical color palette
    categorical_colors = sns.color_palette("tab20")
    
    # Create the scatterplot
    fig = plt.figure(figsize=(16, 16), dpi=100)
    ax = fig.add_subplot(projection='3d')
    
    # Set the same transparency for all points
    alpha_value = 0.8
    
    # Create custom legends grouped by colors
    sorted_labels = sorted(set(y_pred_changed[2*cal_n_traces+1:]))
    legend_elements = []
    
    # Filter labels containing only letters and those that are numeric
    string_labels = [label for label in sorted_labels if any(char.isalpha() for char in label)]
    
    # Assign distinct colors to each non-numeric label
    for i in range(len(string_labels)):
        label = string_labels[i]
        color = categorical_colors[i % len(categorical_colors)]
        legend_elements.append(Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label, alpha=alpha_value))
    
    # Add the legends to the plot only if there are elements in the list
    if legend_elements:
        ax.legend(handles=legend_elements, title='Clusters', fontsize=18)
    
    # Create the scatterplot for each dataset
    for i in range(2*cal_n_traces+1, len(y_pred)):
        label = y_pred_changed[i]
        x_val = data[i, int(np.round(np.shape(pca_samples)[1]/2)-1)]
        y_val = data[i, int(np.round(np.shape(pca_samples)[1]/2))]
        z_val = data[i, int(np.round(np.shape(pca_samples)[1]/2)+1)]
    
        # Get the index of the corresponding categorical color
        if label in string_labels:
            color_index = string_labels.index(label)
            color = categorical_colors[color_index]
        else:
            color = 'gray'  # Gray color for numeric values not found in string_labels
    
        # Create the scatterplot with the color argument instead of c
        ax.scatter(x_val, y_val, z_val, zdir='y', color=color, s=25, marker="o", alpha=alpha_value)
    
    # Adjust legend properties
    ax.get_legend().get_title().set_fontsize(20)
    
    # Configure axis labels and title
    plt.tick_params(axis='both', which='major', labelsize=20)
    ax.set_xlabel('$X$', fontsize=20)
    ax.set_ylabel('$Y$', fontsize=20)
    ax.set_zlabel('$Z$', fontsize=20, rotation=0)
    plt.title("Clustering considering {} elements ({}), {}, Time Domain in {}".format(n_traces, kind, str(get_class_name(db)), device), fontsize=25)
    
    # Save the figure and display
    plt.savefig(save_path + device + "_entries_list_" + date + "_" + str(get_class_name(db)) + "_" + kind + ".png", bbox_inches='tight', dpi=100)
    plt.show()

    return etiquetas_unicas_true, etiquetas_unicas_pred

def processing(input_signal, kind):
    globals()[f"{kind}_outlied_signal"] = outlier_detection(input_signal, kind)
    globals()[f"{kind}_pca_samples"] = pca_technique_application(globals()[f"{kind}_outlied_signal"], kind)
    globals()[f"{kind}_etiquetas_unicas_true"] , globals()[f"{kind}_etiquetas_unicas_pred"] = clustering_procedure(globals()[f"{kind}_pca_samples"], kind)
          
    
def operation():
    
    global EM_calibration_path
    global entries_nfs_path
    global silh
    global pca_plot
    global imp_model
    global ip_address
    global port_list
    global channel
    global n_traces
    global n_samples
    global command_OP
    global date
    global command
    global cal_traces
    global device
    global EM_input_signal
    global PWR_input_signal
     
    
    # EM SIGNALS COLLECTION
    EM_exec_name = [filename for filename in os.listdir(EM_calibration_path) if filename.startswith(device + "_EM_exec_")][0]
    EM_noexec_name = [filename for filename in os.listdir(EM_calibration_path) if filename.startswith(device + "_EM_noexec")][0]
    EM_fuzz_name = device + "_EM_bugs_" + date + ".csv"
    print("Entries signals files: ", EM_fuzz_name)
      
    EM_signal = np.loadtxt(EM_calibration_path + "/" + EM_exec_name,delimiter=',')
    EM_cal_traces = np.shape(EM_signal)[0]
    n_samples = np.shape(EM_signal)[1]
    
    EM_input_signal = np.array([[[0 for z in range(n_samples)] for y in range(n_traces)] for x in range(3)])
    EM_input_signal[0][:EM_cal_traces] = np.abs(np.loadtxt(EM_calibration_path + EM_exec_name,delimiter=','))
    EM_input_signal[1][:EM_cal_traces] = np.abs(np.loadtxt(EM_calibration_path + EM_noexec_name, delimiter=','))
    EM_input_signal[2] = np.abs(np.loadtxt(EM_entries_nfs_path + EM_fuzz_name, delimiter=','))

    processing(EM_input_signal, "EM")
    
    return

def bugs_capture(ssh):
    
    global device
    global date  
    global n_traces
    global n_tests
    global entries
    global entries_list
        
    entries_command = "python3.11 /nfs/general/code/EM_PWR_10x_data_collection.py "
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(entries_command + " " + str(device + "_entries_list_" + date + '.csv'))
    print(ssh_stdout.read().decode())
    print(time.time()-start, "seconds to get faults samples")
    
def calibration(ssh):
    global calibration_command
    global cal_n_traces
    global device
    
    # CALIBRATION SIGNALS CAPTURE
    cal_command = "python3.11 /nfs/general/code/EM_PWR_10x_data_calibration.py "
    command = "a1"
    calibration_command = command
    ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(cal_command + command + " " + str(cal_n_traces) + " " + device)
    print(ssh_stdout.read().decode())
    print(time.time()-start, "seconds to get calibration samples")
    
def entries_list_creation(entries):
    global entries_list
    global list_nfs_path
    global date
    
    input_info = entries.split(',')
    # Array to store 20 repetitions of each string
    entries_list = []

    # Repeat each string 20 times and add it to the new array
    for string in input_info:
        repeated_string = [string] * 20
        entries_list.extend(repeated_string)
            
    now = datetime.datetime.now()
    date_arr = [str(now.day), str(now.month), str(now.year), str(now.hour), str(now.minute)]
    date_arr = [x.zfill(2) for x in date_arr]
    date = date_arr[2] + "_" + date_arr[1] + "_" + date_arr[0] + "_" + date_arr[3] + "o" + date_arr[4]

    with open(list_nfs_path + "_entries_list_" + date + '.csv', 'w') as txt_file:
        for line in entries_list:
            txt_file.write(line + "\n")
            
    print("Entries list created: ", device, '_entries_list_', date, '.csv')
       
start1 = time.time()

# OPENING SSH CONNECTION TO STM32-F429ZI
# ssh = paramiko.client.SSHClient()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
# ssh.connect(stm_ip, username="USER_X", password="PASSWORD_X")  # Anonymized credentials

# print(time.time()-start1, "seconds to start ssh connection")

start = time.time()

# FUZZING ENTRIES SIGNALS CAPTURE
for it in range(n_fuzz):
    it_start = time.time()
    print("ITERATION", it+1)
    
    entries = "a0,a1,00,02,03,04,05,06,07,08,09,0a,0b,0c,0d,0e,0f,a3"
    # entries_list_creation(entries)
    cal_n_traces = 100
    # calibration(ssh)
    # bugs_capture(ssh) # BUGS CAPTURING
    
    with open(list_nfs_path + "_entries_list_" + date + ".csv") as file:
        entries_list = [line.rstrip() for line in file]
    number_entries_list = [mapeo_strings[valor] for valor in entries_list]
    
    n_traces = np.shape(entries_list)[0]
    n_tests = int(np.sqrt(n_traces))
    
    operation() # DATA PROCESSING + CLUSTERING
    print(time.time()-it_start, "seconds to perform fuzzing iteration", it+1)
     
# ssh.close()            
    
print(time.time()-start, "seconds to perform operation")
print(time.time()-start1, "seconds to perform whole process")
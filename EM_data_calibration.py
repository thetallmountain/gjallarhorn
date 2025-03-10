#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:07:16 2024

@author: anonymous
"""

import datetime
import socket
import threading
import time
import numpy as np
from scipy.stats import pearsonr
import datetime
import subprocess
import pyvisa as visa
from tektronik_mso56 import *
import matplotlib.pyplot as plt

# Command line inputs
# command = [bytes.fromhex(sys.argv[1])]
# whole_n_traces = int(sys.argv[2])
# device = sys.argv[3]
command = 0xa1
comando = str(command)[1:3]
n_traces = int(100)
whole_n_traces = n_traces
device = "STM32-F429ZI"

now = datetime.datetime.now()
date_arr = [str(now.day), str(now.month), str(now.year), str(now.hour), str(now.minute)]
date_arr = [x.zfill(2) for x in date_arr]
date = date_arr[0] + "_" + date_arr[1] + "_" + date_arr[2] + "_" + date_arr[3] + "o" + date_arr[4]

# REMOTE EQUIPMENT CONFIGURATION

# IP address and port of remote equipment
remote_ip = "X.X.X.X"
remote_port = 6000
# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)


# OSCILLOSCOPE CONFIGURATION

# IP address and monitored channels of the oscilloscope
osc_ip = "Y.Y.Y.Y"
channel = ['CH2']

## Init Scope

## Set/Get Scope parameters
voltDiv = 1e-3  # V/div
    
# Open connection to oscilloscope by IP address
scope = Tektronik_MSO56()
scope.connect(IPAddress=osc_ip)

timeDiv = 15e-6  # s/div
voltDiv = 200e-3  # V/div

sampleRate = 1.25e9  # S/div
n_samples = int(10*timeDiv*sampleRate)  # time when trigger goes off * sampling frequency (maximum is 10*timeDiv*sampleRate?)

EM_calibration_path = '/home/anonymous/Project_Calibration_Clustering/input_files/calibration/'

comando_rst = 'python3.11 /nfs/general/device_rst_relee.py'

# Data in hexadecimal
# commands_hex = [0x00, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0xa1, 0xa0]

def NOP_execution(channel, n_traces):
    
    global device
    global scope
    global date
    global delta_t0
    global cal_path
    global whole_n_traces
    global n_samples 
 
    delta_t0 = 0.9455416202545166
    EM_NOP_signal = np.array([[0 for z in range(n_samples)] for y in range(whole_n_traces)])
    
    # Main loop
    numTotal = 0
        
    try:
        while numTotal < n_traces:
            print("Idle signal capture, trace", numTotal+1)
            start_time = time.time()  # START TIMER
            while True:
                 if (time.time()-start_time)>=(delta_t0):
                     break  # END TIMER
                     
            EM_NOP_signal[numTotal] = scope.getWaveform(channel=channel[0])
            
            print("(EM) Data output from Ethernet (oscilloscope):", len(EM_NOP_signal[numTotal])) 
            numTotal+=1

        np.savetxt(EM_calibration_path + device + '_EM_noexec_' + date + '.csv', np.c_[EM_NOP_signal], delimiter=',')
        
    except Exception as ex:
        print("ERROR: ", ex)
        scope.clear()

def OP_execution(channel, n_traces, command):

    global sock
    global device    
    global date
    global remote_ip
    global remote_port
    global delta_t0
    global n_sample
    global EM_entries_nfs_path
    global scope
    
    
    EM_OP_signal = np.array([[0 for z in range(n_samples)] for y in range(whole_n_traces)])
    
    scope.setTriggerMode("MANUAL")
            
    try:
        time.sleep(1)
        start_time = time.time()
        data = bytes([command])
        sock.sendto(data, (remote_ip, remote_port))
        # Receive data from client with specified timeout
        data_output, addr = sock.recvfrom(1024)  # Maximum size of data to receive
    
        # Display received data
        print("Data output from ETH ({}): {}" .format(device, data_output))
                    
        # Main loop
        numTotal = 0
        
        while numTotal < n_traces:
            print("Command {}, trace {}" .format(command,numTotal+1))
            scope.write("*CLS")

            start_time = time.time()
            data = bytes([command])
            sock.sendto(data, (remote_ip, remote_port))
            # Set timeout to 500 ms
            sock.settimeout(5)
            
            try:
                
                # Receive data from client with specified timeout
                data_output, addr = sock.recvfrom(1024)  # Maximum size of data to receive
            
                # Display received data
                print("Data output from serial ({}): {}" .format(device, data_output))
                
                epsilon0_time = time.time()
                epsilon0 = epsilon0_time - start_time
                             
                EM_OP_signal[numTotal] = scope.getWaveform(channel=channel[0])
                
                epsilon1_start = time.time()
                            
            except socket.timeout:
                print("Timeout expired waiting for data.")
                data_output = b''

            
            if data_output == b'':
                print("Resetting {}..." .format(device))
                reset = subprocess.run(comando_rst, shell=True, capture_output=True, text=True)
                time.sleep(1)
                
            print("(EM) Data output from Ethernet (oscilloscope):", len(EM_OP_signal[numTotal]))
            
            delta_t0_time = time.time()
            delta_t0 = delta_t0_time - start_time
            epsilon1 = delta_t0_time-epsilon1_start
            
            if numTotal>0:
                print("(EM) Pearson: ", pearsonr(EM_OP_signal[numTotal], EM_OP_signal[numTotal-1])[0]*100)
            numTotal+=1
            
        np.savetxt(EM_calibration_path + device + '_EM_exec_' + comando + "_" + date + '.csv', np.c_[EM_OP_signal], delimiter=',')
                                
    except Exception as ex:
        print("ERROR: ", ex)


OP_execution(channel, n_traces, command)
NOP_execution(channel, n_traces)
sock.close()
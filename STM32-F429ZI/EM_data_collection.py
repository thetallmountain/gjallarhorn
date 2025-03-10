#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 14:36:14 2024

@author: anonymous
"""

import socket
import threading
import time
import numpy as np
from scipy.stats import pearsonr
import datetime
import subprocess
from tektronik_mso56 import *
import sys
import subprocess
import paramiko

# Command line inputs
n_traces = int(sys.argv[1])
list_name = sys.argv[1]
file_date = list_name[-20:-4]
device = list_name[0:-34]


# OSCILLOSCOPE CONFIGURATION

# IP address and monitored channels of the oscilloscope
osc_ip = "X.X.X.X"
channel = ['C3','C4']

## If loadLecroyPanelEnable is True, loads a pre-registered config file to avoid setting scope with command lines
## Tips: you can do manual setting of scope and afterwards save panel config with lecroy3.py, with command: >>python lecroy3.py -s <panelname>.lss
loadLecroyPanelEnabled = False
lecroyPanelFileName = "config/xoodyakHash.lss"

## Init Scope
scope = Lecroy()
scope.connect(IPAdrress = osc_ip)
# scope.displayOff() #Disable the plotting of traces during data collection 

## Set/Get Scope parameters
if(loadLecroyPanelEnabled):
    scope.loadLecroyPanelFromFile(lecroyPanelFileName)
    voltDiv = float(scope.getVoltsDiv(channel))
    timeDiv = float(scope.getTimeDiv())
else:
    scope.setTriggerDelay("0");
    voltDiv = 600e-3#V/div
    timeDiv = 5e-6#s/div
    sampleRate = 1e9#S/s
    scope.setSampleRate(sampleRate)
    scope.setVoltsDiv(channel[0], str(voltDiv))
    voltDiv = 100e-3#V/div
    scope.setVoltsDiv(channel[1], str(voltDiv))
    scope.setTimeDiv(str(timeDiv))

# OSCILLOSCOPE CONFIGURATION

# IP address and monitored channels of the oscilloscope
osc_ip = "X.X.X.X"
channel = ['CH2']

# REMOTE DEVICE CONFIGURATION 

# IP address and port of remote device
remote_ip = "Y.Y.Y.Y"
remote_port = 6000
# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

## Init Scope
## Set/Get Scope parameters
voltDiv = 1e-3  # V/div
    
# Open connection to oscilloscope by IP address
scope = Tektronik_MSO56()
scope.connect(IPAddress=osc_ip)

timeDiv = 15e-6  # s/div
voltDiv = 200e-3  # V/div

sampleRate = 1.25e9  # S/div
scope.setSampleRate(sampleRate)
n_samples = int(10*timeDiv*sampleRate)  # time when trigger goes off * sampling frequency (maximum is 10*timeDiv*sampleRate?)

EM_calibration_path = '/home/anonymous/Project_Calibration_Clustering/input_files/calibration/'

comando_rst = 'python /nfs/general/device_rst_relee.py'  
rst_hex = 0xa5

list_path = '/home/anonymous/Project_Calibration_Clustering/input_files/entries_lists/'
EM_entries_nfs_path = '/home/anonymous/Project_Calibration_Clustering/input_files/entries_results/' 


# RESET CONFIGURATION VIA RASPBERRY
raspi_ip = "Z.Z.Z.Z"
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(raspi_ip, port=22, username="user", password="password")
comando_rst = 'python /home/user/device_rst_relee.py'
 
# List to store hexadecimal numbers as integers
entries_list = []

# Open file and read line by line
with open(list_path + list_name, "r") as file:
    for line in file:
        # Remove line breaks and whitespace around the number
        hex_number = line.strip()

        # Convert hexadecimal number to integer
        decimal_number = bytes.fromhex(hex_number)

        # Add the integer to the list
        entries_list.append(decimal_number)

print("Capturing signals from commands in: ", list_name)

n_traces = np.shape(entries_list)[0]

def OP_execution(channel, n_traces, entries_list):

    global device
    global file_date
    global EM_entries_nfs_path
    global delta_t0
    global n_samples
    global sampleRate
    global comando_rst
    
    EM_OP_signal = np.array([[0 for z in range(n_samples)] for y in range(n_traces)])
  
    ## Set Scope trigger
    print(entries_list)
    start_time = time.time()
    command=entries_list[0]
    data = command
    sock.sendto(data, (remote_ip, remote_port))
    # Receive data from client with specified timeout
    data_output, addr = sock.recvfrom(1024)  # Maximum size of data to receive
    print("Data output from serial ({}): {}" .format(device, data_output))
    
    i = 0
    
    while i < len(entries_list):
        
        try:
            
            command=entries_list[i]  
            
            # REPEAT VERIFICATION
            print("Command {} ({})" .format(command.hex(), i+1))
                
            start_time = time.time()
            
            data = command
            sock.sendto(data, (remote_ip, remote_port))
            # Set timeout to 5 seconds
            sock.settimeout(5)
            
            try: 
                # Receive data from client with specified timeout
                data_output, addr = sock.recvfrom(1024)  # Maximum size of data to receive
            
                # Display received data
                print("Data output from serial ({}): {}" .format(device, data_output))
                
                epsilon0_time = time.time()
                epsilon0 = epsilon0_time - start_time
                ## Get data from Scope
                EM_OP_signal[i] = scope.getWaveform(channel=channel[0])
                
                epsilon1_start = time.time()
                            
            except socket.timeout:
                print("Timeout is over. Resetting {}" .format(device))
                
                ## Get data from Scope
                EM_OP_signal[i] = scope.getWaveform(channel=channel[0])
                
                stdin, stdout, stderr = client.exec_command(comando_rst)
                
                time.sleep(10)
                
            print("(EM) Data output from Ethernet (oscilloscope):", len(EM_OP_signal[i]))
            
            delta_t0_time = time.time()
            delta_t0 = delta_t0_time - start_time
            epsilon1 = delta_t0_time-epsilon1_start
            
            if i>0:
                print("(EM) Pearson: ", pearsonr(EM_OP_signal[i], EM_OP_signal[i-1])[0]*100)
            i+=1
            
        except Exception as ex:
            print("ERROR: ", ex)
            if data_output == b'\x00':
                i+=1
            else:
                continue
            
    np.savetxt(EM_entries_nfs_path + device + '_EM_bugs_' + file_date + '.csv', np.c_[EM_OP_signal], delimiter=',')
    client.close()

OP_execution(channel, n_traces, entries_list)
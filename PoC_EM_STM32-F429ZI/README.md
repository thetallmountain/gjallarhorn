# Electromagnetic Side-Channel Analysis PoC for STM32-F429ZI

This Proof of Concept demonstrates electromagnetic (EM) side-channel analysis techniques for the STM32-F429ZI microcontroller platform. The code processes previously captured EM emissions data to identify vulnerabilities through signal processing and machine learning techniques.

## Overview

This PoC demonstrates:
- Processing of electromagnetic emission traces
- Signal cleaning through outlier detection and robust covariance
- Dimensionality reduction using PCA (Principal Component Analysis)
- Cluster analysis using HDBSCAN (Hierarchical Density-Based Spatial Clustering)
- Visualization of results through confusion matrices and 3D scatter plots

## Requirements

- Python 3.11+
- Required packages:
  - scikit-learn>=1.6.1
  - numpy
  - matplotlib
  - pandas
  - scipy
  - seaborn
  - pyvisa
  - paramiko
  - scp

## Usage

1. Ensure the required dataset is available in the `results` directory:
   - `STM32-F429ZI_EM_bugs_2024_04_17_15o16.zip` 

2. Run the main script:
python3.11 PoC_EM_STM32-F429ZI.py

3. The script will:
- Unzip pre-captured EM data from the device
- Process the signal data using outlier detection
- Apply robust covariance estimation
- Perform PCA for dimensionality reduction
- Apply HDBSCAN clustering to identify patterns
- Generate visualization plots showing the results

## Data Processing Pipeline

1. **Data Preparation**: Unzips pre-captured EM traces and organizes file paths
2. **Outlier Detection**: Removes anomalous data points using quartile-based thresholding
3. **Robust Covariance**: Applies EllipticEnvelope for robust covariance estimation
4. **Dimensionality Reduction**: Uses PCA to reduce the data dimensions while preserving variance
5. **Clustering**: Applies HDBSCAN to identify patterns in the EM traces
6. **Visualization**: Generates confusion matrices and 3D scatter plots to visualize the results

## Results

Analysis results are saved in the `results/STM32-F429ZI_EM_bugs_2024_04_17_15o16/output_figs/` directory, including:
- PCA variance plots
- Confusion matrices showing the accuracy of vulnerability detection
- 3D scatter plots visualizing the clustering results

## Acknowledgments

This PoC is part of the Gjallarhorn framework for electromagnetic side-channel analysis.

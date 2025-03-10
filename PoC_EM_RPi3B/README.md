# Electromagnetic Side-Channel Analysis PoC for Raspberry Pi 3B

This Proof of Concept demonstrates electromagnetic (EM) side-channel analysis techniques for the Raspberry Pi 3B platform. The code processes previously captured EM emissions data to identify vulnerabilities through signal processing and machine learning techniques.

## Overview

This PoC demonstrates:
- Processing of electromagnetic emission traces
- Signal cleaning through outlier detection
- Dimensionality reduction using PCA (Principal Component Analysis)
- Cluster analysis using Affinity Propagation
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
   - `RPi3B_EM_bugs_2024_01_23_14o38.zip` 

2. Run the main script:
python3.11 PoC_EM_RPi3B.py
Copy
3. The script will:
- Unzip pre-captured EM data from the device
- Process the signal data using outlier detection
- Apply PCA for dimensionality reduction
- Perform clustering analysis using Affinity Propagation
- Generate visualization plots showing the results

## Data Processing Pipeline

1. **Data Preparation**: Unzips pre-captured EM traces and organizes file paths
2. **Outlier Detection**: Removes anomalous data points using quartile-based thresholding
3. **Signal Processing**: Applies linear interpolation to clean the signals
4. **Dimensionality Reduction**: Uses PCA to reduce the data dimensions while preserving variance
5. **Clustering**: Applies Affinity Propagation to identify patterns in the EM traces
6. **Visualization**: Generates confusion matrices and 3D scatter plots to visualize the results

## Results

Analysis results are saved in the `results/RPi3B_EM_bugs_2024_01_23_14o38/output_figs/` directory, including:
- PCA variance plots
- Confusion matrices showing the accuracy of vulnerability detection
- 3D scatter plots visualizing the clustering results

## Acknowledgments

This PoC is part of the Gjallarhorn framework for electromagnetic side-channel analysis.

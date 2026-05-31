# 5g-modulation-classifier
OFDM Modulation Classifier (QPSK/16QAM/64QAM) using Random Forest | 92% accuracy at 20dB SNR
# OFDM Modulation Classifier

## What is this project?
A machine learning classifier that identifies the modulation 
scheme (QPSK, 16QAM, 64QAM) of a received OFDM signal — 
the core waveform used in 4G and 5G networks.

## Why does it matter?
5G base stations continuously switch between modulation schemes 
based on channel quality. Identifying the correct scheme from 
a noisy received signal is critical for reliable communication.
This project simulates that process using ML.

## How it works
1. Generate OFDM signals for 3 modulation types
2. Simulate realistic wireless channel (AWGN noise at 
   multiple SNR levels: 0 to 25dB)
3. Extract 10 statistical features from received signal
   (amplitude variance, kurtosis, signal power etc.)
4. Train Random Forest classifier on extracted features

## Results
- 92.4% accuracy at SNR = 20dB
- 94.8% accuracy at SNR = 25dB
- 16QAM and 64QAM show expected confusion at low SNR
  due to denser constellation points
- Amplitude variance and std are most important features

## Tools used
Python, NumPy, Scipy, Scikit-learn, Matplotlib

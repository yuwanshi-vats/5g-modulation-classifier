# 5G OFDM Modulation Classifier

## What is this project?
A machine learning classifier that identifies the modulation 
scheme (QPSK, 16QAM, 64QAM) of a received OFDM signal — 
the core waveform used in 4G and 5G networks.

## Why does it matter?
5G base stations continuously switch between modulation schemes 
based on channel quality. Identifying the correct scheme from 
a noisy received signal is critical for reliable communication.

## How it works
1. Generate OFDM signals for 3 modulation types
2. Simulate realistic wireless channel (AWGN noise, SNR 0-25dB)
3. Extract 10 statistical features from received signal
4. Train Random Forest classifier on extracted features

## Results
- 90.4% accuracy at SNR = 20dB
- 91.8% accuracy at SNR = 25dB
- Amplitude variance is most discriminating feature

## Tools
Python | NumPy | Scipy | Scikit-learn | Matplotlib

## Author
Yuwanshi Vats | B.Tech ECE | JIIT Noida

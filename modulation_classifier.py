import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
N = 64        # subcarriers
CP = 16       # cyclic prefix
NUM = 1000    # samples per modulation type
SNRS = [0, 5, 10, 15, 20, 25]  # multiple SNR levels
def generate_qpsk(n):
    points = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    return points[np.random.randint(0, 4, n)]

def generate_16qam(n):
    vals = [-3, -1, 1, 3]
    points = np.array([x+1j*y for x in vals for y in vals]) / np.sqrt(10)
    return points[np.random.randint(0, 16, n)]

def generate_64qam(n):
    vals = [-7, -5, -3, -1, 1, 3, 5, 7]
    points = np.array([x+1j*y for x in vals for y in vals]) / np.sqrt(42)
    return points[np.random.randint(0, 64, n)]

print("Constellation functions ready")
def generate_ofdm(symbols, snr_db):
    time_signal = np.fft.ifft(symbols.reshape(1, -1), axis=1)
    cp = time_signal[:, -CP:]
    ofdm_tx = np.concatenate([cp, time_signal], axis=1)
    power = np.mean(np.abs(ofdm_tx)**2)
    noise_power = power / (10**(snr_db/10))
    noise = np.sqrt(noise_power/2) * (
        np.random.randn(*ofdm_tx.shape) +
        1j*np.random.randn(*ofdm_tx.shape))
    ofdm_rx = ofdm_tx + noise
    return np.fft.fft(ofdm_rx[:, CP:], axis=1).flatten()

print("OFDM function ready")
def extract_features(signal):
    amplitude = np.abs(signal)
    phase = np.angle(signal)
    real = signal.real
    imag = signal.imag

    f1 = np.mean(amplitude)
    f2 = np.var(amplitude)
    f3 = np.var(phase)
    f4 = kurtosis(amplitude)
    f5 = np.mean(amplitude**2)
    f6 = kurtosis(phase)
    f7 = np.mean(real**2)          # real power
    f8 = np.mean(imag**2)          # imaginary power
    f9 = np.max(amplitude)         # peak amplitude
    f10 = np.std(amplitude)        # amplitude std

    return [f1, f2, f3, f4, f5, f6, f7, f8, f9, f10]

print("Updated feature extractor ready")
x=[]
y=[]
modulations=[generate_qpsk,generate_16qam,generate_64qam]
for label,mod_fun in enumerate(modulations):
    for i in range(NUM):
        snr = np.random.choice(SNRS)
        symbols=mod_fun(N)
        received = generate_ofdm(symbols,snr)
        features = extract_features(received)
        x.append(features)
        y.append(label)

x=np.array(x)
y=np.array(y)

print("xshape",x.shape)
print("yshape",y.shape)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify =y)
scaler=StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

clf= RandomForestClassifier(n_estimators=100,random_state=42)
clf.fit(x_train_scaled,y_train)

y_pred = clf.predict(x_test_scaled)
accuracy= accuracy_score(y_test,y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")
print("\nClassification Report:")
print(classification_report(y_test, y_pred,
      target_names=['QPSK', '16QAM', '64QAM']))

snr_levels = [0, 5, 10, 15, 20, 25]
accuracies = []

for snr in snr_levels:
    X_snr = []
    y_snr = []

    for label, mod_func in enumerate(modulations):
        for _ in range(300):
            symbols = mod_func(N)
            received = generate_ofdm(symbols, snr)
            features = extract_features(received)
            X_snr.append(features)
            y_snr.append(label)

    X_snr = np.array(X_snr)
    y_snr = np.array(y_snr)

    X_scaled_snr = scaler.transform(X_snr)
    y_pred_snr = clf.predict(X_scaled_snr)
    acc = accuracy_score(y_snr, y_pred_snr)
    accuracies.append(acc * 100)
    print(f"SNR = {snr:2d} dB → Accuracy = {acc*100:.1f}%")

# Plot
plt.figure(figsize=(8, 5))
plt.plot(snr_levels, accuracies, 'o-', color='blue', linewidth=2.5, markersize=8)
plt.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='90% target')
plt.xlabel("SNR (dB)")
plt.ylabel("Accuracy (%)")
plt.title("Modulation Classifier — Accuracy vs SNR")
plt.legend()
plt.grid(True)
plt.show()
# Get predictions on test set
y_pred_final = clf.predict(x_test_scaled)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_final)

plt.figure(figsize=(6, 5))
im = plt.imshow(cm, cmap='Blues')
plt.colorbar(im)

classes = ['QPSK', '16QAM', '64QAM']
plt.xticks(range(3), classes)
plt.yticks(range(3), classes)
plt.xlabel('Predicted', fontweight='bold')
plt.ylabel('True', fontweight='bold')
plt.title('Confusion Matrix — Modulation Classifier', fontweight='bold')

for i in range(3):
    for j in range(3):
        plt.text(j, i, str(cm[i,j]),
                ha='center', va='center',
                color='white' if cm[i,j] > cm.max()/2 else 'black',
                fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
feature_names = ['Mean Amp', 'Amp Variance', 'Phase Variance', 
                 'Amp Kurtosis', 'Signal Power', 'Phase Kurtosis',
                 'Real Power', 'Imag Power', 'Peak Amp', 'Amp Std']

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(10), importances[indices], color='steelblue')
plt.xticks(range(10), [feature_names[i] for i in indices], rotation=20, ha='right')
plt.title('Feature Importance — What separates QPSK vs 16QAM vs 64QAM?')
plt.ylabel('Importance')
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

import numpy as np
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
import joblib
from tensorflow.keras.models import load_model
import os

# --- Ortak Data ---
processed_data_path = 'data/processed/'
test_images  = np.load(os.path.join(processed_data_path, 'processed_test_images.npy'))
test_filenames = np.load(os.path.join(processed_data_path, 'test_filenames.npy'))
true_labels = np.array([1 if "HL" in fn else -1 for fn in test_filenames])
X_test = test_images.reshape(len(test_images), -1)

print("Test veri:", test_images.shape)

# --- 1. Autoencoder ---
print("\n==== 1. Autoencoder (CAE) ====\n")
autoencoder = load_model('models/cae_best.keras', compile=False)
X_test_pred = autoencoder.predict(test_images)
test_errors = np.mean((test_images - X_test_pred) ** 2, axis=(1,2,3))
thresholds = np.linspace(test_errors.min(), test_errors.max(), 50)
f1s, precs, recs = [], [], []
for thresh in thresholds:
    pred = np.where(test_errors > thresh, -1, 1)
    f1s.append(f1_score(true_labels, pred, pos_label=-1))
    precs.append(precision_score(true_labels, pred, pos_label=-1, zero_division=0))
    recs.append(recall_score(true_labels, pred, pos_label=-1))
best_idx = np.argmax(f1s)
best_thresh = thresholds[best_idx]
print(f"Best threshold: {best_thresh:.4f}  -->  F1: {f1s[best_idx]:.3f}")
best_pred_labels = np.where(test_errors > best_thresh, -1, 1)
print(classification_report(true_labels, best_pred_labels, zero_division=0))

# --- 2. OCSVM + HOG ---
print("\n==== 2. OCSVM + HOG ====\n")
from skimage.feature import hog
ocsvm = joblib.load('models/ocsvm_model.pkl')
ocsvm_scaler = joblib.load('models/ocsvm_scaler.pkl')
ocsvm_pca = joblib.load('models/ocsvm_pca.pkl')
hog_features_test = [hog(img.squeeze(), pixels_per_cell=(8,8), cells_per_block=(2,2), visualize = False, feature_vector=True)
                     for img in test_images]
X_test_hog = np.array(hog_features_test)
X_test_hog_scaled = ocsvm_scaler.transform(X_test_hog)
X_test_hog_pca = ocsvm_pca.transform(X_test_hog_scaled)
pred = ocsvm.predict(X_test_hog_pca)
print(classification_report(true_labels, pred, zero_division=0))

# --- 3. Isolation Forest + HOG ---
print("\n==== 3. Isolation Forest + HOG ====\n")
iforest = joblib.load('models/isoforest.pkl')
iforest_scaler = joblib.load('models/isoforest_scaler.pkl')
X_test_hog_scaled = iforest_scaler.transform(X_test_hog)
pred = iforest.predict(X_test_hog_scaled)
print(classification_report(true_labels, pred, zero_division=0))

# --- İstersen F1 karşılaştırma tablosu da yazdırabilirsin ---
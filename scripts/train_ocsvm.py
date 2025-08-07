import os
import numpy as np
import joblib
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import hog

# --- PATHLER ---
processed_data_path = 'data/processed/'
models_path = 'models/'
os.makedirs(models_path, exist_ok=True)

# --- 1. Veri Yükle ---
train_images = np.load(os.path.join(processed_data_path, 'processed_train_images.npy'))

# --- 2. HOG Özelliklerini Hesapla ---
hog_features = [
    hog(img.squeeze(), pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
    for img in train_images
]
X_hog = np.array(hog_features)

# --- 3. Standartlaştırma ---
scaler = StandardScaler()
X_hog_scaled = scaler.fit_transform(X_hog)

# --- 4. PCA ile Boyut İndir (isteğe bağlı, hız ve stabilite için önerilir) ---
pca = PCA(n_components=64, random_state=42)
X_hog_pca = pca.fit_transform(X_hog_scaled)

# --- 5. OCSVM Eğitimi ---
oc_svm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.45)
print("OCSVM (HOG feature ile) eğitiliyor...")
oc_svm.fit(X_hog_pca)
print("Eğitim tamam.")

# --- 6. Kaydet ---
joblib.dump(oc_svm,      os.path.join(models_path, 'ocsvm_model.pkl'))
joblib.dump(scaler,      os.path.join(models_path, 'ocsvm_scaler.pkl'))
joblib.dump(pca,         os.path.join(models_path, 'ocsvm_pca.pkl'))
print("Model, scaler ve PCA kaydedildi.")
import os
import numpy as np
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from skimage.feature import hog

# --- PATHLER ---
processed_data_path = 'data/processed/'
models_path = 'models/'
os.makedirs(models_path, exist_ok=True)

# --- 1. Eğitim verisini yükle ---
train_images = np.load(os.path.join(processed_data_path, 'processed_train_images.npy'))

# --- 2. HOG hesapla ---
print("HOG feature çıkarılıyor...")
hog_features = [
    hog(img.squeeze(), pixels_per_cell=(8,8), cells_per_block=(2,2), visualize=False, feature_vector=True)
    for img in train_images
]
X_hog = np.array(hog_features)

# --- 3. Scale ---
scaler = StandardScaler()
X_hog_scaled = scaler.fit_transform(X_hog)


# --- 5. Isolation Forest Eğitimi (contamination=0.40 örnek, optimize etmek için sweep test dosyada) ---
clf = IsolationForest(contamination=0.40, random_state=42)
print("Isolation Forest (HOG feature ile) eğitiliyor...")
clf.fit(X_hog_scaled)
print("Eğitim tamam.")

# --- 6. Kaydet ---
joblib.dump(clf,    os.path.join(models_path, 'isoforest.pkl'))
joblib.dump(scaler, os.path.join(models_path, 'isoforest_scaler.pkl'))
print("Model, scaler ve PCA kaydedildi.")
#Her model, model_training.py scripti kullanılarak eğitim veri seti üzerinde eğitilecek. 
#Eğitim sırasında modelin parametreleri optimize edilecek.
import os
import joblib # Python nesnelerini (modeller gibi) dosya olarak kaydetmek ve yüklemek için kullanılır.
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor

processed_data_path = 'data/processed/'
train_images = np.load(os.path.join(processed_data_path,'processed_train_images.npy'))

oc_svm = OneClassSVM(kernel='rbf', gamma='auto')
print("One-Class SVM eğitiliyor...")
oc_svm.fit(train_images.reshape(len(train_images), -1))
print("One-Class SVM eğitimi tamamlandı.")

iso_forest = IsolationForest(contamination=0.1, random_state=42)
print("Isolation Forest eğitiliyor...")
iso_forest.fit(train_images.reshape(len(train_images), -1))
print("Isolation Forest eğitimi tamamlandı.")

autoencoder = MLPRegressor(hidden_layer_sizes=(128, 64, 128), max_iter=100, random_state=42)
print("Autoencoder eğitiliyor...")
autoencoder.fit(train_images.reshape(len(train_images), -1), train_images.reshape(len(train_images), -1))
print("Autoencoder eğitimi tamamlandı.")

models_path = 'models/'
joblib.dump(oc_svm, os.path.join(models_path, 'oc_svm_model.pkl'))
joblib.dump(iso_forest, os.path.join(models_path, 'iso_forest_model.pkl'))
joblib.dump(autoencoder, os.path.join(models_path, 'autoencoder_model.pkl'))


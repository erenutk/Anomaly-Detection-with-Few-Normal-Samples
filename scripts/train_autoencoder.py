import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# --- 1. Eğitim verisi yükle ---
train_images = np.load('data/processed/processed_train_images.npy').astype(np.float32)

# --- 2. MODEL MİMARİSİ ---
input_shape = (64, 64, 1)
latent_dim = 64

inputs = Input(shape=input_shape)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Flatten()(x)
latent = layers.Dense(latent_dim, activation='relu')(x)
latent = layers.BatchNormalization()(latent)
latent = layers.Dropout(0.25)(latent)

x = layers.Dense(4*4*256, activation='relu')(latent)
x = layers.Reshape((4, 4, 256))(x)
x = layers.Conv2DTranspose(256, (3, 3), strides=2, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(128, (3, 3), strides=2, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = models.Model(inputs, decoded)
def ssim_loss(y_true, y_pred):
    return 1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

def hybrid_loss(y_true, y_pred):
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return 0.5 * mse + 0.5 * ssim_loss(y_true, y_pred)

autoencoder.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss=hybrid_loss)
autoencoder.summary()

# --- 3. CALLBACKS ve Eğitim ---
# Erken durdurma ile epoch'u kısaltıyoruz!
cb = [
    EarlyStopping(patience=7, restore_best_weights=True, monitor='val_loss'),
    ReduceLROnPlateau(patience=3, factor=0.5, verbose=1, min_lr=1e-5, monitor='val_loss')
]

history = autoencoder.fit(
    train_images, train_images,
    epochs=80,                # Bu zaten "en çok 80 epoch" anlamında. Early stopping varsa genelde 10-30 epoch civarıda biter!
    batch_size=32,
    shuffle=True,
    validation_split=0.1,
    callbacks=cb
)

# --- 4. SAVE ---
autoencoder.save('models/cae_best.keras')
print("CAE modeli kaydedildi: models/cae_best.keras")
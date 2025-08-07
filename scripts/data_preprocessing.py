import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A

#raw data setini yükle
def load_and_preprocess_images(folder, image_size=(64, 64)):
    images = []
    filenames = []
    for filename in tqdm(os.listdir(folder)):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Grayscale okuma!
        if img is not None:
            img = cv2.resize(img, image_size)
            img = img.astype(np.float32) / 255.0           # 0-1 arası normalizasyon
            images.append(img)
            filenames.append(filename)
    images = np.array(images)
    images = np.expand_dims(images, -1)  # Shape: (N, 64, 64, 1)
    return images, filenames


# Augmentations pipeline (dilediğin kadar ekleyebilirsin!)
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.2),
    A.Rotate(limit=20, p=0.5),
    A.RandomBrightnessContrast(p=0.4)
])
"""
 A.GaussNoise(p=0.2),
    A.RandomScale(scale_limit=0.15, p=0.3),
    A.ElasticTransform(p=0.15),
    A.RandomResizedCrop(height=64, width=64, scale=(0.7,1.0), p=0.2),
    A.MotionBlur(p=0.1),
    A.GridDistortion(p=0.1),
    A.CLAHE(p=0.1)"""

def augment_images(images, augment_per_image=2):
    augmented = []
    for img in tqdm(images):
        for _ in range(augment_per_image):
            augmented_img = transform(image=(img*255).astype(np.uint8))["image"]
            # HER ZAMAN 64x64'e resize!
            if augmented_img.shape[:2] != (64, 64):
                augmented_img = cv2.resize(augmented_img, (64, 64))
            if len(augmented_img.shape) == 2:  # Sıkça olur (64, 64)
                augmented_img = np.expand_dims(augmented_img, axis=-1)  # (64,64,1)
            elif len(augmented_img.shape) == 3 and augmented_img.shape[-1] != 1:
                # Eğer yanlışlıkla 3 kanal dönerse, channel'ı ort'
                augmented_img = cv2.cvtColor(augmented_img, cv2.COLOR_BGR2GRAY)
                augmented_img = np.expand_dims(augmented_img, axis=-1)
            augmented_img = augmented_img.astype(np.float32) / 255.0
            augmented.append(augmented_img)
    augmented = np.stack(augmented, axis=0)
    return augmented


train_data_path = 'data/raw/train/'
test_data_path = 'data/raw/test/'

train_images, train_filenames = load_and_preprocess_images(train_data_path)
test_images, test_filenames  = load_and_preprocess_images(test_data_path)

augmented_train_images = augment_images(train_images, augment_per_image=10)
total_train_images = np.concatenate((train_images, augmented_train_images), axis=0)

print('Train image shape:', train_images.shape, 
      '| Augmented shape:', augmented_train_images.shape, 
      '| Total shape:', total_train_images.shape)

# 3. İstenirse shuffle (Tavsiye edilir)
idx = np.random.permutation(len(total_train_images))
total_train_images = total_train_images[idx]

processed_data_path = 'data/processed/'
np.save(os.path.join(processed_data_path, 'processed_train_images.npy'), total_train_images)
np.save(os.path.join(processed_data_path, 'processed_test_images.npy'), test_images)
np.save(os.path.join(processed_data_path, 'test_filenames.npy'), np.array(test_filenames))


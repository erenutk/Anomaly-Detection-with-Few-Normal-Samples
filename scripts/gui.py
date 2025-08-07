import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
import joblib
from skimage.feature import hog

# --- MODELLERİ YÜKLE ---
cae = tf.keras.models.load_model("models/cae_best.keras", compile=False)
THRESHOLD_CAE = 0.0025  # kendi optimum threshold'un

# OCSVM + HOG ile pipeline
ocsvm = joblib.load('models/ocsvm_model.pkl')
ocsvm_scaler = joblib.load('models/ocsvm_scaler.pkl')
ocsvm_pca = joblib.load('models/ocsvm_pca.pkl')

# ISOForest + HOG ile pipeline
iforest = joblib.load('models/isoforest.pkl')
iforest_scaler = joblib.load('models/isoforest_scaler.pkl')
iforest_pca = joblib.load('models/isoforest_pca.pkl')


def test_leaf(img, model_name):
    # Grayscale, resize, normalize işlemleri
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rs = cv2.resize(img_gray, (64, 64))

    # --- CAE ---
    if model_name == "Convolutional Autoencoder":
        normed = img_rs.astype(np.float32) / 255.0
        inp = np.expand_dims(normed, axis=(0, -1))
        recon = cae.predict(inp)
        error = np.mean((inp - recon) ** 2)
        sonuc = "‼️ ANOMALİ ‼️" if error > THRESHOLD_CAE else "✔️ SAĞLIKLI"
        result_text = f"MSE: {error:.5f} | {sonuc}"
        rec_img = recon[0,...,0]
        diff_img = np.abs(normed - rec_img)
        return result_text, rec_img, diff_img

    # --- OCSVM + HOG ---
    elif model_name == "OCSVM + HOG":
        hog_feat = hog(img_rs, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
        hog_feat = hog_feat.reshape(1, -1)
        hog_feat_scaled = ocsvm_scaler.transform(hog_feat)
        hog_feat_pca = ocsvm_pca.transform(hog_feat_scaled)
        pred = ocsvm.predict(hog_feat_pca)
        sonuc = "‼️ ANOMALİ ‼️" if pred[0] == -1 else "✔️ SAĞLIKLI"
        result_text = f"OCSVM Tahmin: {sonuc}"
        # Görsel yerine orijinali ve siyah gürültü döndürelim:
        return result_text, img_rs, np.zeros_like(img_rs)

    # --- IF + HOG ---
    elif model_name == "Isolation Forest + HOG":
        hog_feat = hog(img_rs, pixels_per_cell=(8,8), cells_per_block=(2,2), feature_vector=True)
        hog_feat = hog_feat.reshape(1, -1)
        hog_feat_scaled = iforest_scaler.transform(hog_feat)
        pred = iforest.predict(hog_feat_scaled)
        sonuc = "‼️ ANOMALİ ‼️" if pred[0] == -1 else "✔️ SAĞLIKLI"
        result_text = f"IF Tahmin: {sonuc}"
        return result_text, img_rs, np.zeros_like(img_rs)


### Gradio arayüzü
with gr.Blocks(theme=gr.themes.Base()) as demo:
    gr.Markdown("""
    # 🍅 Yaprak Anomali Tespiti
    Model seçimi yap, görseli yükle ve sonucu gör!
    """)
    with gr.Row():
        with gr.Column():
            model_choice = gr.Dropdown(
                label="Model Seçiniz:",
                choices=["Convolutional Autoencoder", "OCSVM + HOG", "Isolation Forest + HOG"],
                value="Convolutional Autoencoder"
            )
            input_image = gr.Image(label="Yaprak Görseli")
            run_button = gr.Button("Test Et!")
        with gr.Column():
            output_label = gr.Textbox(label="Tahmin Sonucu", interactive=False)
            if(model_choice == "Convolutional Autoencoder"):
                rec_img = gr.Image(label="Yeniden İnşa Görseli")
                diff_img = gr.Image(label="Hata Haritası")


    run_button.click(test_leaf, inputs=[input_image, model_choice], outputs=[output_label, rec_img, diff_img])

    gr.Markdown("""
    <small>
    <i>Bu arayüzde seçtiğiniz model ile anomaliyi hızlıca test edebilirsiniz.<br>
    Görsel örnek: 64x64 boyuta göre otomatik ayarlanır.<br>
    "Model Çıktısı" (CAE hariç) anlamlı görsel değildir; sadece CAE'de gerçek yeniden inşa ve error map çıkarılır.
    </i>
    </small>
    """)

demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
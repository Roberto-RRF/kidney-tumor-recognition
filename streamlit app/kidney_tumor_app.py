import streamlit as st
import numpy as np
from PIL import Image
import torch
import joblib
import torch.nn as nn
import cv2
import matplotlib.pyplot as plt
import xgboost as xgb

# Definir la arquitectura de la CNN mejorada para imágenes 512x512
class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        
        # Bloque convolucional 1 - 512x512 -> 256x256
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        
        # Bloque convolucional 2 - 256x256 -> 128x128
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.2)
        )
        
        # Bloque convolucional 3 - 128x128 -> 64x64
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        # Bloque convolucional 4 - 64x64 -> 32x32
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.3)
        )
        
        # Bloque convolucional 5 - 32x32 -> 16x16
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout(0.4)
        )
        
        # Capas densas
        # Para IMG_SIZE=(512,512): después de 5 pooling -> 512->256->128->64->32->16
        # Feature map final: 512 * 16 * 16 = 131072
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 16 * 16, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.fc(x)
        return x

# =========================================================
# 1. LOAD MODELS
# =========================================================
@st.cache_resource
def load_rf():
    return joblib.load("models/random_forest_model.pkl")

@st.cache_resource
def load_xgb():
    model = xgb.XGBClassifier()
    model.load_model("models/xgboost_model.json")
    return model


@st.cache_resource
def load_cnn(num_classes=2):
    """
    Carga un modelo CNN desde un archivo.
    
    Args:
        model_path: Ruta del archivo del modelo
        device: Dispositivo (CPU o GPU)
        num_classes: Número de clases
    
    Returns:
        Modelo CNN cargado
    """
    # Crear instancia del modelo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    modelo = CNN(num_classes=num_classes).to(device)
    
    # Cargar pesos
    checkpoint = torch.load('models\cnn_model_final.pth', map_location=device)
    
    if 'model_state_dict' in checkpoint:
        modelo.load_state_dict(checkpoint['model_state_dict'])
        print(f"Train Accuracy: {checkpoint.get('train_acc', 'N/A')}")
        print(f"Val Accuracy: {checkpoint.get('val_acc', 'N/A')}")
        print(f"Test Accuracy: {checkpoint.get('test_acc', 'N/A')}")
    else:
        modelo.load_state_dict(checkpoint)
        
    
    modelo.eval()
    return modelo


# =========================================================
# 2. PREPROCESSING FOR EACH MODEL
# =========================================================
def preprocess_rf(image):
    """Random Forest expects 512x512"""
    img = image.resize((512, 512))
    arr = np.array(img).astype("float32") / 255.0
    # flatten if your RF uses pixels directly
    return arr.flatten()


def preprocess_xgb(image):
    """XGBoost expects 124x124"""
    img = image.resize((124, 124))
    arr = np.array(img).astype("float32") / 255.0
    return arr.flatten()


def preprocess_cnn(image, device=None):
    """CNN expects 512x512 (3-channel tensor)"""
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    img = image.resize((512, 512))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # to CHW
    tensor = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)
    return tensor


# =========================================================
# 3. GRAD-CAM IMPLEMENTATION
# =========================================================
class GradCAM:
    """Implementa Grad-CAM para visualización de mapas de activación"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Registrar hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        """
        Genera el mapa de activación de clase (CAM)

        Args:
            input_image: Tensor de entrada (1, C, H, W)
            target_class: Clase objetivo (si es None, usa la clase predicha)

        Returns:
            cam: Mapa de calor normalizado
        """
        # Forward pass
        model_output = self.model(input_image)

        if target_class is None:
            target_class = model_output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_score = model_output[:, target_class]
        class_score.backward()

        # Calcular pesos
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        # Ponderar activaciones
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        # Promedio de canales
        heatmap = torch.mean(self.activations, dim=1).squeeze()

        # ReLU y normalización
        heatmap = torch.maximum(heatmap, torch.tensor(0.0))
        heatmap /= torch.max(heatmap) if torch.max(heatmap) > 0 else torch.tensor(1.0)

        return heatmap.cpu().numpy()

def apply_colormap_on_image(org_img, activation_map, alpha=0.5):
    """
    Superpone el mapa de activación sobre la imagen original

    Args:
        org_img: Imagen original (PIL o numpy array)
        activation_map: Mapa de activación
        alpha: Transparencia del mapa de calor

    Returns:
        Imagen con el mapa de calor superpuesto
    """
    # Convertir PIL a numpy si es necesario
    if isinstance(org_img, Image.Image):
        org_img = np.array(org_img)

    # Redimensionar mapa de activación al tamaño de la imagen
    heatmap = cv2.resize(activation_map, (org_img.shape[1], org_img.shape[0]))

    # Aplicar colormap
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Superponer
    superimposed_img = heatmap * alpha + org_img * (1 - alpha)
    superimposed_img = np.uint8(superimposed_img)

    return superimposed_img


# =========================================================
# 4. PREDICTORS
# =========================================================
def predict_with_rf(model, features):
    return model.predict([features])[0]

def predict_with_xgb(model, features):
    # Reshape features to 2D array as expected by XGBoost
    features_reshaped = np.array(features).reshape(1, -1)
    # Use predict_proba to get probabilities, then argmax to get class
    pred_proba = model.predict(features_reshaped)
    # Get the class with highest probability
    pred = np.argmax(pred_proba, axis=1)[0]
    return int(pred)


# =========================================================
# 5. STREAMLIT UI
# =========================================================
st.set_page_config(page_title="Kidney Tumor Classifier", layout="wide")

st.title("🔍 Kidney Tumor Classification")
st.markdown("---")

# Create two main columns
left_col, right_col = st.columns([1, 2])

with left_col:
    st.subheader("📤 Input")

    uploaded_file = st.file_uploader(
        "Upload a kidney image",
        type=["jpg", "png", "jpeg"],
        help="Select a kidney CT scan or MRI image"
    )

    model_choice = st.selectbox(
        "Select Model:",
        ["Random Forest (512×512)", "XGBoost (124×124)", "CNN (512×512)"],
        help="Choose which model to use for prediction"
    )

    if uploaded_file:
        st.markdown("### Preview")
        original = Image.open(uploaded_file).convert("RGB")
        st.image(original, caption="Uploaded Image", use_container_width=True)
    else:
        st.info("👆 Please upload an image to get started")

with right_col:
    if uploaded_file:
        original_resized = original.resize((512, 512))
        class_names = ["Healthy", "Tumor"]

        # ======================================================
        # Always run CNN for heatmap visualization
        # ======================================================
        with st.spinner("Loading CNN model and generating visualization..."):
            cnn_model = load_cnn()
            cnn_model.eval()

            # Enable gradients for Grad-CAM
            cnn_tensor = preprocess_cnn(original)
            cnn_tensor.requires_grad = True

            # Get prediction
            cnn_out = cnn_model(cnn_tensor)
            cnn_probs = torch.softmax(cnn_out, dim=1)[0]
            cnn_label = torch.argmax(cnn_probs).item()
            confidence = cnn_probs[cnn_label].item()

            # Generate Grad-CAM
            grad_cam = GradCAM(cnn_model, cnn_model.conv5[-3])
            cam = grad_cam.generate_cam(cnn_tensor, target_class=cnn_label)

            # Apply colormap on image
            img_with_heatmap = apply_colormap_on_image(original_resized, cam, alpha=0.4)
            heatmap_colored = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Display CNN Grad-CAM Visualization
        st.subheader("🔥 CNN Grad-CAM Visualization")
        st.markdown("*Showing regions the CNN focuses on for prediction*")

        viz_col1, viz_col2, viz_col3 = st.columns(3)

        with viz_col1:
            st.image(original_resized, caption="Original", use_container_width=True)

        with viz_col2:
            st.image(heatmap_colored, caption="Activation Map", use_container_width=True)

        with viz_col3:
            st.image(img_with_heatmap, caption="Overlay", use_container_width=True)

        st.markdown("---")

        # ======================================================
        # Selected Model Prediction
        # ======================================================
        st.subheader("🎯 Model Prediction")

        with st.spinner(f"Running {model_choice.split()[0]} model..."):
            if "Random Forest" in model_choice:
                rf_model = load_rf()
                features = preprocess_rf(original)
                pred = predict_with_rf(rf_model, features)

                # Display result with colored box
                result_color = "green" if pred == 0 else "red"
                st.markdown(f"### Model: **Random Forest**")
                st.markdown(f"### Prediction: :{result_color}[**{class_names[pred]}**]")

            elif "XGBoost" in model_choice:
                xgb_model = load_xgb()
                features = preprocess_xgb(original)
                pred = predict_with_xgb(xgb_model, features)

                result_color = "green" if pred == 0 else "red"
                st.markdown(f"### Model: **XGBoost**")
                st.markdown(f"### Prediction: :{result_color}[**{class_names[pred]}**]")

            elif "CNN" in model_choice:
                result_color = "green" if cnn_label == 0 else "red"
                st.markdown(f"### Model: **CNN (Deep Learning)**")
                st.markdown(f"### Prediction: :{result_color}[**{class_names[cnn_label]}**]")
                st.markdown(f"**Confidence:** {confidence:.2%}")

                # Show probability bar
                st.progress(confidence)

    else:
        st.info("📊 Results will appear here once you upload an image")
        st.markdown("### How to use:")
        st.markdown("""
        1. **Upload** a kidney image using the file uploader on the left
        2. **Select** your preferred model (Random Forest, XGBoost, or CNN)
        3. **View** the Grad-CAM visualization showing where the CNN focuses
        4. **Review** the prediction results

        ---

        **Model Information:**
        - **Random Forest**: Traditional ML, uses 512×512 images
        - **XGBoost**: Gradient boosting, uses 124×124 images
        - **CNN**: Deep learning, uses 512×512 images with visualization
        """)

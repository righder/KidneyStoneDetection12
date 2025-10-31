# import streamlit as st
# import torch
# import torch.nn as nn
# from torchvision import transforms, models
# from PIL import Image

# # ======================================================
# # 1. Page Setup
# # ======================================================
# st.set_page_config(page_title="Kidney Stone Detection", page_icon="🩻", layout="centered")
# st.title("🩺 Kidney Stone Detection ")
# st.write("Upload a **CT scan image** to detect whether it contains a kidney stone or not.")

# # ======================================================
# # 2. Load Model
# # ======================================================
# @st.cache_resource
# def load_model():
#     model = models.efficientnet_b5(weights=None)
#     model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
#     model.load_state_dict(torch.load("best_efficientnet_B5_kidney.pth", map_location=torch.device('cpu')))
#     model.eval()
#     return model

# model = load_model()
# classes = ['Non-Stone', 'Stone']

# # ======================================================
# # 3. Define Transform (same as validation pipeline)
# # ======================================================
# transform = transforms.Compose([
#     transforms.Grayscale(num_output_channels=3),
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406],
#                          [0.229, 0.224, 0.225])
# ])


# # ======================================================
# # 4. Upload Image
# # ======================================================
# uploaded_file = st.file_uploader("📁 Upload CT Image", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert('RGB')
#     st.image(image, caption='Uploaded CT Scan', use_container_width=True)

#     # Preprocess
#     input_tensor = transform(image).unsqueeze(0)

#     # Predict
#     with torch.no_grad():
#         outputs = model(input_tensor)
#         probs = torch.softmax(outputs, dim=1)
#         pred_class = torch.argmax(probs, dim=1).item()
#         confidence = probs[0][pred_class].item() * 100

#     # ==================================================
#     # 5. Display Result
#     # ==================================================
#     st.markdown("---")
#     st.subheader("🧾 Prediction Result")
#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric(label="Predicted Class", value=classes[pred_class])
#     with col2:
#         st.metric(label="Confidence", value=f"{confidence:.2f}%")

#     # Add color indication
#     if pred_class == 1:
#         st.success("✅ Kidney Stone Detected")
#     else:
#         st.info("🩶 No Kidney Stone Detected")

# st.markdown("---")
# st.caption("Developed by Satyam Nayak | Powered by EfficientNet-B5 + Streamlit")









import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

# Remove extra top padding and adjust spacing
st.markdown("""
    <style>
        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 1rem;
        }
        h1 {
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
        }
    </style>
""", unsafe_allow_html=True)


# ======================================================
# 1. Page Setup
# ======================================================
st.set_page_config(page_title="Kidney Stone Detection", page_icon="🩻", layout="wide")
st.title("🩺 Kidney Stone Detection")
st.write("Upload a **CT scan image** to detect whether it contains a kidney stone or not.")

# ======================================================
# 2. Load Model
# ======================================================
@st.cache_resource
def load_model():
    model = models.efficientnet_b5(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("best_efficientnet_B5_kidney.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()
classes = ['Non-Stone', 'Stone']

# ======================================================
# 3. Define Transform
# ======================================================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======================================================
# 4. Upload Image
# ======================================================
uploaded_file = st.file_uploader("📁 Upload CT Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')

    # Preprocess
    input_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred_class].item() * 100

    # ==================================================
    # 5. Layout: Left = Prediction | Right = Image + Chart
    # ==================================================
    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    # LEFT SIDE — Prediction Results
    with col1:
        st.markdown("### 🧾 Prediction Result")
        st.metric(label="Predicted Class", value=classes[pred_class])
        st.metric(label="Confidence", value=f"{confidence:.2f}%")
        if pred_class == 1:
            st.success("✅ Kidney Stone Detected")
        else:
            st.info("🩶 No Kidney Stone Detected")

    # RIGHT SIDE — Image and Chart side by side
    with col2:
        img_col, chart_col = st.columns([1, 1], vertical_alignment="top")

        # Image
        with img_col:
            st.image(image, caption="Uploaded CT Scan", width=320)

        # Horizontal Bar Chart
        with chart_col:
            st.markdown("### 📊 Confidence Distribution")
            fig, ax = plt.subplots(figsize=(3, 1.3))
            vals = probs[0].tolist()
            colors = ['gray', 'green'] if pred_class == 1 else ['green', 'gray']
            ax.barh(classes, vals, color=colors, height=0.4)
            ax.set_xlim([0, 1])
            ax.set_xlabel("Probability", fontsize=8)
            ax.tick_params(labelsize=7)
            for i, v in enumerate(vals):
                ax.text(min(v + 0.02, 0.98), i, f"{v*100:.1f}%", va='center', fontsize=7)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=False)

st.markdown("---")
st.caption("Developed by **Satyam Nayak** | Powered by EfficientNet-B5 + Streamlit")

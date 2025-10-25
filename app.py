import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

@st.cache_resource
def load_model(model_path, num_classes=4):
    """Load PyTorch model from file."""
    try:
        loaded = torch.load(model_path, map_location='cuda')
        
        # If it's already a model, return it
        if isinstance(loaded, nn.Module):
            loaded.eval()
            return loaded
        
        # Otherwise, load state_dict into GoogLeNet
        state_dict = loaded.get('state_dict', loaded) if isinstance(loaded, dict) else loaded
        
        model = models.googlenet(weights=None, aux_logits=True)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )
        
        # Load weights, ignoring shape mismatches
        model_dict = model.state_dict()
        filtered = {k.replace('module.', ''): v for k, v in state_dict.items() 
                   if k.replace('module.', '') in model_dict and 
                   model_dict[k.replace('module.', '')].shape == v.shape}
        model_dict.update(filtered)
        model.load_state_dict(model_dict)
        model.eval()
        return model
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Setup
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class_names = ["Benign", "early Pre-B", "Pre-B", "Pro-B"]
model = load_model('final_model.pth', num_classes=len(class_names))

# UI
st.title("Image Classification App")
st.write("Upload an image for classification into one of four classes.")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image.", use_container_width=True)
    st.write("Classifying...")
    
    # Predict
    input_batch = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_batch)
    
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_idx = torch.argmax(probabilities).item()
    
    # Display results
    st.success(f"Prediction: **{class_names[predicted_idx]}**")
    st.write(f"Confidence: **{probabilities[predicted_idx].item() * 100:.2f}%**")
    
    st.subheader("All Class Probabilities:")
    for name, prob in zip(class_names, probabilities):
        st.write(f"{name}: {prob.item() * 100:.2f}%")

elif uploaded_file:
    st.error("Model failed to load. Check the console for errors.")

st.sidebar.markdown("---")
st.sidebar.info("This app uses a pre-trained PyTorch model for image classification.")
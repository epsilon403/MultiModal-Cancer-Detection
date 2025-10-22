import streamlit as st
import torch
import torch.nn as nn # You might need other torch modules depending on your model
import torchvision.transforms as transforms
from PIL import Image


# Load your trained model
@st.cache_resource # Cache the model loading to avoid reloading on every rerun
def load_model(model_path):
    # Make sure your model class is defined or imported if it's custom
    # If your model is a custom class, you'll need to instantiate it first
    # For example:
    # model = SimpleCNN()
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    # If it's a model saved using torch.save(model, 'final_model.pth')
    model = torch.load(model_path, map_location=torch.device('cpu'))
    model.eval() # Set the model to evaluation mode
    return model

model_path = 'final_model.pth'
model = load_model(model_path)

# Define your image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)), # Adjust size as per your model's input
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
])

# Define your classes
class_names = ["Benign", "early Pre-B", "Pre-B", "Pro-B"]

st.title("Image Classification App")
st.write("Upload an image for classification into one of four classes.")

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image.", use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0) # Create a mini-batch as expected by the model

    with torch.no_grad():
        output = model(input_batch)

    # Get the predicted class
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    predicted_class_idx = torch.argmax(probabilities).item()
    predicted_class_name = class_names[predicted_class_idx]
    confidence = probabilities[predicted_class_idx].item() * 100

    st.success(f"Prediction: **{predicted_class_name}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    st.subheader("All Class Probabilities:")
    for i, prob in enumerate(probabilities):
        st.write(f"{class_names[i]}: {prob.item()*100:.2f}%")

st.sidebar.markdown("---")
st.sidebar.info("This app uses a pre-trained PyTorch model for image classification.")
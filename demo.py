import cv2
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Function to preprocess the image
def preprocess_image(image):
    img = cv2.resize(image, (224, 224))  # Adjust the size according to your model's input size
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))  # Transpose the image to (C, H, W) format
    return img

# Create the Xception model with pre-trained weights using the updated model name
model = timm.create_model('legacy_xception', pretrained=False)

# Modify the classifier for binary classification
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

# Load the state dictionary from the saved checkpoint
checkpoint = torch.load('Xception_model.pth')
model.load_state_dict(checkpoint)


# Class labels
class_labels = ['True', 'False']  # Update with your class labels

st.title("DermaChat")
st.write('Upload a skin lesion image for classification')

# add a button to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:

    # Convert the file to opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    st.image(img, caption='Uploaded Image', use_column_width=True)


    # Preprocess the input image
    def preprocess_image(image):
        transform = transforms.Compose([
            transforms.ToPILImage(),  # Convert numpy array to PIL Image
            transforms.Resize((224, 224)),  # Resize to match the input size of your model
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
        ])
        image = transform(image)
        return image.unsqueeze(0)  # Add batch dimension


    # Perform inference
    def predict(image):
        image_tensor = preprocess_image(image)
        with torch.no_grad():
            model.eval()
            output = model(image_tensor)
            probabilities = torch.sigmoid(output)[0]  # Probabilities for both classes
            prob_true = probabilities[1].item()  # Probability for the positive class (true)
            prob_false = probabilities[0].item()  # Probability for the negative class (false)
            predicted_class = "True" if prob_true > 0.5 else "False"  # ToDo Adjust the threshold ??
        return predicted_class, prob_true, prob_false


    # Example usage
    predicted_class, prob_true, prob_false = predict(img)

    print("Predicted Class:", predicted_class)
    st.write("Predicted Class:", predicted_class)

    print("Probability of True:", prob_true)
    st.write("Probability of True:", prob_true)

    print("Probability of False:", prob_false)
    st.write("Probability of False:", prob_false)

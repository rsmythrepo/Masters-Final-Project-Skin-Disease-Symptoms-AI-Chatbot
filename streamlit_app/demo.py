import cv2
import numpy as np
import streamlit as st
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

st.set_page_config(
    page_title="Multipage App",
    layout='wide'
)
st.title("DermaChat")
col1, col2 = st.columns([1,1])

# Col1 Left pane image processing
col1.markdown("### Images")
col1.write('Upload a skin lesion image for classification')

# Side bar functionality
st.sidebar.success("Select a page above.")

# Export Button
st.sidebar.button('Export Report')
#Todo add export report functionalities here

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
checkpoint = torch.load('../project_code/models/cnn/Xception_model.pth')
model.load_state_dict(checkpoint)


# Class labels
class_labels = ['True', 'False']  # Update with your class labels



# add a button to upload an image
uploaded_file = col1.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:

    # Convert the file to opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image
    col1.image(img, caption='Uploaded Image', use_column_width=False)


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

    # Concatenating strings and variables into a single string
    predicted_class_text = "Predicted Class: " + str(predicted_class)
    prob_true_text = "Probability of True: " + str(prob_true)
    prob_false_text = "Probability of False: " + str(prob_false)

    # Writing the concatenated strings to the column
    #col1 = st.sidebar
    col1.write(predicted_class_text)
    col1.write(prob_true_text)
    col1.write(prob_false_text)

    #print("Predicted Class:", predicted_class)
    #col1.write("Predicted Class:", predicted_class)

    #print("Probability of True:", prob_true)
    #col1.write("Probability of True:", prob_true)

    #print("Probability of False:", prob_false)
    #col1.write("Probability of False:", prob_false)


# Col2 - Right pane - Chatbot
# ToDo Chatbot code here - instead of st. use col2. to print on the left pane
col2.markdown("### Chatbot")
col2.write("example")
col2.write("example")
col2.write("example")








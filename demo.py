import cv2
import numpy as np
import streamlit as st
import pickle
import cv2

# Function to preprocess the image
def preprocess_image(image):
    img = cv2.resize(image, (32, 32))
    img = img.astype(np.float32) / 255.0
    return img

# Load the model
#model = load_model('skin_cancer_model.pkl')  # Update with your model path
model = pickle.load(open('skin_cancer_model.pkl', 'rb'))

# Class labels
class_labels = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']  # Update with your class labels

st.title("DermaChat")
st.write('Upload a skin lesion image for classification')

# add a button to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])
if uploaded_file is not None:

    # Convert the file to opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Display the image
    #st.image(img, channels="BGR", use_column_width=True)
    # Display the uploaded image
    #img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image
    img_array = np.array(img)
    preprocessed_img = preprocess_image(img_array)
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)

    # Perform prediction
    prediction = model.predict(preprocessed_img)
    predicted_class = class_labels[np.argmax(prediction)]

    # Display the prediction
    st.write(f"Predicted Class: {predicted_class}")
    st.write("Prediction Probabilities:")
    for i, prob in enumerate(prediction[0]):
        st.write(f"{class_labels[i]}: {prob}")
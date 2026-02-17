import streamlit as st
import cv2
import numpy as np

st.title("Face Detection App")
uploaded_file = st.file_uploader("Upload an image...", type=['jpg', 'png', 'jpeg'])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the RED bold rectangular frame
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 5)

    # Show result
    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption='Detected Faces', use_column_width=True)

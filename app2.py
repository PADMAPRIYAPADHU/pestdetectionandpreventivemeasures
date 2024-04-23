import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from disease import *
from remedies import *  # Import your disease information


# Load the YOLO model
model = YOLO('best.pt')
crop_options = ['Citrus', 'Paddy', 'Cotton', 'Mango', 'Tomato', 'Ground nuts', "Lady's finger", 'Pomegranate']

# Set confidence threshold
conf_thresh = 0.25
selected_crop = st.radio("Select a Crop:", crop_options)

# Streamlit app
st.title("Pest detection and remedies for crop pest detection")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Predict using the YOLO model
    results = model.predict(image, stream=True)

    # Iterate over the generator
    for result in results:
        # Extract bounding boxes and classes
        boxes = result.boxes.xyxy.cpu().numpy()
        names = result.names
        classes = result.boxes.cls.cpu().numpy()

        # Iterate over boxes and classes
        for box, cls in zip(boxes, classes):
            r = [int(coord) for coord in box[:4]]  # use only the first 4 coordinates
            class_label = names[cls]  # get the class label from the dictionary

            # Draw boxes on the image with blue color
            cv2.rectangle(image, (r[0], r[1]), (r[2], r[3]), (255, 0, 0), 2)  # (255, 0, 0) represents blue in BGR format

            # Display label on the image
            label_text = f"Class: {class_label}"
            cv2.putText(image, label_text, (r[0], r[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)  # (255, 0, 0) for blue text

            # Display the image in Streamlit
            st.image(image, channels="BGR", caption="Predicted Image", use_column_width=True)
            disease= None

            # Display information for each detection
            st.warning(f"Detected class: {class_label}")
            st.write("please refer to the wikipedia link bellow to know more about the pest")
            st.success(f"https://en.wikipedia.org/wiki/{class_label}")
            try:
                disease = crop_diseases[selected_crop][class_label]
            except:
                st.success("no disease found with the associated crop")
            try:
                disease = crop_diseases[selected_crop][class_label]
                st.warning(f"Disease associated with {selected_crop} and {class_label}: {disease}")
                #text_speech.say(disease)
                #text_speech.runAndWait()
            
                
            except KeyError:
                st.success(f"The selected crop {selected_crop} is not affected by the detected pest.")
            
            try:
                if disease is not None:
                    for i in disease:
                        #de=preventive_measures[i]
                        st.success(f"Preventive measures: {preventive_measures[i]}")
                        
            except KeyError:
                st.warning(f"There is no preventive measure associated with the detected disease.")

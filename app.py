import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os

from chest_cancer_classification.pipeline.prediction import PredictionPipeline


def main():
    st.title("Chest Cancer Classification")
    st.write("Upload a chest CT scan image to predict Adenocarcinoma Cancer or Normal.")

    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Open image
        image = Image.open(uploaded_file)

        # Show uploaded image
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("")
        st.write("Classifying...")

        # 🔧 FIX: Convert RGBA / other modes to RGB (JPEG-safe)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Save the uploaded image to a temp file
        temp_filename = "inputImage.jpg"
        image.save(temp_filename, format="JPEG")

        try:
            # Run prediction
            predictor = PredictionPipeline(temp_filename)
            result = predictor.predict()[0]

            st.success("Prediction completed successfully ✅")

            st.write(f"**Prediction:** {result['image']}")
            st.write(f"**Probabilities:** {result['probabilities']}")
            st.write(f"**Predicted class index:** {result['predicted_class']}")

        except Exception as e:
            st.error("Prediction failed ❌")
            st.exception(e)

        finally:
            # Clean up temp file
            if os.path.exists(temp_filename):
                os.remove(temp_filename)


if __name__ == "__main__":
    main()

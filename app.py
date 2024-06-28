import streamlit as st
import os
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import tempfile
from reportlab.pdfgen import canvas
from vertexai.language_models import ChatModel
import vertexai

# Load environment variables from .env
load_dotenv()

# Set Google Cloud credentials from JSON key file
json_key_file = "gen-lang-client-0651086807-7a7c02723fa8.json"  # Replace with your actual JSON key file name

# If the JSON key file is in the current working directory, set its path
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_file

# Initialize Vertex AI
project = "gen-lang-client-0651086807"  # Replace with your Vertex AI project ID
location = "us-central1"  # Replace with your Vertex AI location
vertexai.init(project=project, location=location)

# Load the Vertex AI ChatModel
chat_model = ChatModel.from_pretrained("chat-bison@002")

# Class labels for brain tumor types
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Function to preprocess image for the model
def preprocess_image(img):
    img = image.load_img(img, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Function to load the CNN model
@st.cache(allow_output_mutation=True)
def load_model():
    model_path = 'brain_tumor_classifier.h5'  # Replace with your actual model path
    model = tf.keras.models.load_model(model_path)
    return model

# Function to predict brain tumor type and generate medical advice
def predict_and_generate_text(uploaded_file):
    try:
        if uploaded_file is None:
            raise ValueError("No image file uploaded.")

        # Load the model
        model = load_model()

        # Preprocess uploaded image
        processed_img = preprocess_image(uploaded_file)

        # Predict using the model
        prediction = model.predict(processed_img)
        predicted_class_index = np.argmax(prediction)
        predicted_class = class_labels[predicted_class_index]

        # Debugging: Print prediction details
        st.write(f"Predicted class: {predicted_class}")
        st.write(f"Prediction probabilities: {prediction}")

        # Generate medical advice only for specific tumor classes
        if predicted_class in ['glioma', 'meningioma', 'pituitary']:
            # Initialize Vertex AI ChatModel
            chat_session = chat_model.start_chat(context="")

            # Constructing the prompt based on the predicted class
            prompt = f"The patient has been diagnosed with a {predicted_class} brain tumor. Provide detailed medical advice including:"

            if predicted_class == 'glioma':
                prompt += (
                    " 1. Medical Condition: You have a glioma located in the frontal lobe.\n"
                    " 2. Medication Recommendations: Suggest suitable medications including chemotherapy drugs like Temozolomide, steroids to reduce brain swelling.\n"
                    " 3. Treatment Guidance: Recommend necessary treatments such as surgery to remove the tumor, followed by radiation therapy.\n"
                    " 4. Lifestyle Changes: Suggest lifestyle modifications to manage symptoms or improve overall health, such as regular exercise and a balanced diet.\n"
                    " 5. Monitoring and Specialist Referrals: Recommend ongoing monitoring by neurologists and oncologists for further evaluation and treatment."
                )

            elif predicted_class == 'meningioma':
                prompt += (
                    " 1. Medical Condition: You have a meningioma, typically located in the meninges surrounding the brain.\n"
                    " 2. Medication Recommendations: Suggest medications to control symptoms, possibly including anticonvulsants and corticosteroids.\n"
                    " 3. Treatment Guidance: Recommend surgical removal of the tumor, with radiation therapy considered if complete removal isn't feasible.\n"
                    " 4. Lifestyle Changes: Advise on lifestyle adjustments such as stress reduction techniques and adequate sleep.\n"
                    " 5. Monitoring and Specialist Referrals: Recommend regular follow-up with neurosurgeons and neurologists for monitoring and management."
                )

            elif predicted_class == 'pituitary':
                prompt += (
                    " 1. Medical Condition: You have a pituitary tumor affecting the pituitary gland.\n"
                    " 2. Medication Recommendations: Suggest hormone replacement therapy to manage hormone imbalances caused by the tumor.\n"
                    " 3. Treatment Guidance: Recommend surgical intervention to remove the tumor, or radiation therapy depending on the tumor's size and location.\n"
                    " 4. Lifestyle Changes: Recommend dietary adjustments and regular physical activity to support overall health.\n"
                    " 5. Monitoring and Specialist Referrals: Refer to endocrinologists and neurosurgeons for specialized care and ongoing monitoring."
                )

            # Additional prompts for all tumor types
            prompt += (
                " 6. Emotional Support: Emphasize the importance of emotional support and counseling throughout the treatment process.\n"
                " 7. Patient Education: Educate the patient about the tumor type, its potential complications, and treatment options.\n"
                " 8. Family Involvement: Encourage involvement of family members in decision-making and support.\n"
                " 9. Research and Clinical Trials: Discuss opportunities for participation in clinical trials for new treatments.\n"
                "10. Rehabilitation: Outline potential rehabilitation programs post-treatment to aid recovery and improve quality of life."
            )

            response = chat_session.send_message(prompt)
            return response.text
        else:
            return "No specific medical advice available for tumors classified as 'notumor'."

    except ValueError as ve:
        st.error(f"Error: {ve}")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Function to generate PDF report
def generate_pdf(description):
    if description is None:
        return None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        c = canvas.Canvas(tmpfile.name)
        c.setFont("Helvetica", 12)
        text_lines = description.split('\n')  # Ensure description is not None before splitting
        y = 750
        for line in text_lines:
            c.drawString(100, y, line)
            y -= 20
        c.save()
        tmpfile.close()
        return tmpfile.name

# Streamlit app
def main():
    st.title('Brain Tumor Classifier and Treatment Advisor')
    st.markdown('Upload a brain MRI image for analysis.')

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])  # Allow jpg, png, and jpeg files
    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded MRI Image', use_column_width=True)
        if st.button('Predict and Recommend'):
            try:
                with st.spinner('Analyzing the image...'):
                    output_text = predict_and_generate_text(uploaded_file)
                st.success("Medical Advice:")
                st.write(output_text)

                # Generate and download PDF report
                pdf_filename = generate_pdf(output_text)
                with open(pdf_filename, "rb") as pdf_file:
                    st.download_button(label="Download PDF", data=pdf_file, file_name="medical_advice.pdf", mime="application/pdf")

            except Exception as e:
                st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

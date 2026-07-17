import streamlit as st
import requests
from PIL import Image

# 1. Configuration: Connect to your FastAPI backend
API_URL = "http://localhost:8000/predict"

# 2. UI Layout Setup
st.set_page_config(page_title="Toxicity Vision", page_icon="🛡️", layout="centered")

st.title("🛡️ Visual Toxicity Classifier")
st.write("Upload an image. The system will use BLIP to describe it, and BertBiLSTM to score the text for toxicity.")

# 3. File Uploader component
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", width="stretch")

    # 4. Analysis Trigger
    if st.button("Analyze Image", type="primary"):
        with st.spinner("Processing image and running neural networks..."):
            try:
                # Reset the file pointer to the beginning before sending
                uploaded_file.seek(0)
                
                # Format the payload for FastAPI's UploadFile
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Send the POST request to our backend
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # A. Display the BLIP Caption
                    st.divider()
                    st.subheader("📝 Extracted Description")
                    st.info(f"**\"{data.get('description', 'No description generated.')}\"**")
                    
                    # B. Display the Toxicity Scores
                    st.subheader("📊 Toxicity Analysis")
                    scores = data.get("toxicity_scores", {})
                    
                    # Iterate through the 6 labels and create dynamic visual bars
                    for label, score in scores.items():
                        # Clean up the label name (e.g., 'severe_toxic' -> 'Severe Toxic')
                        display_label = label.replace("_", " ").title()
                        percentage = int(score * 100)
                        
                        # Create a layout with two columns for alignment
                        col1, col2 = st.columns([1, 3])
                        
                        # Determine severity color logic
                        with col1:
                            if score >= 0.5:
                                st.error(f"**{display_label}**")
                            elif score >= 0.2:
                                st.warning(f"**{display_label}**")
                            else:
                                st.success(f"**{display_label}**")
                                
                        with col2:
                            # Render the visual progress bar
                            st.progress(score, text=f"{percentage}%")
                            
                else:
                    st.error(f"Server Error: {response.status_code} - {response.text}")
            
            except requests.exceptions.ConnectionError:
                st.error("🚨 Connection Error: Cannot reach the backend API. Is your FastAPI server running on port 8000?")
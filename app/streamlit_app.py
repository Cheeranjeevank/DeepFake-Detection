import streamlit as st
import tempfile
import os
from PIL import Image
import pandas as pd
from pathlib import Path

# Add project root to python path to allow imports if run directly
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.inference import ImagePredictor
from src.video_processor import VideoPredictor
from src.explainability import GradCAMExplainer

st.set_page_config(
    page_title="DeepGuard - Deepfake Detection",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f3f4f6;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .disclaimer {
        font-size: 0.8rem;
        color: #6b7280;
        font-style: italic;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    predictor = ImagePredictor()
    video_predictor = VideoPredictor(predictor=predictor)
    explainer = GradCAMExplainer(predictor.model, predictor.device)
    return predictor, video_predictor, explainer

def main():
    st.sidebar.title("🛡️ DeepGuard")
    st.sidebar.markdown("Deep Learning-Based Deepfake Detection System")
    
    page = st.sidebar.radio("Navigation", ["Home", "Image Detection", "Video Detection", "Model Information", "About"])
    
    try:
        image_predictor, video_predictor, explainer = load_models()
        models_loaded = True
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        models_loaded = False

    if page == "Home":
        st.markdown('<p class="main-header">DeepGuard</p>', unsafe_allow_html=True)
        st.markdown("### AI-Powered Deepfake Detection System")
        st.write("Welcome to DeepGuard. This system uses transfer learning with EfficientNet-B0 to analyze visual media and detect whether it has been manipulated (Deepfake) or remains authentic (Real).")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("🖼️ **Image Analysis**\nUpload static images for instant authenticity verification.")
        with col2:
            st.info("🎥 **Video Analysis**\nExtract and analyze multiple frames from video content.")
        with col3:
            st.info("🔍 **Explainability**\nVisualize model decisions using Grad-CAM heatmaps.")
            
        st.image("https://images.unsplash.com/photo-1550751827-4bd374c3f58b?ixlib=rb-4.0.3&auto=format&fit=crop&w=1200&q=80", use_container_width=True)
        
    elif page == "Image Detection":
        st.markdown('<p class="main-header">Image Detection</p>', unsafe_allow_html=True)
        st.write("Upload an image to detect if it's a deepfake.")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None and models_loaded:
            image = Image.open(uploaded_file).convert("RGB")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
                
            with st.spinner("Analyzing image..."):
                result = image_predictor.predict_image(image)
                
            with col2:
                st.subheader("Analysis Results")
                
                if "error" in result:
                    st.error(result["error"])
                else:
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    
                    if prediction == "DEEPFAKE":
                        st.error(f"🚨 **PREDICTION: {prediction}**")
                    else:
                        st.success(f"✅ **PREDICTION: {prediction}**")
                        
                    st.metric("Confidence", f"{confidence * 100:.2f}%")
                    
                    st.write("### Probabilities")
                    st.progress(result["fake_probability"], text=f"Fake: {result['fake_probability']*100:.1f}%")
                    st.progress(result["real_probability"], text=f"Real: {result['real_probability']*100:.1f}%")
                    
                    if result.get("face_crop"):
                        st.subheader("Explainability (Grad-CAM)")
                        st.write("Highlighting regions that contributed to the prediction.")
                        target_class = 1 if prediction == "DEEPFAKE" else 0
                        heatmap = explainer.generate_heatmap(result["face_crop"], target_class=target_class)
                        
                        col_h1, col_h2 = st.columns(2)
                        with col_h1:
                            st.image(result["face_crop"], caption="Cropped Face", width=150)
                        with col_h2:
                            st.image(heatmap, caption="Grad-CAM Heatmap", width=150)

    elif page == "Video Detection":
        st.markdown('<p class="main-header">Video Detection</p>', unsafe_allow_html=True)
        st.write("Upload a video to analyze its frames for deepfake artifacts.")
        
        uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
        
        if uploaded_file is not None and models_loaded:
            # Save uploaded video to a temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") 
            tfile.write(uploaded_file.read())
            video_path = tfile.name
            
            st.video(video_path)
            
            if st.button("Analyze Video"):
                with st.spinner("Extracting frames and analyzing..."):
                    result = video_predictor.predict_video(video_path)
                    
                if "error" in result:
                    st.error(result["error"])
                else:
                    st.subheader("Overall Video Result")
                    
                    prediction = result["final_prediction"]
                    if prediction == "DEEPFAKE":
                        st.error(f"🚨 **FINAL PREDICTION: {prediction}**")
                    else:
                        st.success(f"✅ **FINAL PREDICTION: {prediction}**")
                        
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
                    col2.metric("Frames Analyzed", result["frames_analyzed"])
                    col3.metric("Suspicious Frames", result["suspicious_frames"])
                    col4.metric("Avg Fake Prob.", f"{result['mean_fake_probability'] * 100:.2f}%")
                    
                    st.subheader("Frame-Level Analysis")
                    
                    # Create a DataFrame for the table
                    df_data = []
                    for f in result["frame_results"]:
                        df_data.append({
                            "Frame Number": f["frame_num"],
                            "Timestamp (s)": f"{f['timestamp']:.2f}",
                            "Prediction": f["prediction"],
                            "Fake Prob (%)": f"{f['fake_probability']*100:.1f}",
                            "Real Prob (%)": f"{f['real_probability']*100:.1f}"
                        })
                    
                    df = pd.DataFrame(df_data)
                    st.dataframe(df, use_container_width=True)
                    
                    if result["suspicious_frames"] > 0:
                        st.subheader("Suspicious Frames Samples")
                        # Show up to 4 suspicious frames
                        suspicious = [f for f in result["frame_results"] if f["prediction"] == "DEEPFAKE"][:4]
                        cols = st.columns(len(suspicious))
                        for idx, frame_res in enumerate(suspicious):
                            with cols[idx]:
                                st.image(frame_res["face_crop"], caption=f"Frame {frame_res['frame_num']}")
                                
            # Cleanup temp file
            try:
                os.unlink(video_path)
            except Exception:
                pass

    elif page == "Model Information":
        st.markdown('<p class="main-header">Model Information</p>', unsafe_allow_html=True)
        
        st.write("### Architecture")
        st.write("**Model:** EfficientNet-B0")
        st.write("EfficientNet-B0 is a lightweight convolutional neural network that provides state-of-the-art accuracy with fewer parameters compared to older architectures like ResNet or VGG. It uses a compound scaling method that uniformly scales the network's width, depth, and resolution.")
        
        st.write("### Pipeline Details")
        st.markdown("""
        1. **Face Detection**: MTCNN (Multi-task Cascaded Convolutional Networks) is used to locate and crop faces from input media.
        2. **Transfer Learning**: The EfficientNet-B0 backbone is initialized with weights pre-trained on ImageNet.
        3. **Classification Head**: The final fully-connected layer is replaced with a custom layer to output two classes: `REAL` and `DEEPFAKE`.
        4. **Input Size**: Images are resized to 224×224 pixels and normalized.
        5. **Framework**: Developed using PyTorch and torchvision.
        """)
        
    elif page == "About":
        st.markdown('<p class="main-header">About DeepGuard</p>', unsafe_allow_html=True)
        st.write("This project was built as an academic requirement for the B.Tech Artificial Intelligence and Data Science program.")
        st.write("It demonstrates an end-to-end pipeline from data preprocessing to model deployment.")
        
    st.markdown('<div class="disclaimer">Disclaimer: DeepGuard provides an AI-based estimate and should not be treated as definitive forensic evidence. Results may vary depending on compression, lighting, and manipulation techniques.</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()

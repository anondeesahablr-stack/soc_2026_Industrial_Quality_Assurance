import streamlit as st
from PIL import Image
import pandas as pd
import time
from detector import load_model, detect_defects
from report_generator import generate_report

# Page configuration for streamlit
st.set_page_config(
    page_title = "AI-powered Industrial Quality Assurance system",
    page_icon  = "!",
    layout     = "wide"
)

# Custom CSS 
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: bold;
        color: #1f4e79; text-align: center; margin-bottom: 0.2rem;
    }
    .sub-header {
        text-align: center; color: #666; margin-bottom: 2rem;
    }
    .metric-card {
        background: #f0f4f8; border-radius: 10px;
        padding: 1rem; text-align: center;
    }
    .defect-critical { color: #c0392b; font-weight: bold; }
    .defect-high     { color: #e67e22; font-weight: bold; }
    .defect-medium   { color: #f39c12; font-weight: bold; }
    .defect-low      { color: #27ae60; font-weight: bold; }
</style>
""", unsafe_allow_html=True) # required to pass raw CSS to Streamlit 

# Header
st.markdown('<p class="main-header">AI-powered Industrial Quality Assurance system</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-header">YOLOv8 Defect Detection and AI Inspection Report</p>',
            unsafe_allow_html=True)
st.divider() # draws a horizontal line to visually separate sections

# Load Model
# Without caching, Streamlit would reload the YOLOv8 model from disk every single time, making the app very slow 
# With caching, it loads once and stays in memory for the entire session
@st.cache_resource
def get_model():
    return load_model("models/best.pt")

model = get_model()

# Sidebar
with st.sidebar:
    st.header("Settings")
    conf_threshold = st.slider(
        "Confidence Threshold", 0.1, 0.9, 0.25, 0.05,
        help="Lower = detect more (but more false positives). Higher = stricter."
    )
    st.divider()
    st.markdown("**Model Info**")
    st.markdown("- YOLOv8 Nano")
    st.markdown("- 6 defect classes")
    st.markdown("- NE Steel Surface Dataset")
    st.divider()
    st.markdown("**Defect Classes**")
    classes = ["Crazing", "Inclusion", "Patches",
               "Pitted Surface", "Rolled-in Scale", "Scratches"]
    for c in classes:
        st.markdown(f"• {c}")

# Upload Section 
st.subheader("Upload Steel Surface Image")
uploaded_file = st.file_uploader(
    "Upload a steel surface image for inspection",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    # Detection
    with st.spinner("Running defect detection"): # shows a loading animation while the code inside runs
        # measures how long inference took in milliseconds, displayed at the bottom of the page later
        start_time       = time.time()
        annotated_image, defects = detect_defects(model, image, conf_threshold)
        inference_time   = round((time.time() - start_time) * 1000, 1)

    # Image Display
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)   
    with col2:
        st.subheader("Detected Defects")
        st.image(annotated_image, use_container_width=True)

    st.divider()

    # Defects Table
    if defects:
        st.subheader("Defect Details")

        df = pd.DataFrame([{
            "Defect Type" : d['class'].replace('_', ' ').title(),
            "Confidence"  : f"{d['confidence']}%",
            "Severity"    : d['severity'],
            "Area (px²)"  : d['area_px'],
            "Location"    : f"({d['bbox']['x1']}, {d['bbox']['y1']}) → ({d['bbox']['x2']}, {d['bbox']['y2']})"
        } for d in defects])

        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.success("No defects detected! Surface meets quality standards.")

    st.divider()

    # LLM Report
    st.subheader("Inspection Report")

    if st.button("Generate Inspection Report", type="primary", use_container_width=True):
        with st.spinner("Llama 3.2 is generating your inspection report..."):
            try:
                report = generate_report(defects, uploaded_file.name)
                st.markdown(report)

            except Exception as e:
                st.error(f"Could not connect to Ollama: {e}")
                st.info("Make sure Ollama is running: open a terminal and run `ollama serve`")

    st.caption(f"⚡ Inference time: {inference_time}ms")

else:
    # Placeholder when no image uploaded
    st.info("Upload a steel surface image above to begin inspection.")
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64

# Set page config first
st.set_page_config(
    page_title="Insurance AI Provider",
    page_icon="🏥",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- 1. Session State for Dark Mode ---
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# --- 2. Theme & CSS Logic ---
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def load_css(is_dark):
    # Define Colors based on mode
    if is_dark:
        # Dark Theme Palette
        text_color = "#ecf0f1"
        card_bg = "rgba(30, 40, 50, 0.90)" # Dark semi-transparent
        overlay_color = "rgba(0, 0, 0, 0.75)" # Darker overlay
        accent_color = "#00e5ff" # Neon Teal
        button_bg = "#2980b9"
        button_hover = "#3498db"
        result_box_bg = "#2c3e50"
    else:
        # Light Theme Palette (Medical Clean)
        text_color = "#2c3e50"
        card_bg = "rgba(255, 255, 255, 0.95)" # White semi-transparent
        overlay_color = "rgba(255, 255, 255, 0.5)" # Light overlay
        accent_color = "#2c3e50" # Dark Blue
        button_bg = "#4CAF50" # Medical Green
        button_hover = "#45a049"
        result_box_bg = "#2c3e50"

    # Try to load background image
    bg_css = ""
    try:
        bin_str = get_base64_of_bin_file("medical-background.jpg")
        bg_css = f"""
            background-image: linear-gradient({overlay_color}, {overlay_color}), url("data:image/jpg;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        """
    except FileNotFoundError:
        # Fallback Gradient
        scale = "linear-gradient(135deg, #2c3e50 0%, #000000 100%)" if is_dark else "linear-gradient(135deg, #E0F7FA 0%, #80DEEA 100%)"
        bg_css = f"""
            background: {scale};
            background-attachment: fixed;
        """

    return f"""
    <style>
        /* Main App Background */
        .stApp {{
            {bg_css}
        }}
        
        /* General Text Color */
        .stMarkdown, .stText, h1, h2, h3, label, .stSlider > div > div > div > div {{
            color: {text_color} !important;
        }}
        
        /* Card Containers */
        .css-1r6slb0, .css-12oz5g7, [data-testid="stForm"] {{
            background-color: {card_bg};
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 10px 25px rgba(0,0,0,0.1);
            backdrop-filter: blur(5px);
        }}
        
        /* Headers */
        h1 {{
            font-family: 'Helvetica Neue', sans-serif;
            font-weight: 700;
            text-align: center;
            margin-bottom: 0.5rem;
            text-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        h3 {{
            font-weight: 400;
            text-align: center;
            margin-bottom: 2rem;
            opacity: 0.9;
        }}
        
        /* Buttons */
        .stButton>button {{
            width: 100%;
            background-color: {button_bg};
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 0.8rem;
            font-weight: 600;
            transition: all 0.3s ease;
        }}
        
        .stButton>button:hover {{
            background-color: {button_hover};
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }}
        
        /* Custom Toggle Switch Styling (Hack) */
        .stToggle label {{
            color: {text_color} !important;
            font-weight: bold;
        }}
        
        /* Force Radio Button Text to Black (User Request) */
        .stRadio label p {{
            color: black !important;
            font-weight: 600;
        }}

        /* Result Box */
        .result-box {{
            background-color: {result_box_bg};
            padding: 1.5rem;
            border-radius: 10px;
            text-align: center;
            margin-top: 2rem;
            animation: fadeIn 0.5s ease-in;
            border: 1px solid {accent_color};
            box-shadow: 0 0 20px rgba(0,0,0,0.3);
        }}
        
        .result-box h2 {{
             color: #ecf0f1 !important;
             margin: 0;
        }}
        .result-box h1 {{
             color: #2ecc71 !important;
             margin: 1rem 0;
             font-size: 3.5rem;
        }}
        .result-box p {{
             color: #bdc3c7 !important;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
    </style>
    """

# Inject CSS based on state
st.markdown(load_css(st.session_state.dark_mode), unsafe_allow_html=True)

# Load Models (Cached)
@st.cache_resource
def load_artifacts():
    scaler = joblib.load("scaler.joblib")
    le_gender = joblib.load("label_encoder_gender.pkl")
    le_smoker = joblib.load("label_encoder_smoker.pkl")
    le_diabetic = joblib.load("label_encoder_diabetic.pkl")
    model = joblib.load("best_model.pkl")
    return scaler, le_gender, le_smoker, le_diabetic, model

import streamlit.components.v1 as components

def main():
    # Top Bar - Toggle
    col_t1, col_t2 = st.columns([6, 1])
    with col_t2:
        st.toggle("Dark Mode", key="toggle_dark", value=st.session_state.dark_mode, on_change=toggle_theme)

    try:
        scaler, le_gender, le_smoker, le_diabetic, model = load_artifacts()
    except FileNotFoundError as e:
        st.error("Error loading models. Please check files.")
        return

    st.title("🏥 Health Insurance Cost Estimator")
    st.markdown("### AI-Powered Precision Pricing")

    with st.form("main_form"):
        col1, col2 = st.columns([1, 1], gap="medium")
        
        with col1:
            st.subheader("Personal Details")
            age = st.slider("Age", 18, 100, 30)
            gender = st.radio("Gender", options=le_gender.classes_, horizontal=True)
            children = st.number_input("Children", 0, 10, 0)
            
        with col2:
            st.subheader("Health Metrics")
            bmi = st.slider("BMI", 10.0, 60.0, 25.0, 0.1)
            bloodpressure = st.slider("Blood Pressure", 60, 200, 120)
            
            st.subheader("Lifestyle")
            c1, c2 = st.columns(2)
            with c1:
                smoker = st.selectbox("Smoker?", options=le_smoker.classes_)
            with c2:
                diabetic = st.selectbox("Diabetic?", options=le_diabetic.classes_)

        st.markdown("---")
        submitted = st.form_submit_button("Calculate Estimated Premium")

    if submitted:
        input_dict = {
            "age": [age], "bmi": [bmi], "children": [children],
            "bloodpressure": [bloodpressure], "gender": [gender],
            "diabetic": [diabetic], "smoker": [smoker]
        }
        input_df = pd.DataFrame(input_dict)

        # Preprocessing (Same Robust Logic)
        input_df["gender"] = le_gender.transform(input_df["gender"])
        input_df["diabetic"] = le_diabetic.transform(input_df["diabetic"])
        input_df["smoker"] = le_smoker.transform(input_df["smoker"])
        
        num_features = ["age", "bmi", "bloodpressure"]
        input_df[num_features] = scaler.transform(input_df[num_features])
        
        for cat in ["gender", "diabetic", "smoker"]:
            input_df[cat] = input_df[cat].astype(int)

        model_cols = ["age", "gender", "bmi", "bloodpressure", "diabetic", "children", "smoker"]
        final_input = input_df[model_cols]

        try:
            prediction = model.predict(final_input)[0]
            st.markdown(f"""
            <div class="result-box" id="result_anchor">
                <h2>Estimated Yearly Premium</h2>
                <h1>${prediction:,.2f}</h1>
                <p>Based on your provided health profile.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Auto-scroll Logic
            components.html(
                """
                <script>
                    window.parent.document.querySelector('.result-box').scrollIntoView({behavior: 'smooth'});
                </script>
                """,
                height=0
            )
            
        except Exception as e:
            st.error(f"Prediction Error: {e}")

if __name__ == "__main__":
    main()

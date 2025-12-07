import streamlit as st
import numpy as np
import tensorflow as tf
import re
import os
from tensorflow.keras import layers

# Page configuration
st.set_page_config(
    page_title="SQL Injection & XSS Detection",
    page_icon="🛡️",
    layout="wide"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .safe {
        background-color: #d4edda;
        border: 2px solid #28a745;
    }
    .sql-injection {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
    }
    .xss {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
    }
    </style>
""", unsafe_allow_html=True)

# Preprocessing functions (from notebook)
def remove_comment(text):
    """Remove comments from text"""
    # Use raw string (r'') to fix SyntaxWarning for escape sequences
    text = re.sub(r'//.*?\n|/\*.*?\*/', '', text, flags=re.S)
    text = text.split('--')[0]+"--"
    if '\'' in text:
        removeTarget = text.split('\'')[0]
        text = text.replace(removeTarget, "")
    return text

def data2char_index(X, max_len, is_remove_comment=False):
    """Convert text to character indices"""
    alphabet = " abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    result = [] 
    for data in X:
        mat = []
        if is_remove_comment == True: 
            data = remove_comment(data)
        for ch in data:
            ch = ch.lower()
            if ch not in alphabet:
                continue
            mat.append(alphabet.index(ch))
        result.append(mat)
    X_char = tf.keras.preprocessing.sequence.pad_sequences(
        np.array(result, dtype=object), 
        padding='post',
        truncating='post', 
        maxlen=max_len
    )
    return X_char

def data_to_symbol_tag(X, max_len, is_remove_comment=False):
    """Convert text to symbol tags"""
    symbol = " -,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    result = [] 
    for data in X:
        mat = []
        if is_remove_comment == True: 
            data = remove_comment(data)
        for ch in data:
            ch = ch.lower()
            if ch not in symbol:
                mat.append(0)
            else:
                mat.append(symbol.index(ch))
        result.append(mat)
    X_char = tf.keras.preprocessing.sequence.pad_sequences(
        np.array(result, dtype=object), 
        padding='post',
        truncating='post', 
        maxlen=max_len
    )
    return X_char

# Load model function
@st.cache_resource
def load_model():
    """Load the trained model - tries multiple methods"""
    import warnings
    warnings.filterwarnings('ignore')
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths
    model_keras_path = os.path.join(script_dir, 'sqli_xss_model_latest.keras')
    
    # =========================================================================
    # METHOD 1: Load .keras model (NEW STANDARD)
    # =========================================================================
    if os.path.exists(model_keras_path):
        try:
            model = tf.keras.models.load_model(model_keras_path)
            st.session_state.model_loaded_from = ".keras file (latest format) ✓"
            return model
        except Exception as e:
            st.session_state.keras_error = f".keras load failed: {str(e)}"
    
    return None

# Initialize model
script_dir = os.path.dirname(os.path.abspath(__file__))
model_keras_path = os.path.join(script_dir, 'sqli_xss_model_latest.keras')

model = load_model()

# Display errors if model loading failed
if model is None:
    st.error("❌ Could not load model from any available format.")
    
    # Check which files exist
    files_status = []
    if os.path.exists(model_keras_path):
        files_status.append(f"✅ `sqli_xss_model_latest.keras` exists at: {model_keras_path}")
    else:
        files_status.append(f"❌ `sqli_xss_model_latest.keras` not found at: {model_keras_path}")
    
    st.info("**File Status:**\n" + "\n".join(files_status))
    
    # Version info
    st.info(f"TensorFlow Version: {tf.__version__}")

    # Show detailed error if available
    if 'keras_error' in st.session_state:
        with st.expander("View Detailed Error Information", expanded=True):
            st.code(st.session_state.keras_error)
    
    # Show current working directory and file paths for debugging
    st.info(f"**Current working directory:** `{os.getcwd()}`\n**Script directory:** `{script_dir}`")


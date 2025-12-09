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

# Main UI
st.markdown('<div class="main-header">🛡️ SQL Injection & XSS Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Deep Learning Security Scanner</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    if model is not None:
        st.success("✅ Model Loaded Successfully")
        
        # Show loading method
        if 'model_loaded_from' in st.session_state:
            st.info(f"**Loaded from:** {st.session_state.model_loaded_from}")
        
        st.markdown("---")
        st.info("**Model Architecture:**\n- Dual Input (Text + Symbols)\n- CNN + Bidirectional GRU\n- Cross-Attention Mechanism\n- 3-Class Classification")
    else:
        st.error("❌ Model Not Loaded")
    
    st.markdown("---")
    st.header("ℹ️ About")
    st.markdown("""
    This system detects:
    - **SQL Injection** attacks
    - **XSS** (Cross-Site Scripting) attacks
    - **Normal** safe inputs
    
    Enter a payload or query string to analyze.
    """)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.header("🔍 Input Payload")
    
    # Example payloads
    st.subheader("📝 Example Payloads")
    example_col1, example_col2, example_col3 = st.columns(3)
    
    with example_col1:
        if st.button("SQL Injection Example"):
            st.session_state.payload_input = "admin' OR '1'='1'--"
    
    with example_col2:
        if st.button("XSS Example"):
            st.session_state.payload_input = "<script>alert('XSS')</script>"
    
    with example_col3:
        if st.button("Normal Example"):
            st.session_state.payload_input = "Hassan Rashid"
    
    # Text input with key for proper state management
    input_text = st.text_area(
        "Enter the payload or query string to analyze:",
        value=st.session_state.get('payload_input', ''),
        height=200,
        placeholder="Example: SELECT * FROM users WHERE id = 1 OR '1'='1'",
        key="payload_input"
    )
    
    # Analyze button
    analyze_button = st.button("🚀 Analyze Payload", type="primary", use_container_width=True)

with col2:
    st.header("📈 Results")
    
    if analyze_button and input_text.strip():
        if model is None:
            st.error("Model not loaded. Please check the model file.")
        else:
            with st.spinner("Analyzing payload..."):
                # Preprocess input
                text_index = data2char_index([input_text], max_len=1000)
                symbol_index = data_to_symbol_tag([input_text], max_len=1000)
                
                # Make prediction
                prediction = model.predict([text_index, symbol_index], verbose=0)
                
                # Get class probabilities
                # Order matches notebook: [SQLInjection, XSS, Normal]
                classes = ['SQL Injection', 'XSS', 'Normal']
                probabilities = prediction[0]
                
                # Debug: Show raw prediction values (can be removed later)
                with st.expander("🔍 Debug: Raw Prediction Values"):
                    st.write(f"**Raw probabilities:** {probabilities}")
                    st.write(f"**Sum:** {np.sum(probabilities):.4f} (should be ~1.0)")
                    st.write(f"**Class indices:** 0=SQL Injection, 1=XSS, 2=Normal")
                    
                    st.write("---")
                    st.write("**Input Analysis:**")
                    st.write(f"Text Input Shape: {text_index.shape}")
                    st.write(f"Symbol Input Shape: {symbol_index.shape}")
                    st.write(f"Non-zero text chars: {np.count_nonzero(text_index)}")
                    st.write(f"Non-zero symbol chars: {np.count_nonzero(symbol_index)}")
                    
                    for idx, (cls, prob) in enumerate(zip(classes, probabilities)):
                        st.write(f"  - Index {idx} ({cls}): {float(prob):.6f}")
                
                predicted_class_idx = np.argmax(probabilities)
                predicted_class = classes[predicted_class_idx]
                confidence = float(probabilities[predicted_class_idx]) * 100
                
                # Display results
                st.markdown("### Prediction Result")
                
                # Color-coded result box
                if predicted_class == 'Normal':
                    st.markdown(f"""
                    <div class="prediction-box safe">
                        <h2 style="color: #28a745; margin: 0;">✅ {predicted_class}</h2>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0;"><strong>{confidence:.2f}%</strong> confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                elif predicted_class == 'SQL Injection':
                    st.markdown(f"""
                    <div class="prediction-box sql-injection">
                        <h2 style="color: #dc3545; margin: 0;">⚠️ {predicted_class}</h2>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0;"><strong>{confidence:.2f}%</strong> confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:  # XSS
                    st.markdown(f"""
                    <div class="prediction-box xss">
                        <h2 style="color: #ffc107; margin: 0;">⚠️ {predicted_class}</h2>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0;"><strong>{confidence:.2f}%</strong> confidence</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability breakdown
                st.markdown("### Probability Breakdown")
                for i, (cls, prob) in enumerate(zip(classes, probabilities)):
                    # Convert numpy float32 to Python float for st.progress
                    prob_float = float(prob)
                    st.progress(prob_float, text=f"{cls}: {prob_float*100:.2f}%")
                
                # Detailed probabilities
                with st.expander("View Detailed Probabilities"):
                    for cls, prob in zip(classes, probabilities):
                        st.write(f"**{cls}**: {prob*100:.4f}%")
    
    elif analyze_button and not input_text.strip():
        st.warning("Please enter a payload to analyze.")
    else:
        st.info("👆 Enter a payload and click 'Analyze Payload' to get started.")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>Built with Streamlit & TensorFlow | Security Detection System</div>",
    unsafe_allow_html=True
)


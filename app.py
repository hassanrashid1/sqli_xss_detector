import os
# Set environment variable to use Keras 2 (tf-keras) with TensorFlow 2.16+
# MUST be set before importing tensorflow or streamlit
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import streamlit as st
import numpy as np
import tensorflow as tf
import re
from tensorflow.keras import layers
from tensorflow.keras.layers import GRU, Bidirectional, MultiHeadAttention, Conv1D, MaxPooling1D, Embedding, Dense, Flatten, Add

# Define GRUWrapper at module level for proper deserialization
class GRUWrapper(tf.keras.layers.GRU):
    """Custom GRU wrapper that handles compatibility issues"""
    def __init__(self, *args, **kwargs):
        # Remove time_major if present (compatibility fix)
        kwargs.pop('time_major', None)
        super().__init__(*args, **kwargs)
    
    def get_config(self):
        config = super().get_config()
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

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

# Model structure (from notebook)
def model_struct(max_len):
    """Define the model architecture"""
    pool_siz = 10
    num_heads = 3
    
    # Text input layer
    input_text = tf.keras.layers.Input(shape=(max_len,))
    # Embedding layer
    embed1 = tf.keras.layers.Embedding(
        input_dim=70, 
        output_dim=105, 
        input_length=max_len, 
        trainable=False
    )(input_text)
    cnn1 = tf.keras.layers.Conv1D(32, 3, padding='same', strides=1, activation='relu')(embed1)
    cnn1 = tf.keras.layers.MaxPooling1D(pool_size=pool_siz)(cnn1)
    # GRU - Match EXACT original structure from notebook
    # Note: go_backwards=True was in original, but causes issues in newer TF
    # We'll handle this by trying both structures
    try:
        # Try original structure first (with go_backwards=True)
        GRU0 = layers.Bidirectional(
            tf.keras.layers.GRU(32, return_sequences=True, go_backwards=True)
        )(cnn1)
        if 'gru_status' not in st.session_state:
            st.session_state.gru_status = "✅ GRU initialized with go_backwards=True"
    except Exception as e:
        # Fallback for newer TF versions
        if 'gru_status' not in st.session_state:
            st.session_state.gru_status = f"⚠️ GRU fallback (go_backwards=False). Error: {str(e)}"
        GRU0 = layers.Bidirectional(
            tf.keras.layers.GRU(32, return_sequences=True)
        )(cnn1)
    
    # Symbol input layer
    input_symbol = tf.keras.layers.Input(shape=(max_len,))
    embed2 = tf.keras.layers.Embedding(
        input_dim=34, 
        output_dim=51, 
        input_length=max_len, 
        trainable=False
    )(input_symbol)
    cnn1s = tf.keras.layers.Conv1D(32, 3, padding='same', strides=1, activation='relu')(embed2)
    cnn1s = tf.keras.layers.MaxPooling1D(pool_size=pool_siz)(cnn1s)
    # GRU - Match EXACT original structure
    try:
        GRU0s = layers.Bidirectional(
            tf.keras.layers.GRU(32, return_sequences=True, go_backwards=True)
        )(cnn1s)
    except Exception as e:
        # Fallback for newer TF versions
        GRU0s = layers.Bidirectional(
            tf.keras.layers.GRU(32, return_sequences=True)
        )(cnn1s)
    
    # Cross Attention
    CrossAT1 = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=1)(GRU0, GRU0s)
    
    # Connect the outputs
    combined = tf.keras.layers.add([GRU0 + CrossAT1, GRU0s + CrossAT1])
    
    flat = tf.keras.layers.Flatten()(combined)
    dnn1 = tf.keras.layers.Dense(3, activation="softmax")(flat)
    
    # Output model
    model = tf.keras.Model(inputs=[input_text, input_symbol], outputs=dnn1)
    return model

# Load model function
@st.cache_resource
def load_model():
    """Load the trained model - tries multiple methods"""
    import warnings
    import traceback
    import h5py
    warnings.filterwarnings('ignore')
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define file paths
    weights_path = os.path.join(script_dir, 'sqli_xss_model_weights.h5')
    model_h5_path = os.path.join(script_dir, 'sqli_xss_detection_model.h5')
    model_dir_path = os.path.join(script_dir, 'sqli_xss_detection_model')
    
    # Custom objects for loading - use module-level GRUWrapper
    custom_objects = {
        'GRU': tf.keras.layers.GRU,
        'GRUWrapper': GRUWrapper,
        'Bidirectional': tf.keras.layers.Bidirectional,
        'MultiHeadAttention': tf.keras.layers.MultiHeadAttention,
    }
    
    # =========================================================================
    # METHOD 1: Load complete .h5 model (BEST - has exact structure)
    # =========================================================================
    if os.path.exists(model_h5_path):
        try:
            # Try standard loading first
            model = tf.keras.models.load_model(
                model_h5_path, 
                custom_objects=custom_objects,
                compile=False
            )
            model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
            st.session_state.model_loaded_from = ".h5 file (complete model) ✓"
            return model
        except Exception as e1:
            try:
                # Try with safe_mode=False
                model = tf.keras.models.load_model(
                    model_h5_path, 
                    custom_objects=custom_objects,
                    compile=False,
                    safe_mode=False
                )
                model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
                st.session_state.model_loaded_from = ".h5 file (safe_mode=False) ✓"
                return model
            except Exception as e2:
                st.session_state.h5_error = f".h5 load failed: {str(e2)}"
    
    # =========================================================================
    # METHOD 2: Load weights into recreated structure
    # =========================================================================
    if os.path.exists(weights_path):
        # Check TensorFlow version to decide which structure to use
        tf_version = tf.__version__
        st.session_state.tf_version = tf_version
        
        try:
            # First, try to inspect the weights file to understand its structure
            with h5py.File(weights_path, 'r') as f:
                weight_names = []
                def get_names(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        weight_names.append(name)
                f.visititems(get_names)
                st.session_state.weight_count = len(weight_names)
        except Exception as e:
            st.session_state.weight_count = "unknown"
        
        # Try creating model and loading weights
        try:
            # Create model structure (will use fallback if go_backwards fails)
            model = model_struct(max_len=1000)
            
            # Build model
            dummy_text = np.zeros((1, 1000), dtype=np.int32)
            dummy_symbol = np.zeros((1, 1000), dtype=np.int32)
            _ = model([dummy_text, dummy_symbol])
            
            # Try loading weights
            try:
                model.load_weights(weights_path, by_name=False)
                st.session_state.weights_loaded_method = "strict (by_name=False)"
            except:
                try:
                    model.load_weights(weights_path, by_name=True)
                    st.session_state.weights_loaded_method = "by_name=True"
                except:
                    model.load_weights(weights_path, skip_mismatch=True)
                    st.session_state.weights_loaded_method = "skip_mismatch (may be incorrect!)"
                    st.session_state.weights_warning = "⚠️ Structure mismatch detected!"
            
            model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
            st.session_state.model_loaded_from = "weights file (recreated structure)"
            
            # Test if model produces varied outputs
            test_results = []
            test_inputs = [
                "admin' OR '1'='1'--",  # SQL Injection
                "<script>alert('x')</script>",  # XSS
                "SELECT id FROM users WHERE id = 5"  # Normal
            ]
            for test_input in test_inputs:
                text_idx = data2char_index([test_input], max_len=1000)
                symbol_idx = data_to_symbol_tag([test_input], max_len=1000)
                pred = model.predict([text_idx, symbol_idx], verbose=0)
                test_results.append(np.argmax(pred[0]))
            
            # Check if all predictions are the same (bad sign)
            if len(set(test_results)) == 1:
                st.session_state.model_warning = f"⚠️ Model predicts same class ({test_results[0]}) for all test inputs - may not be working correctly!"
            else:
                st.session_state.model_status = f"✓ Model produces varied predictions: {test_results}"
            
            return model
            
        except Exception as e:
            st.session_state.error_details = f"Weights loading failed: {str(e)}\n{traceback.format_exc()}"
    
    # =========================================================================
    # METHOD 3: SavedModel format
    # =========================================================================
    if os.path.exists(model_dir_path):
        try:
            model = tf.keras.models.load_model(model_dir_path, compile=False)
            model.compile("adam", "categorical_crossentropy", metrics=["accuracy"])
            st.session_state.model_loaded_from = "SavedModel format ✓"
            return model
        except Exception as e:
            pass
    
    return None

# Initialize model
script_dir = os.path.dirname(os.path.abspath(__file__))
weights_path = os.path.join(script_dir, 'sqli_xss_model_weights.h5')
model_h5_path = os.path.join(script_dir, 'sqli_xss_detection_model.h5')
model_dir_path = os.path.join(script_dir, 'sqli_xss_detection_model')

model = load_model()

# Display errors if model loading failed
if model is None:
    st.error("❌ Could not load model from any available format.")
    
    # Check which files exist
    files_status = []
    if os.path.exists(weights_path):
        files_status.append(f"✅ `sqli_xss_model_weights.h5` exists at: {weights_path}")
    else:
        files_status.append(f"❌ `sqli_xss_model_weights.h5` not found at: {weights_path}")
    
    if os.path.exists(model_h5_path):
        files_status.append(f"✅ `sqli_xss_detection_model.h5` exists at: {model_h5_path}")
    else:
        files_status.append(f"❌ `sqli_xss_detection_model.h5` not found at: {model_h5_path}")
    
    if os.path.exists(model_dir_path):
        files_status.append(f"✅ `sqli_xss_detection_model/` directory exists at: {model_dir_path}")
    else:
        files_status.append(f"❌ `sqli_xss_detection_model/` directory not found at: {model_dir_path}")
    
    st.info("**File Status:**\n" + "\n".join(files_status))
    
    # Version info
    st.info(f"TensorFlow Version: {tf.__version__}")
    if os.environ.get('TF_USE_LEGACY_KERAS'):
        st.success("Legacy Keras mode enabled (TF_USE_LEGACY_KERAS=1)")

    # Show detailed error if available
    if 'error_details' in st.session_state:
        with st.expander("View Detailed Error Information", expanded=True):
            st.code(st.session_state.error_details)
    
    # Show current working directory and file paths for debugging
    st.info(f"**Current working directory:** `{os.getcwd()}`\n**Script directory:** `{script_dir}`")
    
    st.info("""
    **Troubleshooting:**
    - If weights file exists but fails to load, the model structure might not match
    - Check TensorFlow version compatibility (try: `pip install tensorflow==2.13.0`)
    - Ensure the model structure matches the saved weights exactly
    - Try running the app from the directory containing the model files
    """)

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
        if 'weights_loaded_method' in st.session_state:
            st.info(f"**Weights method:** {st.session_state.weights_loaded_method}")
        if 'tf_version' in st.session_state:
            st.info(f"**TensorFlow:** {st.session_state.tf_version}")
        
        # Show warnings
        if 'gru_status' in st.session_state:
            if "fallback" in st.session_state.gru_status:
                st.warning(st.session_state.gru_status)
            else:
                st.info(st.session_state.gru_status)
                
        if 'weights_warning' in st.session_state:
            st.warning(st.session_state.weights_warning)
        if 'model_warning' in st.session_state:
            st.error(st.session_state.model_warning)
        if 'model_status' in st.session_state:
            st.success(st.session_state.model_status)
        if 'h5_error' in st.session_state:
            with st.expander("⚠️ .h5 loading failed"):
                st.code(st.session_state.h5_error)
        
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
            st.session_state.payload_input = "SELECT name FROM users WHERE id = 123"
    
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


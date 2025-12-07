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

    pool_siz = 10
    num_heads = 3

    # ----- TEXT INPUT BRANCH -----
    input_text = layers.Input(shape=(max_len,), name="text_in")
    embed1 = layers.Embedding(input_dim=70, output_dim=105, trainable=False)(input_text)
    cnn1 = layers.Conv1D(32, 3, padding="same", activation="relu")(embed1)
    cnn1 = layers.MaxPooling1D(pool_size=pool_siz)(cnn1)
    gru_text = layers.Bidirectional(layers.GRU(32, return_sequences=True))(cnn1)

    # ----- SYMBOL INPUT BRANCH -----
    input_symbol = layers.Input(shape=(max_len,), name="symbol_in")
    embed2 = layers.Embedding(input_dim=34, output_dim=51, trainable=False)(input_symbol)
    cnn2 = layers.Conv1D(32, 3, padding="same", activation="relu")(embed2)
    cnn2 = layers.MaxPooling1D(pool_size=pool_siz)(cnn2)
    gru_symbol = layers.Bidirectional(layers.GRU(32, return_sequences=True))(cnn2)

    # ----- CROSS ATTENTION -----
    # Query = text GRU, Keys/Values = symbol GRU
    cross_att = layers.MultiHeadAttention(
        num_heads=num_heads,
        key_dim=32,     # must match last dim of GRU
        dropout=0.0
    )(gru_text, gru_symbol)

    # Combine both sequences + cross attention
    combined = layers.Add()([gru_text, gru_symbol, cross_att])

    # ----- CLASSIFICATION -----
    flat = layers.Flatten()(combined)
    output = layers.Dense(3, activation="softmax")(flat)

    return tf.keras.Model(inputs=[input_text, input_symbol], outputs=output)

# Load model function
@st.cache_resource

def load_model():
    try:
        model = tf.keras.models.load_model('model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Initialize model
script_dir = os.path.dirname(os.path.abspath(__file__))


model = load_model()

# Display errors if model loading failed
if model is None:
    st.error("❌ Could not load model from any available format.")
    
    
# Main UI
st.markdown('<div class="main-header">🛡️ SQL Injection & XSS Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Deep Learning Security Scanner</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("📊 Model Information")
    if model is not None:
        st.success("✅ Model Loaded Successfully")
      
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


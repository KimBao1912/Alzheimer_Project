import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
@st.cache_resource
def load_trained_model():
    """Load model với caching để tránh load lại nhiều lần"""
    return load_model('densenet_final_model.keras')

model = load_trained_model()

# Define the image size for model input (phù hợp với training config)
IMG_SIZE = (224, 224)

# Add custom CSS for aesthetics
st.markdown(
    """
    <style>
    .title {
        margin-top:0px;
        color: #FF5733;
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        margin-bottom: 10px;
    }
    
    .text {
        color: #EFA18A;
        font-size: 20px;
        font-weight: italic;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .prediction {
        color: #FF5733;
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
        text-align: center;
    }
    
    .confidence {
        color: #FF5600;
        font-size: 18px;
        margin-bottom: 20px;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.set_option('deprecation.showPyplotGlobalUse', False)

# Display the title
st.markdown("<h1 class='title'>Alzheimer's Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<h1 class='text'>Alzheimer's Disease Prediction is a web application that utilizes a pre-trained DenseNet121 deep learning model to predict the presence of Alzheimer's disease based on uploaded brain MRI images. Users can upload an image through the sidebar and the app will process the image using the trained model.</h1>", unsafe_allow_html=True)

st.sidebar.title("Upload Image")
st.sidebar.markdown("Please upload a brain MRI image.")

def preprocess_image(image):
    """
    Preprocess image theo đúng cách model đã được train:
    - Resize to 224x224 (theo IMG_SIZE trong training)
    - Convert to RGB (3 channels) nếu cần
    - Rescale pixel values to [0, 1] (theo rescaling layer trong training: 1./255)
    """
    # Convert to RGB nếu image không phải RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Resize ảnh về kích thước model yêu cầu (224x224)
    img = image.resize(IMG_SIZE)
    
    # Chuyển sang numpy array
    img_array = np.array(img)
    
    # Rescale pixel values về [0, 1] - ĐÚNG THEO TRAINING
    img_array = img_array.astype('float32') / 255.0
    
    # Thêm batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def predict(image):
    """
    Make prediction on the input image
    Returns: predicted class index, confidence score, and all predictions
    """
    img_array = preprocess_image(image)
    prediction = model.predict(img_array, verbose=0)
    
    predicted_idx = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    
    return predicted_idx, confidence, prediction[0]

# Display the file uploader
uploaded_file = st.sidebar.file_uploader(label="", type=['jpg', 'jpeg', 'png'])

# Make predictions and display the result
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Hiển thị ảnh đã upload
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(image, caption='Original Image', use_column_width=True)
    
    with col2:
        # Hiển thị ảnh sau preprocessing
        processed_img = image.convert('RGB').resize(IMG_SIZE)
        st.image(processed_img, caption='Processed Image (224x224 RGB)', use_column_width=True)
    
    # Thực hiện prediction
    with st.spinner('Analyzing image...'):
        predicted_idx, confidence, all_predictions = predict(image)
    
    # Class labels - ĐÚNG THEO THỨ TỰ TRAINING
    # Theo notebook: ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    class_labels = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']
    predicted_label = class_labels[predicted_idx]
    
    # Hiển thị kết quả
    st.markdown(f"<p class='prediction'>Prediction: {predicted_label}</p>", unsafe_allow_html=True)
    st.markdown(f"<p class='confidence'>Confidence: {confidence:.2f}%</p>", unsafe_allow_html=True)
    
    # Vẽ bar chart cho predictions
    st.write("### Prediction Probabilities")
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ['#FF6B6B' if i == predicted_idx else '#4ECDC4' for i in range(len(class_labels))]
    bars = ax.bar(class_labels, all_predictions * 100, color=colors)
    ax.set_ylabel('Probability (%)', fontsize=12)
    ax.set_xlabel('Classification', fontsize=12)
    ax.set_title('Prediction Distribution Across All Classes', fontsize=14, fontweight='bold')
    plt.xticks(rotation=15, ha='right')
    
    # Thêm giá trị lên mỗi bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # Hiển thị chi tiết predictions
    with st.expander("View Detailed Predictions"):
        for i, label in enumerate(class_labels):
            st.write(f"**{label}**: {all_predictions[i]*100:.2f}%")
    
    # Warning nếu confidence thấp
    if confidence < 50:
        st.warning("⚠️ Low confidence prediction. Please verify with a medical professional.")
    
    # Disclaimer
    st.info("ℹ️ **Disclaimer**: This tool is for educational purposes only and should not be used as a substitute for professional medical diagnosis.")

else:
    st.sidebar.write("Please upload an image to begin analysis.")
    
    # Hướng dẫn sử dụng
    st.write("## How to Use")
    st.write("""
    1. Click on **'Browse files'** in the sidebar
    2. Select a brain MRI image (JPG, JPEG, or PNG format)
    3. Wait for the analysis to complete
    4. View the prediction results and probability distribution
    """)
    
    st.write("## Classification Categories")
    st.write("""
    - **Non Demented**: No signs of dementia
    - **Very Mild Demented**: Very early stage of dementia
    - **Mild Demented**: Mild cognitive impairment
    - **Moderate Demented**: Moderate stage of Alzheimer's disease
    """)
    
    st.write("## Model Information")
    st.write("""
    - **Architecture**: DenseNet121 (Transfer Learning)
    - **Input Size**: 224x224 RGB images
    - **Training Accuracy**: ~97% on test set
    - **Preprocessing**: Images are automatically resized and normalized to [0, 1]
    """)

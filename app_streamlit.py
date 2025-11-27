import streamlit as st
import requests
import pandas as pd
import io

API_URL = st.secrets["API_URL"]

st.set_page_config(page_title="Iris Classifier", page_icon="⚜️", layout="wide")

# CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 3.5rem;
        font-weight: 600;
        color: #1e3a8a;
        margin-bottom: 0.5rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #64748b;
    }
    .metric-card {
        background-color: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
    }
    .stButton>button {
        background-color: #1e40af;
        color: white;
        font-weight: 500;
        border-radius: 6px;
        padding: 0.5rem 2rem;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1e3a8a;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.markdown('<p class="main-title">Iris Species Classifier</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">ML model for iris flower classification</p>')
st.divider()

# Model Performance Metrics
st.subheader("📊 Model Performance")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", "93.3%", help="Overall prediction accuracy")
with col2:
    st.metric("Precision", "94.2%", help="Precision score (macro avg)")
with col3:
    st.metric("Recall", "94.2%", help="Recall score (macro avg)")
with col4:
    st.metric("F1-Score", "94.2%", help="F1 score (macro avg)")

st.divider()

# Main content: Two columns
left_col, right_col = st.columns([1, 1])

# LEFT COLUMN: Single Prediction
with left_col:
    st.subheader("Single Prediction")
    st.markdown("Enter measurements for one flower:")
    
    sepal_length = st.number_input(
        "Sepal Length (cm)", 
        min_value=4.3, 
        max_value=7.9, 
        value=5.8,
        step=0.1,
        help="Range: 4.3-7.9 cm"
    )
    
    sepal_width = st.number_input(
        "Sepal Width (cm)", 
        min_value=2.0, 
        max_value=4.4, 
        value=3.0,
        step=0.1,
        help="Range: 2.0-4.4 cm"
    )
    
    petal_length = st.number_input(
        "Petal Length (cm)", 
        min_value=1.0, 
        max_value=6.9, 
        value=5.1,
        step=0.1,
        help="Range: 1.0-6.9 cm"
    )
    
    petal_width = st.number_input(
        "Petal Width (cm)", 
        min_value=0.1, 
        max_value=2.5, 
        value=1.8,
        step=0.1,
        help="Range: 0.1-2.5 cm"
    )
    
    if st.button("Predict", key="single"):
        payload = {
            "sepal_length": sepal_length,
            "sepal_width": sepal_width,
            "petal_length": petal_length,
            "petal_width": petal_width
        }
        
        try:
            response = requests.post(f"{API_URL}/predict", json=payload)
            
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                species_names = ["Setosa", "Versicolor", "Virginica"]
                
                predicted_species = species_names[prediction]
                st.success(f"**Predicted Species:** {species_names[prediction]}")
                st.image(f"images/{predicted_species.lower()}.jpg", width=200)
            else:
                st.error("Prediction failed. Please try again.")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# RIGHT COLUMN: Batch Prediction
with right_col:
    st.subheader("Batch Prediction")
    st.markdown("Upload a CSV file with multiple flowers:")
    
    # Format instructions
    with st.expander("📋 CSV Format Requirements"):
        st.markdown("""
        **Required columns (in this order):**
        - `sepal_length` (cm)
        - `sepal_width` (cm)
        - `petal_length` (cm)
        - `petal_width` (cm)
        
        **Example CSV:**
        ```
        sepal_length,sepal_width,petal_length,petal_width
        5.1,3.5,1.4,0.2
        6.7,3.0,5.2,2.3
        5.8,2.7,5.1,1.9
        ```
        
        **Valid ranges:**
        - Sepal Length: 4.3-7.9 cm
        - Sepal Width: 2.0-4.4 cm
        - Petal Length: 1.0-6.9 cm
        - Petal Width: 0.1-2.5 cm
        """)
    
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_cols = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
            if not all(col in df.columns for col in required_cols):
                st.error(f"Missing required columns. Need: {', '.join(required_cols)}")
            else:
                st.write(f"**Loaded {len(df)} samples**")
                st.dataframe(df.head(), width='stretch') 
                
                if st.button("Predict Batch", key="batch"):
                    # Prepare batch payload
                    batch_data = df[required_cols].to_dict('records')
                    payload = {"feature_list": batch_data}
                    
                    try:
                        response = requests.post(f"{API_URL}/predict-batch", json=payload) 
                        
                        if response.status_code == 200:
                            predictions = response.json()
                            species_names = ["Setosa", "Versicolor", "Virginica"]
                            df['predicted_species'] = [species_names[p] for p in predictions]
                            
                            st.success(f"✅ Predicted {len(predictions)} samples")
                            st.dataframe(df, width='stretch')
                            
                            # Download results
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "Download Results",
                                csv,
                                "predictions.csv",
                                "text/csv"
                            )
                        else:
                            st.error("Batch prediction failed.")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        except Exception as e:
            st.error(f"Error reading CSV: {str(e)}")

# Footer
st.divider()
st.caption("Built with Streamlit • FastAPI • Deployed on AWS EC2")

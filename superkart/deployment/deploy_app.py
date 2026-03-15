import os
import shutil
import pandas as pd
from huggingface_hub import HfApi, login, create_repo

# --- Setup Section --- #
hf_username = "deepakietb"  # NOTE: Replace this with your Hugging Face username
space_name = "superkart-forecast-app"  # Target Hugging Face Space name
model_repo_name = "superkart-sales-forecast-prediction"  # Model repository identifier

deployment_folder = "superkart/deployment"
app_file_path = os.path.join(deployment_folder, "app.py")
dockerfile_path = os.path.join(deployment_folder, "Dockerfile")
requirements_file_path = os.path.join(deployment_folder, "requirements.txt")

# Retrieve HF_TOKEN from environment variables
hf_token = os.environ.get('HF_TOKEN')
if hf_token is None:
    raise ValueError("HF_TOKEN not found in environment variables. Set it before running.")

# Authenticate to Hugging Face Hub
login(hf_token)

# Ensure the deployment directory exists
os.makedirs(deployment_folder, exist_ok=True)

# --- 1. Generate Streamlit application (app.py) --- #
# This app should load the ML model from Hugging Face Hub and provide a prediction interface
app_content = f"""
import streamlit as st
import pandas as pd
import mlflow
from huggingface_hub import snapshot_download
import os
import shutil

# Disable MLflow automatic run tracking to prevent unwanted logs
mlflow.set_tracking_uri("file:///dev/null")

# Hugging Face configuration for downloading the model
hf_username = "{hf_username}"
model_repo_name = "{model_repo_name}"
repo_id_model = f"{{hf_username}}/{{model_repo_name}}"

@st.cache_resource
def load_model():
    try:
        repo_path = snapshot_download(repo_id=repo_id_model)
        # Corrected model artifact path for Superkart regression model
        model_path = os.path.join(repo_path, "random_forest_regressor_model")
        loaded_model = mlflow.pyfunc.load_model(model_path)
        return loaded_model
    except Exception as e:
        st.error(f"Unable to load model from Hugging Face: {{e}}")
        return None

model = load_model()

st.title("Superkart Sales Forecast Prediction")
st.write("Estimate the expected sales value for a product-store combination.")

if model is None:
    st.stop()

# Input fields aligned with Superkart dataset columns
st.header("Product and Store Information")
product_id = st.text_input("Product ID", "FDX07")
product_weight = st.slider("Product Weight", 5.0, 25.0, 12.83)
product_sugar_content = st.selectbox("Product Sugar Content", ['Low Sugar', 'Regular', 'No Sugar'])
product_allocated_area = st.slider("Product Allocated Area", 0.0, 0.2, 0.0701, step=0.0001, format="%.4f")
product_type = st.selectbox("Product Type", ['Fruits and Vegetables','Snack Foods','Meat','Dairy','Baking Goods','Household',
     'Frozen Foods','Canned','Hard Drinks','Soft Drinks','Health and Hygiene',
     'Breakfast','Starchy Foods','Seafood','Bread','Others'])
product_mrp = st.slider("Product MRP", 50.0, 300.0, 140.00)
store_id = st.text_input("Store ID", "OUT018")
store_establishment_year = st.slider("Store Establishment Year", 1980, 2020, 2009)
store_size = st.selectbox("Store Size", ['Medium', 'High', 'Small'])
store_location_city_type = st.selectbox("Store Location City Type", ['Tier 1', 'Tier 2', 'Tier 3'])
store_type = st.selectbox("Store Type", ['Supermarket Type2','Departmental Store','Supermarket Type1','Food Mart'])

# Assemble input into a DataFrame for prediction
input_data = pd.DataFrame([{{
    'Product_Id': product_id,
    'Product_Weight': product_weight,
    'Product_Sugar_Content': product_sugar_content,
    'Product_Allocated_Area': product_allocated_area,
    'Product_Type': product_type,
    'Product_MRP': product_mrp,
    'Store_Id': store_id,
    'Store_Establishment_Year': store_establishment_year,
    'Store_Size': store_size,
    'Store_Location_City_Type': store_location_city_type,
    'Store_Type': store_type
}}])

if st.button("Predict Sales"): # Changed button text to 'Predict Sales'
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted Product Store Sales Total: {{prediction[0]:.2f}}") # Display continuous prediction
    except Exception as e:
        st.error(f"Prediction process failed: {{e}}")
"""

# Write the Streamlit app file
with open(app_file_path, "w") as f:
    f.write(app_content)
print(f"app.py for Streamlit has been written to {app_file_path}")

# --- 2. Dockerfile setup --- #
dockerfile_content = """
# Python 3.9 lightweight image
FROM python:3.9

# Set working directory inside container
WORKDIR /app

# Copy all deployment files into the container
COPY . .

# Install required Python packages
RUN pip3 install -r requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user     PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app
COPY --chown=user . $HOME/app

# Launch Streamlit app on port 8507
CMD ["streamlit", "run", "app.py", "--server.port=8507", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
"""
with open(dockerfile_path, "w") as f:
    f.write(dockerfile_content)
print(f"Dockerfile has been generated at {dockerfile_path}")

# --- 3. Requirements file --- #
requirements_content = """
pandas
scikit-learn
mlflow
huggingface_hub
datasets
streamlit
"""
with open(requirements_file_path, "w") as f:
    f.write(requirements_content)
print(f"requirements.txt created at {requirements_file_path}")

# --- 4. Create/ensure Hugging Face Space --- #
api = HfApi()
repo_id_space = f"{hf_username}/{space_name}"

try:
    # Corrected space_sdk to 'docker' as we are providing a Dockerfile
    create_repo(repo_id=repo_id_space, repo_type="space", exist_ok=True, token=hf_token, space_sdk='docker')
    print(f"Hugging Face Space '{space_name}' is ready (created if it did not exist).")
except Exception as e:
    print(f"Failed to create/check Hugging Face Space: {{e}}")

# --- 5. Upload all deployment files --- #
print(f"Starting upload of deployment files from '{deployment_folder}' to '{repo_id_space}'...")

try:
    api.upload_folder(
        folder_path=deployment_folder,
        repo_id=repo_id_space,
        repo_type="space",
        token=hf_token,
        commit_message="Deploy Streamlit app along with Dockerfile and requirements.txt"
    )
    print(f"All deployment files successfully uploaded: https://huggingface.co/spaces/{repo_id_space}")
except Exception as e:
    print(f"Error during upload of deployment files: {{e}}")

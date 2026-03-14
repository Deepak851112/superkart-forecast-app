
import os
import shutil
import pandas as pd
from huggingface_hub import HfApi, login, create_repo

# --- Configuration --- #
hf_username = "rakesh1248"  # IMPORTANT: Replace with your HF username
space_name = "wellness-tourism-app"  # Name for your Hugging Face Space
model_repo_name = "wellness-tourism-package-prediction-model" # Name of your model repo

deployment_folder = "tourism_project/deployment"
app_file_path = os.path.join(deployment_folder, "app.py")
dockerfile_path = os.path.join(deployment_folder, "Dockerfile")
requirements_file_path = os.path.join(deployment_folder, "requirements.txt")

# Get HF_TOKEN from environment variable
hf_token = os.environ.get('HF_TOKEN')
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable not set.")

# Log in to Hugging Face
login(hf_token)

# Ensure deployment folder exists
os.makedirs(deployment_folder, exist_ok=True)

# --- 1. Create Streamlit app (app.py) --- #
# The content of app.py needs to load the model from HF Model Hub
# and provide an interface for predictions.
app_content = f"""
import streamlit as st
import pandas as pd
import mlflow
from huggingface_hub import snapshot_download
import os
import shutil

# Disable MLflow tracking to avoid creating new runs in the deployed app
mlflow.set_tracking_uri("file:///dev/null")

# Define Hugging Face details for model download
hf_username = "{hf_username}"
model_repo_name = "{model_repo_name}"
repo_id_model = f"{{hf_username}}/{{model_repo_name}}"

@st.cache_resource
def load_model():
    try:
        repo_path = snapshot_download(repo_id=repo_id_model)

        model_path = os.path.join(repo_path, "logistic_regression_model")

        loaded_model = mlflow.pyfunc.load_model(model_path)

        return loaded_model

    except Exception as e:
        st.error(f"Error loading model from Hugging Face: {{e}}")
        return None


model = load_model()

st.title("Wellness Tourism Package Prediction")
st.write("Predict whether a customer will purchase the Wellness Tourism Package.")

if model is None:
    st.stop()

# Input fields for customer details (matching your dataset columns)
st.header("Customer Information")
age = st.slider("Age", 18, 80, 40)
type_of_contact = st.selectbox("Type of Contact", ['Self Inquiry', 'Company Invited'])
city_tier = st.selectbox("City Tier", [1, 2, 3])
duration_of_pitch = st.slider("Duration of Pitch (minutes)", 5, 60, 15)
occupation = st.selectbox("Occupation", ['Salaried', 'Small Business', 'Large Business', 'Freelancer', 'Government'])
gender = st.selectbox("Gender", ['Male', 'Female', 'Fe Male'])
number_of_person_visiting = st.slider("Number of Persons Visiting", 1, 10, 2)
number_of_followups = st.slider("Number of Follow-ups", 0, 10, 3)
product_pitched = st.selectbox("Product Pitched", ['Deluxe', 'Basic', 'Standard', 'Super Deluxe', 'King'])
preferred_property_star = st.slider("Preferred Property Star", 1.0, 5.0, 4.0)
marital_status = st.selectbox("Marital Status", ['Married', 'Single', 'Divorced'])
number_of_trips = st.slider("NumberOfTrips Annually", 1.0, 50.0, 5.0)
passport = st.selectbox("Passport", [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
pitch_satisfaction_score = st.slider("Pitch Satisfaction Score", 1, 5, 4)
own_car = st.selectbox("Own Car", [0, 1], format_func=lambda x: 'Yes' if x==1 else 'No')
number_of_children_visiting = st.slider("Number of Children Visiting (below 5)", 0.0, 5.0, 1.0)
designation = st.selectbox("Designation", ['Manager', 'Executive', 'Senior Manager', 'AVP', 'VP', 'Director'])
monthly_income = st.slider("Monthly Income", 10000.0, 150000.0, 50000.0)

# Create DataFrame for prediction
input_data = pd.DataFrame([{{
    'Age': age,
    'TypeofContact': type_of_contact,
    'CityTier': city_tier,
    'DurationOfPitch': duration_of_pitch,
    'Occupation': occupation,
    'Gender': gender,
    'NumberOfPersonVisiting': number_of_person_visiting,
    'NumberOfFollowups': number_of_followups,
    'ProductPitched': product_pitched,
    'PreferredPropertyStar': preferred_property_star,
    'MaritalStatus': marital_status,
    'NumberOfTrips': number_of_trips,
    'Passport': passport,
    'PitchSatisfactionScore': pitch_satisfaction_score,
    'OwnCar': own_car,
    'NumberOfChildrenVisiting': number_of_children_visiting,
    'Designation': designation,
    'MonthlyIncome': monthly_income
}}])

if st.button("Predict Purchase"):
    try:
        prediction = model.predict(input_data)

        if prediction[0] == 1:
            st.success(f"Prediction: Customer is likely to purchase the package! ")
        else:
            st.info(f"Prediction: Customer is unlikely to purchase the package.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {{e}}")
"""

with open(app_file_path, "w") as f:
    f.write(app_content)
print(f"Streamlit app.py created at {app_file_path}")

# --- 2. Create Dockerfile ---
# The Dockerfile content is already present in the notebook from previous steps.
# We will ensure it exists in the deployment folder.
dockerfile_content = """
# Use a minimal base image with Python 3.9 installed
FROM python:3.9

# Set the working directory inside the container to /app
WORKDIR /app

# Copy all files from the current directory on the host to the container's /app directory
COPY . .

# Install Python dependencies listed in requirements.txt
RUN pip3 install -r requirements.txt

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user 	PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user . $HOME/app

# Define the command to run the Streamlit app on port "8501" and make it accessible externally
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
"""
with open(dockerfile_path, "w") as f:
    f.write(dockerfile_content)
print(f"Dockerfile created at {dockerfile_path}")

# --- 3. Create requirements.txt for deployment ---
# The requirements.txt content is already present in the notebook from previous steps.
# We will ensure it exists in the deployment folder.
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

# --- 4. Create/Update Hugging Face Space --- #
api = HfApi()
repo_id_space = f"{hf_username}/{space_name}"

try:
    create_repo(repo_id=repo_id_space, repo_type="space", exist_ok=True, token=hf_token, space_sdk='streamlit')
    print(f"Hugging Face Space '{space_name}' created or already exists.")
except Exception as e:
    print(f"Error creating/checking Hugging Face Space: {{e}}")

# --- 5. Upload Deployment Files --- #
print(f"Uploading deployment files from '{deployment_folder}' to '{repo_id_space}'...")

try:
    api.upload_folder(
        folder_path=deployment_folder,
        repo_id=repo_id_space,
        repo_type="space",
        token=hf_token,
        commit_message="Deploy Streamlit app with Dockerfile and requirements.txt"
    )
    print(f"Deployment files successfully uploaded to Hugging Face Space at: https://huggingface.co/spaces/{repo_id_space}")
except Exception as e:
    print(f"Error uploading deployment files to Hugging Face Space: {{e}}")

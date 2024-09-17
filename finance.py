import streamlit as st
import pickle
import pandas as pd

# Load the trained model from the GitHub repository
model_path = 'Financial_inclusion.pkl'

# Open the file and load the model
with open(model_path, 'rb') as file:
    model = pickle.load(file)





# Mappings for categorical features
country_mapping = {0: 'Kenya', 1: 'Rwanda', 2: 'Tanzania', 3: 'Uganda'}
bank_account_mapping = {0: 'No', 1: 'Yes'}
location_type_mapping = {0: 'Rural', 1: 'Urban'}
cellphone_access_mapping = {0: 'No', 1: 'Yes'}
gender_mapping = {0: 'Female', 1: 'Male'}
relationship_mapping = {0: 'Child', 1: 'Head of Household', 2: 'Other non-relatives', 3: 'Other relative', 4: 'Parent',
                        5: 'Spouse'}
marital_status_mapping = {0: 'Divorced/Separated', 1: 'Dont know', 2: 'Married/Living together',
                          3: 'Single/Never Married', 4: 'Widowed'}
education_level_mapping = {0: 'No formal education', 1: 'Other/Dont know/RTA', 2: 'Primary education',
                           3: 'Secondary education', 4: 'Tertiary education', 5: 'Vocational/Specialised training'}
job_type_mapping = {0: 'Dont Know/Refuse to answer', 1: 'Farming and Fishing', 2: 'Formally employed Government',
                    3: 'Formally employed Private', 4: 'Government Dependent', 5: 'Informally employed', 6: 'No Income',
                    7: 'Other Income', 8: 'Remittance Dependent', 9: 'Self employed'}

# Streamlit app
st.set_page_config(page_title="Bank Account Prediction", layout="wide")

st.title("Bank Account Prediction")

st.sidebar.header("User Input")
st.sidebar.write("Please provide the following information:")

# User inputs
country = st.sidebar.selectbox('Country', options=list(country_mapping.values()))
bank_account = st.sidebar.selectbox('Bank Account', options=list(bank_account_mapping.values()))
location_type = st.sidebar.selectbox('Location Type', options=list(location_type_mapping.values()))
cellphone_access = st.sidebar.selectbox('Cellphone Access', options=list(cellphone_access_mapping.values()))
gender_of_respondent = st.sidebar.selectbox('Gender of Respondent', options=list(gender_mapping.values()))
relationship_with_head = st.sidebar.selectbox('Relationship with Head', options=list(relationship_mapping.values()))
marital_status = st.sidebar.selectbox('Marital Status', options=list(marital_status_mapping.values()))
education_level = st.sidebar.selectbox('Education Level', options=list(education_level_mapping.values()))
job_type = st.sidebar.selectbox('Job Type', options=list(job_type_mapping.values()))
household_size = st.sidebar.number_input('Household Size', value=3, min_value=1, max_value=20)
age_of_respondent = st.sidebar.number_input('Age of Respondent', value=30, min_value=0, max_value=120)


# Map inputs to encoded values
def map_input(input_value, mapping):
    return list(mapping.keys())[list(mapping.values()).index(input_value)]


# Create DataFrame for prediction
input_data = {
    'year': 2018,  # Placeholder value
    'household_size': household_size,
    'age_of_respondent': age_of_respondent,
    'country_encoded': map_input(country, country_mapping),
    'location_type_encoded': map_input(location_type, location_type_mapping),
    'cellphone_access_encoded': map_input(cellphone_access, cellphone_access_mapping),
    'gender_of_respondent_encoded': map_input(gender_of_respondent, gender_mapping),
    'relationship_with_head_encoded': map_input(relationship_with_head, relationship_mapping),
    'marital_status_encoded': map_input(marital_status, marital_status_mapping),
    'education_level_encoded': map_input(education_level, education_level_mapping),
    'job_type_encoded': map_input(job_type, job_type_mapping)
}

# Create DataFrame with the same structure as during training
input_df = pd.DataFrame([input_data])

# Make prediction
prediction = model.predict(input_df)

# Display prediction
st.sidebar.header("Prediction Result")
if st.sidebar.button('Predict'):
    result = "The individual is predicted to have a bank account." if prediction[
                                                                          0] == 1 else "The individual is predicted NOT to have a bank account."
    st.sidebar.write(result)

st.markdown("""
    **Instructions:**
    - Fill in the details on the left sidebar.
    - Click the 'Predict' button to see the prediction.

    **Note:**
    - Ensure all fields are filled in appropriately to get accurate predictions.
""")

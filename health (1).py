import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv # Still loading for other potential env vars if any
import json
import numpy as np # Ensure numpy is imported for utility functions
import requests # Ensure requests is imported for API calls

# Load environment variables (kept for other potential non-API key env vars if needed)
load_dotenv()

# --- 1. Gemini AI Client (for Patient Chat and Treatment Plans) ---
class GeminiAIClient:
    def __init__(self, api_key: str, project_id: str): # Added project_id to init
        self.api_key = api_key
        self.project_id = project_id # Store project_id
        self.api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
        self.headers = {
            'Content-Type': 'application/json',
        }
        self.base_prompt_settings = {
            "temperature": 0.7,
            "max_output_tokens": 500,
            "top_p": 0.95,
            "top_k": 40
        }

        # Removed ValueError check here as key is now directly passed/hardcoded for demo
        # A note: For production, load API_KEY and PROJECT_ID from environment variables (e.g., os.getenv)
        # and validate their presence.

    def generate_text(self, prompt: str) -> str:
        """
        Generates text using the Gemini 2.0 Flash model via a POST request.
        Args:
            prompt (str): The input text prompt for the model.
        Returns:
            str: The generated text response from the model.
        """
        chat_history = []
        chat_history.append({
            "role": "user",
            "parts": [{"text": prompt}]
        })
        
        payload = {
            "contents": chat_history,
            "generationConfig": self.base_prompt_settings,
            # project_id might be needed in some Gemini API calls, adding here for completeness
            # "project_id": self.project_id
        }

        try:
            response = requests.post(f"{self.api_url}?key={self.api_key}", headers=self.headers, json=payload)
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
            result = response.json()
            
            if result.get("candidates") and len(result["candidates"]) > 0 and \
               result["candidates"][0].get("content") and \
               result["candidates"][0]["content"].get("parts") and \
               len(result["candidates"][0]["content"]["parts"]) > 0:
                return result["candidates"][0]["content"]["parts"][0]["text"]
            else:
                st.error(f"Gemini API returned an unexpected response structure: {result}")
                return "I'm sorry, I received an empty or malformed response from the AI. Please try again."
        except requests.exceptions.RequestException as e:
            st.error(f"Error communicating with Gemini API: {e}")
            if hasattr(response, 'text'):
                st.error(f"Gemini API Error Response: {response.text}")
            return "I apologize, but I'm currently unable to connect to the AI service. Please try again later."
        except Exception as e:
            st.error(f"An unexpected error occurred during Gemini API call: {e}")
            return "An unexpected error occurred."


# --- 2. Gemini-based Disease Prediction (simulated with generative text) ---
class GeminiDiseasePredictor:
    def __init__(self, api_key: str, project_id: str): # Added project_id to init
        self.gemini_client = GeminiAIClient(api_key, project_id) # Pass project_id
        
    def predict_disease(self, symptoms: list) -> dict:
        """
        Simulates disease prediction using Gemini API by generating text based on symptoms.
        
        Args:
            symptoms (list): A list of strings representing the user's symptoms.
        
        Returns:
            dict: A dictionary containing the simulated prediction and advice.
        """
        symptoms_str = ", ".join(symptoms) if symptoms else "no specific symptoms"
        
        prompt = f"""You are HealthAI's Disease Predictor, powered by a large language model.
        Given the following symptoms, suggest one or two potential conditions.
        Then, provide a brief likelihood assessment (e.g., "High", "Medium", "Low" - based on common knowledge, not statistical model).
        Finally, state clear next steps and emphasize the necessity of consulting a medical professional for diagnosis.
        
        Symptoms: {symptoms_str}
        
        Format your response strictly as follows, filling in the blanks:
        Potential Condition: [Condition Name]
        Likelihood Assessment: [High/Medium/Low]
        Recommended Next Steps: [Actionable advice]
        
        Disclaimer: This is an AI-generated prediction based on common knowledge and is not a substitute for professional medical diagnosis.
        """
        
        generated_text = self.gemini_client.generate_text(prompt)
        
        # Parse the generated text into a structured dictionary
        prediction_info = {
            "predicted_disease": "Unknown",
            "likelihood": "N/A",
            "next_steps": "Please consult a healthcare professional for an accurate diagnosis and treatment plan."
        }
        
        lines = generated_text.split('\n')
        for line in lines:
            if line.startswith("Potential Condition:"):
                prediction_info["predicted_disease"] = line.replace("Potential Condition:", "").strip()
            elif line.startswith("Likelihood Assessment:"):
                prediction_info["likelihood"] = line.replace("Likelihood Assessment:", "").strip()
            elif line.startswith("Recommended Next Steps:"):
                prediction_info["next_steps"] = line.replace("Recommended Next Steps:", "").strip()
        
        return prediction_info

# --- 3. Utility Functions (for Health Analytics) ---
def generate_sample_health_data(num_days=30):
    """
    Generates synthetic health data for demonstration purposes.
    This simulates vital signs over a period of days.
    """
    dates = pd.date_range(end=pd.Timestamp.now(), periods=num_days, freq='D')
    # Generate somewhat realistic but random values for vital signs
    heart_rate = np.random.randint(60, 100, num_days) + np.random.randn(num_days) * 5
    blood_pressure_systolic = np.random.randint(110, 140, num_days) + np.random.randn(num_days) * 5
    blood_pressure_diastolic = np.random.randint(70, 90, num_days) + np.random.randn(num_days) * 3
    blood_glucose = np.random.randint(80, 120, num_days) + np.random.randn(num_days) * 10
    
    df = pd.DataFrame({
        'Date': dates,
        'Heart Rate (bpm)': heart_rate.astype(int),
        'Blood Pressure (Systolic)': blood_pressure_systolic.astype(int),
        'Blood Pressure (Diastolic)': blood_pressure_diastolic.astype(int),
        'Blood Glucose (mg/dL)': blood_glucose.astype(int)
    })
    return df

def analyze_health_trends_simple(df: pd.DataFrame) -> str:
    """
    Performs a simple analysis of health data and provides basic insights.
    This is a basic rule-based analysis; a real system would use more complex ML.
    """
    insights = []

    # Analyze heart rate
    avg_hr = df['Heart Rate (bpm)'].mean()
    if avg_hr > 90:
        insights.append(f"Your average heart rate ({avg_hr:.0f} bpm) is consistently on the higher side. It might be beneficial to discuss this with a healthcare provider.")
    elif avg_hr < 60:
        insights.append(f"Your average heart rate ({avg_hr:.0f} bpm) is a bit low. If you're not an athlete, you might want to monitor this.")
    else:
        insights.append(f"Your average heart rate ({avg_hr:.0f} bpm) is generally within a healthy range.")

    # Analyze blood pressure
    avg_bp_systolic = df['Blood Pressure (Systolic)'].mean()
    avg_bp_diastolic = df['Blood Pressure (Diastolic)'].mean()
    if avg_bp_systolic > 130 or avg_bp_diastolic > 85:
        insights.append(f"Your average blood pressure ({avg_bp_systolic}/{avg_bp_diastolic} mmHg) appears elevated. Regular monitoring and lifestyle adjustments, or professional consultation, are recommended.")
    else:
        insights.append(f"Your average blood pressure ({avg_bp_systolic}/{avg_bp_diastolic} mmHg) is generally healthy.")

    # Analyze blood glucose
    avg_glucose = df['Blood Glucose (mg/dL)'].mean()
    if avg_glucose > 110: # Assuming non-fasting
        insights.append(f"Your average blood glucose ({avg_glucose:.0f} mg/dL) is on the higher side. Consider dietary adjustments and consult a doctor, especially if readings are consistently high.")
    elif avg_glucose < 70:
        insights.append(f"Your average blood glucose ({avg_glucose:.0f} mg/dL) is low. Monitor for symptoms of hypoglycemia and discuss with your doctor.")
    else:
        insights.append(f"Your average blood glucose ({avg_glucose:.0f} mg/dL) is within a healthy range.")
        
    # Check for recent significant changes (simple example)
    if len(df) >= 7:
        last_week_hr = df['Heart Rate (bpm)'].tail(7).mean()
        prev_week_hr = df['Heart Rate (bpm)'].iloc[-14:-7].mean()
        if abs(last_week_hr - prev_week_hr) > 10: # Significant change
            insights.append("Noticeable fluctuations in your recent heart rate data. This could be due to various factors, but consistent large changes warrant attention.")


    if not insights:
        return "No significant health trends or concerns identified based on the provided data."
    return "\n\n".join(insights)


# --- Initialize Clients for Streamlit Application ---
# These are initialized once when the app starts.
# Error handling ensures the app can still run even if one service isn't configured.
gemini_ai_client = None
gemini_disease_predictor = None

# Using the provided string directly for demonstration.
# In a real application, retrieve this from environment variables.
DEMO_API_KEY = "AIzaSyA2V5ZNaovCbktBnUCDXlQ-gklmjQaDC20"
# Note: A project ID is usually a separate string. This is a placeholder; replace with your actual Project ID.
DEMO_PROJECT_ID = "244519717988" 

try:
    gemini_ai_client = GeminiAIClient(DEMO_API_KEY, DEMO_PROJECT_ID)
    gemini_disease_predictor = GeminiDiseasePredictor(DEMO_API_KEY, DEMO_PROJECT_ID)
except ValueError as e:
    st.sidebar.error(f"Gemini API setup error: {e}")
    st.sidebar.info("Patient Chat, Treatment Plans, and Disease Prediction features will be limited.")
except Exception as e:
    st.sidebar.error(f"Unexpected error initializing Gemini clients: {e}")


# --- Streamlit Application UI ---
st.set_page_config(layout="wide", page_title="HealthAI - Intelligent Healthcare Assistant")

st.sidebar.title("HealthAI Navigation")
page = st.sidebar.radio("Go to", ["Patient Chat", "Disease Prediction", "Treatment Plans", "Health Analytics"])

st.title("🏥 HealthAI - Intelligent Healthcare Assistant")
st.markdown("Harnessing Gemini AI to provide intelligent healthcare assistance.")

# --- Patient Chat (Scenario 4) ---
if page == "Patient Chat":
    st.header("💬 Patient Chat")
    st.write("Ask any health-related question and get an intelligent response. Please note, this AI is for informational purposes only and cannot provide medical diagnoses or treatment.")

    if gemini_ai_client:
        # Initialize chat history in session state if not present
        if "messages" not in st.session_state:
            st.session_state.messages = []
            st.session_state.messages.append({"role": "assistant", "content": "Hello! How can I assist you with your health questions today?"})

        # Display previous chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input for the user
        if prompt := st.chat_input("Ask your health question here..."):
            # Add user message to chat history and display
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate and display AI response
            with st.chat_message("assistant"):
                with st.spinner("HealthAI is thinking..."):
                    # Craft a specific prompt for the LLM to act as a helpful health assistant
                    full_prompt = f"""You are HealthAI, a helpful and empathetic AI healthcare assistant. 
                    Provide accurate, concise medical information based on the user's question.
                    Always advise seeking professional medical advice for diagnoses or treatment, and clarify that you are an AI.
                    User question: {prompt}
                    """
                    response = gemini_ai_client.generate_text(full_prompt)
                    
                    # Add a prominent disclaimer to every AI response in chat
                    response_with_disclaimer = (
                        f"{response}\n\n"
                        "--- \n"
                        "**Disclaimer:** This information is for general knowledge and informational purposes only, "
                        "and does not constitute medical advice. Always consult with a qualified healthcare professional "
                        "for any health concerns or before making any decisions related to your health or treatment. "
                        "I am an AI and cannot provide medical diagnoses."
                    )
                    st.markdown(response_with_disclaimer)
                    st.session_state.messages.append({"role": "assistant", "content": response_with_disclaimer})
    else:
        st.warning("Patient Chat is unavailable. Please ensure the Gemini API key is correctly provided by the Canvas environment or set up for local development.")


# --- Disease Prediction (Scenario 1) ---
elif page == "Disease Prediction":
    st.header("🦠 Disease Prediction")
    st.write("Enter your symptoms to get potential condition predictions powered by Generative AI.")
    st.info("This feature uses the Gemini API to analyze symptoms and suggest potential conditions based on its knowledge. It is not a statistical model trained on specific medical datasets.")

    common_symptoms = [
        "headache", "fatigue", "fever", "sore throat", "cough", "nausea",
        "vomiting", "stomach pain", "dizziness", "chest pain", "shortness of breath",
        "rash", "joint pain", "muscle aches", "runny nose", "sneezing", "chills", "body aches",
        "loss of taste", "loss of smell", "difficulty breathing", "diarrhea"
    ]
    
    selected_symptoms = st.multiselect(
        "Select your symptoms from the list below (you can type to search):",
        options=common_symptoms
    )

    other_symptoms_text = st.text_input("Enter any other symptoms (comma-separated):")
    if other_symptoms_text:
        other_symptoms_list = [s.strip().lower() for s in other_symptoms_text.split(',') if s.strip()]
        selected_symptoms.extend(other_symptoms_list)
        selected_symptoms = list(dict.fromkeys(selected_symptoms)) 

    st.markdown(f"**Your selected symptoms:** {', '.join(selected_symptoms) if selected_symptoms else 'None'}")

    if st.button("Predict Potential Conditions"):
        if selected_symptoms:
            st.write(f"Analyzing symptoms: {', '.join(selected_symptoms)}...")

            if gemini_disease_predictor:
                with st.spinner("Consulting Gemini AI for prediction..."):
                    try:
                        prediction_info = gemini_disease_predictor.predict_disease(selected_symptoms)
                        
                        st.subheader("Potential Condition Predictions:")
                        st.success(f"Potential Condition: **{prediction_info['predicted_disease']}**")
                        st.info(f"Likelihood Assessment: {prediction_info['likelihood']}")
                        st.warning(f"Recommended Next Steps: {prediction_info['next_steps']}")
                        st.markdown("**Disclaimer:** This is an AI-generated prediction based on general knowledge and is not a substitute for professional medical diagnosis. Always consult a healthcare professional for an accurate diagnosis and treatment plan.")
                    except Exception as e:
                        st.error(f"Error during disease prediction: {e}. Please ensure Gemini API is configured.")
            else:
                st.warning("Disease Prediction is unavailable. Please ensure the Gemini API key is correctly provided by the Canvas environment or set up for local development.")
        else:
            st.warning("Please select or enter symptoms to get a prediction.")


# --- Treatment Plans (Scenario 2) ---
elif page == "Treatment Plans":
    st.header("💊 Personalized Treatment Plans")
    st.write("Get evidence-based treatment recommendations for a diagnosed condition using Generative AI.")

    if gemini_ai_client:
        diagnosed_condition = st.text_input("Enter your diagnosed condition (e.g., Type 2 Diabetes, Hypertension, Migraine):")
        
        patient_profile_info = st.text_area(
            "Optionally, provide relevant patient profile details (e.g., age, gender, existing medical conditions, "
            "allergies, current medications, lifestyle factors like diet/exercise habits, severity of condition):"
        )

        if st.button("Generate Treatment Plan"):
            if diagnosed_condition:
                prompt = f"""You are HealthAI, a specialized AI for generating comprehensive, evidence-based treatment plans.
                Generate a treatment plan for a patient diagnosed with: "{diagnosed_condition}".
                Consider the following patient details if provided: {patient_profile_info if patient_profile_info else 'No additional patient details provided.'}
                
                The plan should be structured clearly with the following sections:
                1.  **Overview of the Condition:** (Brief, high-level understanding for the patient)
                2.  **Medications:** (Commonly prescribed drug classes or types, general considerations, always emphasize consultation with a doctor for specific prescriptions)
                3.  **Lifestyle Modifications:** (Detailed advice on diet, exercise, stress management, sleep, avoiding triggers specific to the condition)
                4.  **Follow-up Testing/Monitoring:** (Recommended tests, frequency, what key metrics to monitor)
                5.  **When to Seek Immediate Medical Attention:** (Clear warning signs or emergencies)
                
                Maintain a helpful, informative, and empathetic tone.
                Strictly end your response with a clear, bolded disclaimer: "**Disclaimer:** This AI-generated treatment plan is for informational purposes only and is not a substitute for professional medical advice, diagnosis, or treatment. Always consult with your doctor or other qualified healthcare provider for any questions regarding a medical condition or before starting any new treatment."
                """
                with st.spinner("Generating personalized treatment plan..."):
                    treatment_plan = gemini_ai_client.generate_text(prompt)
                    st.subheader("Generated Treatment Plan:")
                    st.markdown(treatment_plan)
            else:
                st.warning("Please enter a diagnosed condition to generate a treatment plan.")
    else:
        st.warning("Treatment Plans feature is unavailable. Please ensure the Gemini API key is correctly provided by the Canvas environment or set up for local development.")


# --- Health Analytics (Scenario 3) ---
elif page == "Health Analytics":
    st.header("📈 Health Analytics Dashboard")
    st.write("Visualize your vital signs over time and receive AI-generated insights.")

    st.markdown("**(Demonstration using synthetic data by default. For real use, connect to your actual health data source or upload a CSV.)**")

    # Initialize health_data in session state
    if "health_data" not in st.session_state:
        st.session_state.health_data = generate_sample_health_data(num_days=60) # Generate 60 days of data

    df = st.session_state.health_data
    
    st.subheader("Patient Health Metrics Over Time")

    # Select box for different metrics to plot
    metric_options = ['Heart Rate (bpm)', 'Blood Pressure (Systolic)', 'Blood Pressure (Diastolic)', 'Blood Glucose (mg/dL)']
    selected_metric = st.selectbox("Select a metric to visualize:", metric_options)

    # Plotting the selected metric using Matplotlib and Seaborn
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=df, x='Date', y=selected_metric, ax=ax, marker='o', linestyle='-')
    ax.set_title(f'{selected_metric} Over Time', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(selected_metric, fontsize=12)
    plt.xticks(rotation=45, ha='right') # Rotate dates for better readability
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout() # Adjust layout to prevent labels from overlapping
    st.pyplot(fig)
    plt.close(fig) # Close the figure to free up memory

    st.subheader("AI-Generated Health Insights")
    if st.button("Generate Insights from Data"):
        with st.spinner("Analyzing data and generating insights..."):
            # First, get rule-based insights from our utility function
            simple_insights = analyze_health_trends_simple(df)
            st.info(simple_insights)
            
            # If Gemini client is available, get more nuanced insights from LLM
            if gemini_ai_client:
                # Prepare a summary of the data for the LLM
                data_summary_for_ai = (
                    f"Overall Health Data Summary (last {len(df)} days):\n"
                    f"- Average Heart Rate: {df['Heart Rate (bpm)'].mean():.0f} bpm (Range: {df['Heart Rate (bpm)'].min()}-{df['Heart Rate (bpm)'].max()} bpm)\n"
                    f"- Average Systolic BP: {df['Blood Pressure (Systolic)'].mean():.0f} mmHg (Range: {df['Blood Pressure (Systolic)'].min()}-{df['Blood Pressure (Systolic)'].max()} mmHg)\n"
                    f"- Average Diastolic BP: {df['Blood Pressure (Diastolic)'].mean():.0f} mmHg (Range: {df['Blood Pressure (Diastolic)'].min()}-{df['Blood Pressure (Diastolic)'].max()} mmHg)\n"
                    f"- Average Blood Glucose: {df['Blood Glucose (mg/dL)'].mean():.0f} mg/dL (Range: {df['Blood Glucose (mg/dL)'].min()}-{df['Blood Glucose (mg/dL)'].max()} mg/dL)\n"
                    f"\nRule-based observations: {simple_insights}\n"
                )

                prompt_for_ai_insight = f"""You are HealthAI. Analyze the following health data summary and provide concise, actionable general health insights and recommendations. Do NOT give specific medical diagnoses or treatment plans. Focus on general well-being, trends, and lifestyle suggestions.
                
                {data_summary_for_ai}
                
                Please ensure your response concludes with a clear, bolded disclaimer: "**Disclaimer:** These insights are for general informational purposes only and should not be taken as medical advice. Always consult a healthcare professional for personalized guidance."
                """
                ai_insights_llm = gemini_ai_client.generate_text(prompt_for_ai_insight)
                st.markdown("---")
                st.subheader("Advanced AI Insights:")
                st.markdown(ai_insights_llm)
            else:
                st.warning("Advanced AI insights are unavailable. Please configure your Gemini API key.")
    
    st.subheader("Upload Your Own Health Data (CSV)")
    st.markdown("Upload a CSV file with columns: `Date`, `Heart Rate (bpm)`, `Blood Pressure (Systolic)`, `Blood Pressure (Diastolic)`, `Blood Glucose (mg/dL)`.")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file, parse_dates=['Date'])
            # Basic validation for required columns
            required_cols = ['Date', 'Heart Rate (bpm)', 'Blood Pressure (Systolic)', 'Blood Pressure (Diastolic)', 'Blood Glucose (mg/dL)']
            if all(col in uploaded_df.columns for col in required_cols):
                st.session_state.health_data = uploaded_df
                st.success("Health data uploaded successfully! The dashboard has been updated.")
                st.experimental_rerun() # Rerun to update the plots and analyses
            else:
                st.error(f"Missing one or more required columns in your CSV. Please ensure it contains: {', '.join(required_cols)}")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}. Please ensure it's a valid CSV with correct date format in the 'Date' column.")


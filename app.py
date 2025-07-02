import streamlit as st
import google.generativeai as genai
from ml_models import predict_sleep_quality

# Configure Gemini
genai.configure(api_key=st.secrets["API_KEY"])
model = genai.GenerativeModel("gemini-2.5-flash")

st.title("😴 Sleep Pattern Analyzer")

st.markdown("""
Analyze your sleep habits, predict your sleep quality, and get AI-powered personalized advice!
""")

# Inputs
sleep_duration = st.slider("Sleep Duration (hours)", 3.0, 12.0, 7.0, step=0.5)
bed_hour = st.slider("Bedtime Hour (24h)", 20, 2, 23)
wake_hour = st.slider("Wake-up Hour (24h)", 4, 12, 7)
interruptions = st.number_input("Number of interruptions during sleep", min_value=0, max_value=10, value=1)

if st.button("Analyze Sleep"):
    with st.spinner("Analyzing your sleep pattern..."):
        quality = predict_sleep_quality(sleep_duration, bed_hour, wake_hour, interruptions)
        
        # Gemini prompt
        prompt = f"""
        Here are my sleep details:
        - Sleep duration: {sleep_duration} hours
        - Bedtime: {bed_hour}:00
        - Wake time: {wake_hour}:00
        - Number of interruptions: {interruptions}

        My predicted sleep quality: {quality}.

        Provide me with:
        - Possible reasons for this quality.
        - Practical tips to improve sleep.
        - Friendly motivational message.

        Present this as an encouraging note from a sleep coach.
        """

        response = model.generate_content(prompt)
        st.success(f"Predicted Sleep Quality: **{quality.capitalize()}**")
        st.markdown("---")
        st.markdown(response.text)

st.caption("🌙 Built with Python, Streamlit, Gemini API & ML")

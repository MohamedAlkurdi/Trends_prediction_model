import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ----------------------
# Title and Introduction
# ----------------------
st.set_page_config(page_title="Interactive API Documentation", layout="wide")
st.title("📘 Interactive Documentation for AI API")

st.markdown("""
This Streamlit app serves as a **live, interactive documentation interface** for your AI API and related models.

Instead of a static document, this UI enables you to:
- Test endpoints with sample data
- Visualize input/output examples
- Read expandable sections of code and descriptions
- See live results from models or mock endpoints

**Target Audience:** Developers, Data Scientists, and Stakeholders who want to explore how the system works.

---
""")

# ----------------------
# Sample Data Placeholder
# ----------------------
st.header("📊 Example Input Data")
st.markdown("Preview of input data used for testing the models/API endpoints.")

sample_data = pd.DataFrame({
    'text': ["The economy is recovering quickly.", "Celebrity gossip dominates the media."],
    'country': ["Germany", "USA"],
    'timestamp': ["2024-01-01", "2024-01-02"]
})

st.dataframe(sample_data)

# ----------------------
# API Endpoints Section
# ----------------------
st.header("🔌 API Endpoints")

with st.expander("/predict-interest-trend (POST)"):
    st.markdown("""
    **Description:** Returns predicted interest trends based on topic, region, and date.

    **Payload Example:**
    ```json
    {
        "topic": "entertainment",
        "country": "Germany",
        "date": "2024-01-01"
    }
    ```

    **Response Example:**
    ```json
    {
        "predicted_trend": 0.76,
        "confidence": 0.92
    }
    ```
    """)

# ----------------------
# Code Snippet Section
# ----------------------
st.header("🧠 Code Logic Snippets")

with st.expander("📌 Forecasting Function (Prophet-based)"):
    st.code('''
def forecast_interest(data):
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat']]
    ''', language='python')

# ----------------------
# Visual Output Placeholder
# ----------------------
st.header("📈 Visualization Example")
st.markdown("Placeholder for trend plots or model outputs.")

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [0.6, 0.8, 0.9], label='Predicted Interest')
ax.set_title("Predicted Trend Over Time")
ax.legend()
st.pyplot(fig)

# ----------------------
# Conclusion Section
# ----------------------
st.header("🧾 Conclusion")
st.markdown("""
This documentation is a **living app**. Feel free to:
- Plug in real endpoints
- Connect to your models
- Extend sections for deeper technical breakdowns (e.g., BERTopic pipeline, NER examples)
- Add performance metrics or logs

> 🛠️ **Next Steps:** Integrate live API calls and secure them with authentication.
""")

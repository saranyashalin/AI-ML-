import streamlit as st
import pandas as pd
import joblib
from io import BytesIO
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import matplotlib.pyplot as plt

# === Streamlit Page Config ===
st.set_page_config(page_title="📞 EMI Call Advisor with GenAI", layout="wide")
st.title("📞 EMI Call Prediction & AI Explanation")

# === Sidebar Navigation ===
with st.sidebar:
    st.markdown("## 🚀 Navigation")
    page = st.radio(
        "Select a section:",
        ["📋 Predictions", "📊 Visualization"],
        index=0,
        help="Switch between prediction view and chart view"
    )

# === Custom CSS Styling ===
def local_css():
    st.markdown("""
        <style>
        body, .main, .block-container {
            background-color: #1e1e2f;
            color: white;
        }
        .block-container {
            max-width: 95% !important;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        h1 {
            font-size: 28px !important;
        }
        .stFileUploader, .stDownloadButton > button, .stButton>button {
            background-color: #2a2a3d;
            color: white !important;
            border-radius: 10px;
        }
        .stFileUploader label {
            color: white !important;
            font-size: 16px !important;
        }
        .stDownloadButton > button:hover {
            background-color: #3a3a4d !important;
        }
        .stTextInput>div>div>input,
        .stSelectbox>div>div>div>input,
        .stTextArea textarea {
            background-color: #2a2a3d;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# === Load Trained Random Forest Model ===
model = joblib.load("random_forest_model.pkl")

# === Load GenAI Model (FLAN-T5) ===
@st.cache_resource
def load_flan_model():
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    return tokenizer, model

tokenizer, flan_model = load_flan_model()

# === Function: Generate Explanation ===
def generate_explanation_flan(input_row, prediction, best_hour, best_prob):
    call_status = "likely to answer" if prediction == 1 else "unlikely to answer"
    prompt = (
        f"A customer service agent is planning to call this person at {best_hour}:00.\n"
        f"Customer Profile:\n"
        f"- Age: {input_row['customer_age']}\n"
        f"- Occupation: {input_row['occupation']}\n"
        f"- City: {input_row.get('city', 'Unknown')}\n"
        f"- Days overdue: {input_row['days_overdue']}\n"
        f"- Number of previous calls: {input_row['num_prev_calls']}\n"
        f"- Day type: {'Weekend' if input_row['is_weekend'] else 'Weekday'}\n"
        f"- Model Prediction: {call_status} (confidence {best_prob*100:.2f}%)\n\n"
        f"Q: Based on the profile, give a helpful 1-line reason why {best_hour}:00 is the best call time.\n"
        f"A:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = flan_model.generate(
        **inputs,
        max_new_tokens=60,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.95
    )
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text.split("A:")[-1].strip()

# === Function: Predict Best Call Hour ===
def predict_best_call_time(customer_row, model):
    max_prob = -1
    best_hour = 0
    for hour in range(9, 18):
        test_row = customer_row.copy()
        test_row["hour"] = hour
        df = pd.DataFrame([test_row])
        df_encoded = pd.get_dummies(df, drop_first=True)
        for col in model.feature_names_in_:
            if col not in df_encoded.columns:
                df_encoded[col] = 0
        df_encoded = df_encoded[model.feature_names_in_]
        prob = model.predict_proba(df_encoded)[0][1]
        if prob > max_prob:
            max_prob = prob
            best_hour = hour
    return best_hour, max_prob

# === Cache Predictions and Explanations ===
@st.cache_data(show_spinner=False)
def get_predictions(df_raw, _model):  # Note the underscore!
    feature_cols = ['day_of_week', 'is_weekend', 'days_overdue', 'num_prev_calls', 'occupation', 'customer_age']
    df_encoded = pd.get_dummies(df_raw[feature_cols], drop_first=True)
    for col in _model.feature_names_in_:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    df_encoded = df_encoded[_model.feature_names_in_]

    predictions = _model.predict(df_encoded)
    best_times, best_probs, explanations = [], [], []

    for i in range(len(df_raw)):
        row = df_raw.iloc[i]
        best_hour, best_prob = predict_best_call_time(row.to_dict(), _model)
        explanation = generate_explanation_flan(row, predictions[i], best_hour, best_prob)
        best_times.append(f"{best_hour}:00")
        best_probs.append(f"{best_prob*100:.2f}%")
        explanations.append(explanation)

    df_output = df_raw.copy()
    df_output["prediction"] = predictions
    df_output["Best_Call_Time"] = best_times
    df_output["Best_Prob%"] = best_probs
    df_output["AI_Call_Advisor"] = explanations
    return df_output

# === File Upload ===
uploaded_file = st.file_uploader("📂 Upload Excel or CSV File", type=["xlsx", "csv"])

if uploaded_file is not None:
    df_raw = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(".xlsx") else pd.read_csv(uploaded_file)

    st.write("### 🔍 Preview of Uploaded Input Data")
    st.dataframe(df_raw.head())

    if page == "📋 Predictions":
        df = get_predictions(df_raw, model)

        final_cols = ["customer_id", "city", "gender", "customer_age", "occupation",
                      "days_overdue", "num_prev_calls", "prediction", "AI_Call_Advisor",
                      "Best_Call_Time", "Best_Prob%"]
        final_cols = [col for col in final_cols if col in df.columns]

        st.write("### ✅ AI Predictions & Call Advisor Insights")
        st.dataframe(df[final_cols])

        output = BytesIO()
        df[final_cols].to_excel(output, index=False)
        st.download_button("📥 Download Results (Excel)", data=output.getvalue(),
                           file_name="call_predictions_with_explanations.xlsx", mime="application/vnd.ms-excel")

    elif page == "📊 Visualization":
        df = get_predictions(df_raw, model)

        st.markdown("### ⏰ Overall Best Call Time Distribution")
        if "Best_Call_Time" in df.columns:
            overall_counts = df["Best_Call_Time"].value_counts().sort_index()
            fig, ax = plt.subplots(figsize=(2.8, 2.8))
            wedges, texts, autotexts = ax.pie(
                overall_counts,
                labels=overall_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 6},
                pctdistance=0.75
            )
            for text in texts:
                text.set_fontsize(6)
            for autotext in autotexts:
                autotext.set_fontsize(6)
            ax.axis("equal")
            st.pyplot(fig, use_container_width=False)

                # === Pie Chart by Simplified Age Group ===
        if "customer_age" in df.columns:
            st.markdown("### 👤 Best Call Time by Age Group")

            # Define two age groups: Below 35 and 35+
            df["Age_Group"] = df["customer_age"].apply(lambda x: "Below 35" if x < 35 else "35 and Above")

            for age_group in df["Age_Group"].unique():
                group_df = df[df["Age_Group"] == age_group]
                if not group_df.empty:
                    st.write(f"**Age Group: {age_group}**")
                    group_counts = group_df["Best_Call_Time"].value_counts().sort_index()
                    fig, ax = plt.subplots(figsize=(2.8, 2.8))
                    wedges, texts, autotexts = ax.pie(
                        group_counts,
                        labels=group_counts.index,
                        autopct='%1.1f%%',
                        startangle=90,
                        textprops={'fontsize': 6},
                        pctdistance=0.75
                    )
                    for text in texts:
                        text.set_fontsize(6)
                    for autotext in autotexts:
                        autotext.set_fontsize(6)
                    ax.axis("equal")
                    st.pyplot(fig, use_container_width=False)
   
        # === Pie Chart by Gender ===
        if "gender" in df.columns:
            st.markdown("### 🚻 Best Call Time by Gender")
            for gender in df["gender"].dropna().unique():
                gender_df = df[df["gender"] == gender]
                if not gender_df.empty:
                    st.write(f"**Gender: {gender}**")
                    gender_counts = gender_df["Best_Call_Time"].value_counts().sort_index()
                    fig, ax = plt.subplots(figsize=(2.8, 2.8))
                    wedges, texts, autotexts = ax.pie(
                        gender_counts,
                        labels=gender_counts.index,
                        autopct='%1.1f%%',
                        startangle=90,
                        textprops={'fontsize': 6},
                        pctdistance=0.75
                    )
                    for text in texts:
                        text.set_fontsize(6)
                    for autotext in autotexts:
                        autotext.set_fontsize(6)
                    ax.axis("equal")
                    st.pyplot(fig, use_container_width=False)

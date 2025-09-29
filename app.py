import os
import streamlit as st
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# --- Force CPU + disable SDPA ---
os.environ["DISABLE_TRANSFORMERS_SDPA"] = "1"

# Define labels (must match training labels order)
labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Cache model to avoid reloading each run
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Alfonso-E/emotion-classifier-model")
    model = AutoModelForSequenceClassification.from_pretrained(
        "Alfonso-E/emotion-classifier-model",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False
    ).to("cpu")  # ✅ force CPU

    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=-1  # ✅ -1 = CPU
    )

classifier = load_model()

# --- Streamlit UI ---
st.title("Emotion Classifier")
st.write("Enter text and I will predict the emotion:")

user_input = st.text_area("Your text here:")
if st.button("Predict"):
    result = classifier(user_input, return_all_scores=True)

    st.write(f"**Text:** {user_input}")

    # Convert results to lists
    scores = [r["score"] for r in result[0]]

    # Show probabilities
    st.subheader("Emotion Probabilities:")
    for label, score in zip(labels, scores):
        st.write(f"- {label}: {score:.4f}")

    # Bar chart
    df = pd.DataFrame({"Emotion": labels, "Probability": scores})
    st.bar_chart(df.set_index("Emotion"))

    # Predicted emotion
    predicted_emotion = labels[scores.index(max(scores))]
    st.subheader(f"🏆 Predicted Emotion: {predicted_emotion}")

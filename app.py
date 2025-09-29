import os
import streamlit as st
from transformers import pipeline
import pandas as pd

# -------------------------------
# 1. Cache model so it loads once
# -------------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="Alfonso-E/emotion-classifier-model",  # your HF repo
        return_all_scores=True
    )

classifier = load_model()

# Define label names (order must match training labels)
labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# -------------------------------
# 2. Streamlit UI
# -------------------------------
st.title('Emotion Classifier 🎭')
st.write('Enter text and I will predict the emotion:')

user_input = st.text_area('Your text here:')

if st.button('Predict'):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            result = classifier(user_input, return_all_scores=True)

        st.write(f"**Text:** {user_input}")

        # Convert results into lists for plotting
        scores = [r['score'] for r in result[0]]

        # Display emotion probabilities
        st.subheader("📊 Emotion Probabilities:")
        for label, score in zip(labels, scores):
            st.write(f"- {label}: {score:.4f}")

        # Bar chart visualization
        df = pd.DataFrame({"Emotion": labels, "Probability": scores})
        st.bar_chart(df.set_index("Emotion"))

        # Predicted emotion
        predicted_emotion = labels[scores.index(max(scores))]
        st.subheader(f"🏆 Predicted Emotion: {predicted_emotion}")

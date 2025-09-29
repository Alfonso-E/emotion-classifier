import os
import streamlit as st
import pandas as pd
import torch
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Force GPU if available, otherwise fallback to CPU
device = 0 if torch.cuda.is_available() else -1

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Alfonso-E/emotion-classifier-model")
    model = AutoModelForSequenceClassification.from_pretrained(
        "Alfonso-E/emotion-classifier-model",
        torch_dtype=torch.float32  # make sure weights load eagerly, not as meta
    )
    return pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device=device
    )

classifier = load_model()

# Define label names (order must match your training labels)
labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Streamlit UI
st.title('Emotion Classifier')
st.write('Enter text and I will predict the emotion:')

user_input = st.text_area('Your text here: ')
if st.button('Predict') and user_input.strip():
    result = classifier(user_input, return_all_scores=True)

    st.write(f"**Text:** {user_input}")

    # Convert results into lists for plotting
    scores = [r['score'] for r in result[0]]

    # Display emotion probabilities
    st.subheader("Emotion Probabilities:")
    for label, score in zip(labels, scores):
        st.write(f"- {label}: {score:.4f}")

    # Bar chart visualization
    df = pd.DataFrame({"Emotion": labels, "Probability": scores})
    st.bar_chart(df.set_index("Emotion"))

    # Predicted emotion
    predicted_emotion = labels[scores.index(max(scores))]
    st.subheader(f"🏆 Predicted Emotion: {predicted_emotion}")

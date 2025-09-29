import os
import streamlit as st
# Force CPU + disable SDPA (fixes meta tensor errors)
os.environ["DISABLE_TRANSFORMERS_SDPA"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # ensures Hugging Face loads on CPU
from transformers import pipeline
import pandas as pd

from transformers import pipeline
import pandas as pd

# Path to your local model
model_path = os.path.join(os.path.dirname(__file__), "emotion_model")

# Load pipeline
classifier = pipeline(
    "text-classification",
    model="Alfonso-E/emotion-classifier-model",  # your actual HF repo
    return_all_scores=True
)


# Define label names (order must match training labels)
labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# Streamlit UI
st.title('Emotion Classifier')
st.write('Enter text and I will predict the emotion:')


user_input = st.text_area('Your text here: ')
if st.button('Predict'):
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

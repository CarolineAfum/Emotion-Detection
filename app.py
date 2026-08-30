# ============================
# app.py — Streamlit deployment
# Run locally with: streamlit run app.py
# Deploy via Streamlit Community Cloud, pointing at this file.
# ============================
import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from transformers import BertModel, BertTokenizer

LABEL_MAP = {0: "sad", 1: "happy", 2: "love", 3: "angry", 4: "fear", 5: "surprised"}


def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    return text.lower().strip()


@st.cache_resource(show_spinner="Loading BERT model (first run only)...")
def load_bert():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    bert_model = BertModel.from_pretrained("bert-base-uncased")
    bert_model.eval()
    return tokenizer, bert_model


@st.cache_resource(show_spinner="Loading trained classifier...")
def load_classifier():
    # emotion_model.pkl must sit next to app.py in the deployed repo.
    # It's produced by running train_model.py locally first.
    if not os.path.exists("emotion_model.pkl"):
        return None
    return joblib.load("emotion_model.pkl")


def get_embedding(text: str, tokenizer, bert_model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).numpy()


# ---------------- Pages ----------------

def page1():
    st.title("Corpora Viewer")
    if os.path.exists("emotion_distribution.png"):
        st.image("emotion_distribution.png", caption="Emotion Distribution")
    else:
        st.warning(
            "emotion_distribution.png not found. Run train_model.py first — "
            "it now generates this file automatically — then copy it next to app.py."
        )


def page2():
    st.title("Embeddings")
    st.caption("A 2D PCA projection of BERT embeddings for the test set, colored by true emotion.")

    if not os.path.exists("embeddings_demo.pkl"):
        st.warning(
            "embeddings_demo.pkl not found. Re-run the latest train_model.py — "
            "it now saves this file automatically — then copy it next to app.py."
        )
        return

    data = joblib.load("embeddings_demo.pkl")
    coords = data["coords"]
    labels = data["labels"]
    label_map = data["label_map"]
    var_explained = data.get("explained_variance")

    fig, ax = plt.subplots(figsize=(7, 6))
    for label_id, label_name in label_map.items():
        mask = labels == label_id
        if mask.any():
            ax.scatter(coords[mask, 0], coords[mask, 1], label=label_name, alpha=0.7)
    ax.set_xlabel("PCA Component 1")
    ax.set_ylabel("PCA Component 2")
    ax.set_title("BERT Embeddings (test set)")
    ax.legend()
    st.pyplot(fig)

    if var_explained is not None:
        st.caption(
            f"These 2 components explain {var_explained.sum() * 100:.1f}% of the "
            f"variance in the original 768-dimensional BERT embeddings."
        )


def page3():
    st.title("Model Evaluation")
    st.caption("Metrics computed on the held-out test set during training.")

    if not os.path.exists("eval_results.pkl"):
        st.warning(
            "eval_results.pkl not found. Re-run the latest train_model.py — "
            "it now saves this file automatically — then copy it next to app.py."
        )
        return

    eval_results = joblib.load("eval_results.pkl")
    label_map = eval_results["label_map"]
    class_names = [label_map[i] for i in sorted(label_map)]

    model_choice = st.radio("Model", ["log_reg", "svm"], format_func=lambda m: "Logistic Regression" if m == "log_reg" else "SVM")
    result = eval_results[model_choice]

    col1, col2 = st.columns(2)
    col1.metric("Accuracy", f"{result['accuracy']:.3f}")
    col2.metric("ROC-AUC (macro, OvR)", f"{result['roc_auc']:.3f}")

    st.subheader("Classification Report")
    report_df = pd.DataFrame(result["report"]).transpose()
    st.dataframe(report_df.round(3))

    st.subheader("Confusion Matrix")
    cm = np.array(result["cm"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax)
    st.pyplot(fig)


def page4():
    st.title("Emotion Detection App")
    st.caption("Enter a sentence and the model will predict the underlying emotion.")

    classifier = load_classifier()
    if classifier is None:
        st.error(
            "emotion_model.pkl not found. Run train_model.py first to train "
            "and save the model, then copy emotion_model.pkl next to app.py."
        )
        return

    tokenizer, bert_model = load_bert()

    user_input = st.text_area("Enter text:")
    if st.button("Predict"):
        if not user_input.strip():
            st.warning("Please enter some text first.")
        else:
            cleaned = clean_text(user_input)
            embedding = get_embedding(cleaned, tokenizer, bert_model)
            prediction = classifier.predict(embedding)[0]
            emotion_label = LABEL_MAP.get(prediction, str(prediction))

            st.subheader(f"Predicted Emotion: {emotion_label}")

            # Display corresponding emotion image if it exists.
            image_path = f"images/{emotion_label}.png"
            if os.path.exists(image_path):
                st.image(image_path, caption=emotion_label.capitalize(), use_container_width=True)
            else:
                st.info(f"(No image found at {image_path} — add one to show it here.)")

            # Show confidence breakdown if available.
            if hasattr(classifier, "predict_proba"):
                proba = classifier.predict_proba(embedding)[0]
                proba_dict = {LABEL_MAP.get(i, str(i)): float(p) for i, p in enumerate(proba)}
                st.bar_chart(proba_dict)


# ---------------- Navigation (this was missing before) ----------------

PAGES = {
    "Corpora Viewer": page1,
    "Embeddings": page2,
    "Model Evaluation": page3,
    "Emotion Detection": page4,
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[selection]()

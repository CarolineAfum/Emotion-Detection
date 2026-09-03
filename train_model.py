# ============================
# train_model.py
# ============================
import os
import re

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC
from transformers import BertModel, BertTokenizer

# ============================
# 1. Load Dataset
# ============================
DATA_DIR = "Hugging_Face_Dataset"


def load_split(name: str) -> pd.DataFrame:
    local_path = os.path.join(DATA_DIR, f"{name}-00000-of-00001.parquet")
    if os.path.exists(local_path):
        return pd.read_parquet(local_path)

    print(f"Local file not found at {local_path!r}; downloading dair-ai/emotion "
          f"split '{name}' from the Hugging Face Hub instead.")
    from datasets import load_dataset

    split_name = "validation" if name == "validation" else name
    ds = load_dataset("dair-ai/emotion", split=split_name)
    return ds.to_pandas()


train_df = load_split("train")
test_df = load_split("test")
val_df = load_split("validation")

# dair-ai/emotion label mapping (0-5)
LABEL_MAP = {0: "sad", 1: "happy", 2: "love", 3: "angry", 4: "fear", 5: "surprised"}


# ============================
# 2. Clean Text
# ============================
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-zA-Z\s]", "", text)       # remove special characters/numbers
    return text.lower().strip()


for df in (train_df, test_df, val_df):
    df["clean_text"] = df["text"].apply(clean_text)


# ============================
# 3. Exploratory Analysis
# ============================
def page1():
    print(train_df[["text", "clean_text"]].head())
    train_df["label"].map(LABEL_MAP).value_counts().plot(kind="bar", color="skyblue")
    plt.title("Emotion Distribution (Training Set)")
    plt.xlabel("Emotion")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("emotion_distribution.png")
    plt.close()
    print("Saved emotion_distribution.png")


# This was defined but never called before, so app.py's Corpora Viewer page
# had nothing to display. Calling it here fixes that.
page1()

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

# ============================
# 4. Text Representation (BERT Embeddings)
# ============================
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")
bert_model.eval()


def get_embeddings(text_list, batch_size=32):
    """Batch the embedding calls so this scales past small samples."""
    all_embeddings = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i : i + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
        all_embeddings.append(outputs.last_hidden_state.mean(dim=1).numpy())
    return np.vstack(all_embeddings)


# NOTE: using a subset for speed. Increase these if training time allows.
N_TRAIN = 500
N_TEST = 200

X_train = get_embeddings(train_df["clean_text"].tolist()[:N_TRAIN])
y_train = train_df["label"].tolist()[:N_TRAIN]

X_test = get_embeddings(test_df["clean_text"].tolist()[:N_TEST])
y_test = test_df["label"].tolist()[:N_TEST]

# ============================
# 5. Model Building
# ============================
log_reg = LogisticRegression(max_iter=1000).fit(X_train, y_train)
svm_clf = SVC(kernel="linear", probability=True).fit(X_train, y_train)

# ============================
# 6. Model Evaluation
# ============================
def page3():
    print("Logistic Regression Report:")
    print(classification_report(y_test, log_reg.predict(X_test)))

    print("SVM Report:")
    print(classification_report(y_test, svm_clf.predict(X_test)))

    print("Confusion Matrix (Logistic Regression):")
    print(confusion_matrix(y_test, log_reg.predict(X_test)))

    print("Confusion Matrix (SVM):")
    print(confusion_matrix(y_test, svm_clf.predict(X_test)))

page3()

# ============================
# 6b. Save evaluation results + an embeddings demo for the Streamlit app
# ============================
y_test_bin = label_binarize(y_test, classes=sorted(LABEL_MAP.keys()))


def compute_roc_auc(model, X, y_true_bin):
    """Macro-average one-vs-rest ROC-AUC across all 6 emotion classes."""
    proba = model.predict_proba(X)
    return roc_auc_score(y_true_bin, proba, average="macro", multi_class="ovr")


eval_results = {
    "log_reg": {
        "accuracy": accuracy_score(y_test, log_reg.predict(X_test)),
        "roc_auc": compute_roc_auc(log_reg, X_test, y_test_bin),
        "report": classification_report(y_test, log_reg.predict(X_test), output_dict=True),
        "cm": confusion_matrix(y_test, log_reg.predict(X_test)),
    },
    "svm": {
        "accuracy": accuracy_score(y_test, svm_clf.predict(X_test)),
        "roc_auc": compute_roc_auc(svm_clf, X_test, y_test_bin),
        "report": classification_report(y_test, svm_clf.predict(X_test), output_dict=True),
        "cm": confusion_matrix(y_test, svm_clf.predict(X_test)),
    },
    "label_map": LABEL_MAP,
}
joblib.dump(eval_results, "eval_results.pkl")
print("Saved evaluation results to eval_results.pkl")
print(f"Logistic Regression -> Accuracy: {eval_results['log_reg']['accuracy']:.3f}, "
      f"ROC-AUC (macro OvR): {eval_results['log_reg']['roc_auc']:.3f}")
print(f"SVM -> Accuracy: {eval_results['svm']['accuracy']:.3f}, "
      f"ROC-AUC (macro OvR): {eval_results['svm']['roc_auc']:.3f}")

# Same story for embeddings - nothing was ever saved for the "Embeddings"
# page to show. Reduce the test-set embeddings to 2D with PCA so the app
# can plot how BERT separates the emotion classes.
pca = PCA(n_components=2, random_state=42)
coords_2d = pca.fit_transform(X_test)
embeddings_demo = {
    "coords": coords_2d,
    "labels": np.array(y_test),
    "label_map": LABEL_MAP,
    "explained_variance": pca.explained_variance_ratio_,
}
joblib.dump(embeddings_demo, "embeddings_demo.pkl")
print("Saved embeddings_demo.pkl")

# ============================
# 7. Save the trained model
# ============================
# Pick whichever model performed best on the report above.
BEST_MODEL = log_reg  # swap to svm_clf if SVM scores higher

joblib.dump(BEST_MODEL, "emotion_model.pkl")
print("Saved trained model to emotion_model.pkl")
print("Now copy emotion_model.pkl, emotion_distribution.png, eval_results.pkl, and")
print("embeddings_demo.pkl into the same folder as app.py before deploying.")

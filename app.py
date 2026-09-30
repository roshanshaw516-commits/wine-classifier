import pandas as pd
import streamlit as st
from sklearn.datasets import load_wine
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

st.set_page_config(page_title="Wine Classifier", page_icon="🍷")


@st.cache_resource
def load_model():
    """Train once and reuse on every interaction (Streamlit reruns the script on each click)."""
    wine = load_wine()
    X = pd.DataFrame(wine.data, columns=wine.feature_names)
    y = wine.target
    # Pipeline = scaler + model together, so raw user input is scaled correctly at predict time
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", probability=True, random_state=42))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv)
    model.fit(X, y)
    return model, X, wine.target_names, scores


model, X, class_names, cv_scores = load_model()

st.title("🍷 Wine Cultivar Classifier")
st.write(
    "Set the 13 chemical measurements in the sidebar and the model predicts which of the "
    "3 cultivars the wine comes from. Trained on scikit-learn's Wine dataset (178 samples)."
)
st.caption(f"Model: SVM (RBF) · 5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

st.sidebar.header("Chemical measurements")
values = {}
for col in X.columns:
    lo, hi, mean = float(X[col].min()), float(X[col].max()), float(X[col].mean())
    values[col] = st.sidebar.slider(col.replace("_", " ").title(), lo, hi, mean)

row = pd.DataFrame([values])
pred = model.predict(row)[0]
proba = model.predict_proba(row)[0]

st.subheader(f"Prediction: {class_names[pred]}")
st.bar_chart(pd.DataFrame({"Probability": proba}, index=class_names))

with st.expander("Show input values"):
    st.dataframe(row.T.rename(columns={0: "value"}))

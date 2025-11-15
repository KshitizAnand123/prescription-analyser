# ai_models/retrain_text_classifier.py
import pandas as pd, joblib, os, argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.calibration import CalibratedClassifierCV
from ai_models.preprocessing import normalize_text

parser = argparse.ArgumentParser()
parser.add_argument("--csv", default="ai_models/text_classification_data_large.csv")
parser.add_argument("--outdir", default="ai_models")
args = parser.parse_args()
os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.csv)
df['text'] = df['text'].astype(str).apply(normalize_text)

X = df['text']; y = df['category']
# simple split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y if len(set(y)) <= len(y)/2 else None)

vectorizer = TfidfVectorizer(stop_words='english', ngram_range=(1,3), max_features=20000, min_df=2)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

base = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
model = CalibratedClassifierCV(base, cv=3)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
print("Classification report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

joblib.dump(model, os.path.join(args.outdir, "text_classifier_model.joblib"))
joblib.dump(vectorizer, os.path.join(args.outdir, "tfidf_vectorizer.joblib"))
print("Saved model & vectorizer to", args.outdir)

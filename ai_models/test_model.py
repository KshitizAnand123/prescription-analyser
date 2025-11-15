# ai_models/test_model.py  (debug version)
import joblib, os, sys, traceback

MODEL_PATH = os.path.join("ai_models", "text_classifier_model.joblib")
VEC_PATH = os.path.join("ai_models", "tfidf_vectorizer.joblib")

print("cwd:", os.getcwd())
print("MODEL_PATH exists?", os.path.exists(MODEL_PATH))
print("VEC_PATH exists?", os.path.exists(VEC_PATH))

try:
    vec = joblib.load(VEC_PATH)
    print("Loaded vectorizer type:", type(vec))
except Exception as e:
    print("Failed to load vectorizer:", repr(e))
    traceback.print_exc()
    sys.exit(1)

try:
    model = joblib.load(MODEL_PATH)
    print("Loaded model type:", type(model))
except Exception as e:
    print("Failed to load model:", repr(e))
    traceback.print_exc()
    sys.exit(1)

print("Model classes:", getattr(model, "classes_", None))

# Paste the OCR text you copied from the app terminal below (between triple quotes)
ocr_text = """BLOOD TEST REPORT
Patient Name: Rahul Sharma Age: 28 years
Gender: Male Date: 12/11/2025
Test Results:

Fasting Blood Sugar: 88 mg/dL
Postprandial Blood Sugar: 118 mg/dL
HbAIc: 5.4%

Total Cholesterol 168 mg/dL

LDL Cholesterol 92 mg/dL
HDL Cholesterol 50 mg/dL
Triglycerides 130 mg/dL
Blood Pressure: 118/76 mmHg
Hemoglobin 14.5 g/dL

TSH 2.5 pIU/mL

T3 110 ng/dL

T4 8.2 pg/dL
Interpretation: All parameters are within the normal
reference range. The patient shows no indication of
diabetes, thyroid disorder, or cardiovascular risk.
Recommendation: Continue balanced diet and
regular physical activity. Routine follow-up after 1 year."""

if not ocr_text or len(ocr_text.strip()) == 0:
    print("WARNING: ocr_text is empty. Please paste the OCR output between the triple quotes in this file and re-run.")
    sys.exit(1)

print("OCR text length:", len(ocr_text))
print("OCR text preview (first 400 chars):")
print(ocr_text[:400])
print("----- running vectorizer.transform and model.predict_proba -----")

try:
    X = vec.transform([ocr_text])
    print("Vectorized shape:", getattr(X, "shape", None))
except Exception as e:
    print("Vectorizer.transform failed:", repr(e))
    traceback.print_exc()
    sys.exit(1)

try:
    probs = model.predict_proba(X)[0]
    pred = model.predict(X)[0]
    classes = list(model.classes_)
    print("Predicted:", pred)
    print("Per-class probabilities (sorted):")
    pairs = sorted(zip(classes, probs), key=lambda x: -x[1])
    for c,p in pairs:
        print(f"  {c}: {p:.4f}")
except Exception as e:
    print("Model predict failed:", repr(e))
    traceback.print_exc()
    sys.exit(1)

print("Done.")

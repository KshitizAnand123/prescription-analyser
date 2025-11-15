# app.py  (OCR preprocessing + AI integration)
import os
import re
import pdfplumber
from word2number import w2n

from PIL import Image
import pytesseract
import speech_recognition as sr
from pydub import AudioSegment
from pdf2image import convert_from_path
from flask import Flask, render_template, request, redirect, url_for, flash

# NEW: joblib for loading trained model & vectorizer
import joblib

# NEW: import normalizer (ai_models must be a package/folder)
from ai_models.preprocessing import normalize_text

# NEW: OpenCV + numpy for preprocessing
import numpy as np
import cv2

pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['SECRET_KEY'] = '1f0f141754759101a4eb5992bd424010'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------- AI MODEL LOADING ----------
MODEL_DIR = os.path.join(os.path.dirname(__file__), "ai_models")
MODEL_PATH = os.path.join(MODEL_DIR, "text_classifier_model.joblib")
VEC_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.joblib")

model = None
model_loaded = False

vectorizer = None
classes = []

def try_load_models():
    global model, vectorizer, classes, model_loaded
    model_loaded = False
    if os.path.exists(MODEL_PATH) and os.path.exists(VEC_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VEC_PATH)
            classes = list(model.classes_)
            model_loaded = True
            print("Loaded text classifier & vectorizer from ai_models/")
        except Exception as e:
            print("Error loading AI models:", e)
            model = None
            vectorizer = None
            model_loaded = False
    else:
        print("AI model files not found in ai_models/. Classifier disabled until models exist.")
        model_loaded = False



# Attempt load at startup

try_load_models()

# ---------- Image preprocessing helper ----------
def preprocess_image_for_ocr(pil_img):
    """
    Input: PIL.Image
    Output: PIL.Image (preprocessed for tesseract)
    Steps: grayscale -> optional upscale -> denoise -> adaptive threshold -> morphological open
    """
    # convert to grayscale numpy array
    img = np.array(pil_img.convert('L'))

    # upscale small images (helps tesseract)
    h, w = img.shape
    scale = 2 if max(h, w) < 1500 else 1
    if scale != 1:
        img = cv2.resize(img, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_LINEAR)

    # denoise while preserving edges
    img = cv2.bilateralFilter(img, 9, 75, 75)

    # adaptive thresholding for uneven illumination
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, 15, 9)

    # small morphological opening to remove specks
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    # return back as PIL image
    return Image.fromarray(img)

# ---------- Helper Functions (unchanged logic) ----------

def extract_text_from_file(filepath, filename):
    """Extract text from PDF, image, or audio files."""
    text = ""
    custom_config = r'--oem 3 --psm 6 -l eng'  # general config for tesseract

    if filename.lower().endswith('.pdf'):
        with pdfplumber.open(filepath) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        # fallback: convert PDF pages to images and OCR them
        if not text.strip():
            images = convert_from_path(filepath)
            for img in images:
                proc = preprocess_image_for_ocr(img)
                text_piece = pytesseract.image_to_string(proc, config=custom_config)
                text += text_piece + "\n"

    elif filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(filepath)
        proc = preprocess_image_for_ocr(img)
        text = pytesseract.image_to_string(proc, config=custom_config)

    elif filename.lower().endswith(('.mp3', '.wav', '.m4a')):
        recognizer = sr.Recognizer()
        if filename.lower().endswith(('.mp3', '.m4a')):
            sound = AudioSegment.from_file(filepath)
            filepath = filepath.rsplit('.', 1)[0] + '.wav'
            sound.export(filepath, format="wav")
        with sr.AudioFile(filepath) as source:
            audio = recognizer.record(source)
            text = recognizer.recognize_google(audio, language='en-IN')

    return text


def extract_numeric_values(text):
    """Extract key lab values (HbA1c, BP, TSH, Hemoglobin) using flexible regex and natural phrasing support."""
    import re
    from word2number import w2n

    values = {}
    # normalize and fix common OCR misreads before parsing numbers
    text_lower = text.lower()

    # Fix common OCR confusions for HbA1c (I / l / i ↔ 1)
    # Covers: "HbAIc", "hbaic", "hb aic", "hb a1c", "ha1c", etc.
    text_lower = re.sub(r'hb\s*a[iil]c', 'hba1c', text_lower)   # 'hb aic', 'hb ai c', 'hb ailc' -> 'hba1c'
    text_lower = re.sub(r'hba[il]c', 'hba1c', text_lower)       # 'hbaic', 'hbailc' -> 'hba1c'
    text_lower = re.sub(r'\bhaic\b', 'hba1c', text_lower)       # 'haic' -> 'hba1c'
    text_lower = re.sub(r'\ba[iil]c\b', 'a1c', text_lower)      # 'aic'/'alc' -> 'a1c'

    # Fix units and common token merges
    text_lower = re.sub(r'\bmgdl\b', ' mg/dl', text_lower)      # 'mgdl' -> 'mg/dl'
    text_lower = re.sub(r'\bgdl\b', ' g/dl', text_lower)        # 'gdl' -> 'g/dl'
    text_lower = re.sub(r'\bmmhg\b', ' mmhg', text_lower)       # ensure 'mmhg' token exists
    # Fix µ/ micro unit misreads like 'piu/ml' -> 'µiu/ml' is optional; ensure digits preserved
    text_lower = re.sub(r'piu\/ml', ' µiu/ml', text_lower)

    # (Optional) collapse weird spaces within tokens like 'h b a 1 c'
    text_lower = re.sub(r'h\s*b\s*a\s*1\s*c', 'hba1c', text_lower)


    # --- Step 1: Convert number words to digits where possible ---
    words = text_lower.split()
    for i, word in enumerate(words):
        try:
            num = w2n.word_to_num(word)
            words[i] = str(num)
        except:
            continue
    text_lower = " ".join(words)

    # --- Step 2: HbA1c / Sugar / Glucose detection ---
    if any(k in text_lower for k in ["hba1c", "sugar", "glucose", "a1c"]):
        match = re.search(r"(hba1c|hb a1c|a1c|sugar|glucose)(?:\s+[:\-]?\s*\w*){0,3}\s*([0-9]+(?:\.[0-9]+)?)", text_lower)
        if match:
            try:
                values["HbA1c"] = float(match.group(2))
            except:
                pass

    # --- Step 3: Blood Pressure detection ---
    bp_match = None

    # Pattern 1: Numeric form (120/80 or 120-80)
    bp_match = re.search(r"(\d{2,3})[\/\-](\d{2,3})", text_lower)

    # Pattern 2: Spoken form (120 over 80)
    if not bp_match:
        bp_match = re.search(r"(\d{2,3})\s*(?:over|by)\s*(\d{2,3})", text_lower)

    # Pattern 3: Partial BP (systolic or diastolic)
    if not bp_match:
        sys_match = re.search(r"(?:bp|pressure|systolic)(?:\s+\w+){0,3}\s*([0-9.]+)", text_lower)
        dia_match = re.search(r"(?:diastolic)(?:\s+\w+){0,3}\s*([0-9.]+)", text_lower)
        if sys_match and dia_match:
            try:
                values["BP"] = (int(float(sys_match.group(1))), int(float(dia_match.group(1))))
            except:
                pass
        elif sys_match:
            try:
                values["BP"] = (int(float(sys_match.group(1))), None)
            except:
                pass
        elif dia_match:
            try:
                values["BP"] = (None, int(float(dia_match.group(1))))
            except:
                pass

    # Pattern 4: Paired BP (two separate numbers)
    if not bp_match:
        pair = re.findall(r"\b(\d{2,3})\b", text_lower)
        if len(pair) >= 2 and ("bp" in text_lower or "pressure" in text_lower):
            try:
                values["BP"] = (int(pair[0]), int(pair[1]))
            except:
                pass
    elif bp_match:
        try:
            values["BP"] = (int(bp_match.group(1)), int(bp_match.group(2)))
        except:
            pass

    # --- Step 4: Thyroid levels (TSH, T3, T4) ---
    for hormone in ["tsh", "t3", "t4"]:
        match = re.search(fr"{hormone}(?:\s+\w+){{0,3}}\s*([0-9]+(?:\.[0-9]+)?)", text_lower)
        if match:
            try:
                values[hormone.upper()] = float(match.group(1))
            except:
                pass

    # --- Step 5: Hemoglobin ---
    match = re.search(r"(hemoglobin|hb)(?:\s+[^\d]{0,6})?([0-9]+(?:\.[0-9]+)?)", text_lower)
    if match:
        try:
            values["Hemoglobin"] = float(match.group(2))
        except:
            pass

    return values


def generate_professional_advice(values):
    """Generate contextual, professional-grade advice based on detected values."""
    advice = []

    # Diabetes check
    if "HbA1c" in values:
        h = values["HbA1c"]
        if h < 5.7:
            advice.append("✅ HbA1c is within the normal range. Maintain a balanced diet and regular exercise.")
        elif 5.7 <= h < 6.5:
            advice.append("⚠️ Prediabetic range detected. Consider lifestyle adjustments and recheck in 3 months.")
        else:
            advice.append("❗ High HbA1c indicates diabetes. Consult an endocrinologist for medication and diet planning.")

    # Blood Pressure check
    if "BP" in values:
        sys, dia = values["BP"]
        try:
            sys_v = sys if sys is not None else 0
            dia_v = dia if dia is not None else 0
        except:
            sys_v, dia_v = 0, 0
        if sys_v < 120 and dia_v < 80:
            advice.append("✅ Blood pressure is optimal. Maintain healthy habits.")
        elif sys_v < 140 and dia_v < 90:
            advice.append("⚠️ Slightly elevated BP detected. Reduce salt intake and monitor regularly.")
        else:
            advice.append("❗ Hypertension suspected. Consult a cardiologist and track BP daily.")

    # Thyroid check
    if "TSH" in values:
        tsh = values["TSH"]
        if tsh < 0.4:
            advice.append("⚠️ Low TSH levels suggest hyperthyroidism. Seek endocrinology consultation.")
        elif tsh > 4.0:
            advice.append("❗ Elevated TSH suggests hypothyroidism. Medication adjustment may be required.")
        else:
            advice.append("✅ TSH is within the normal range.")

    # Hemoglobin check
    if "Hemoglobin" in values:
        hb = values["Hemoglobin"]
        if hb < 12:
            advice.append("❗ Low hemoglobin detected. Possible anemia — increase iron intake and visit a hematologist.")
        elif hb > 17:
            advice.append("⚠️ High hemoglobin levels detected — ensure proper hydration and consult your doctor.")
        else:
            advice.append("✅ Hemoglobin is in the healthy range.")

    if not advice:
        advice.append("ℹ️ No key lab markers detected. Please upload a clearer or more detailed report.")

    return "\n".join(advice)


# ---------- AI classification helper ----------
def classify_text_block(full_text, min_confidence=0.60, debug=True):
    """
    Improved classification:
      - uses document-level and line-level predictions and averages them
      - uses numeric overrides (HbA1c, BP, Hemoglobin) when clear signals exist
      - returns (predicted_category_or_None, confidence_float_or_None)
    """
    global model, vectorizer, classes
    if not model or not vectorizer:
        return None, None

    import numpy as np

    # Normalize full text and lines
    normalized_full = normalize_text(full_text)
    lines = [l.strip() for l in re.split(r'[\r\n]+', full_text) if l.strip()]
    if not lines:
        lines = [full_text]
    normalized_lines = [normalize_text(s) for s in lines]

    # -------------------------
    # Numeric overrides (strong signals)
    # -------------------------
    try:
        numeric_vals = extract_numeric_values(full_text)
    except Exception:
        numeric_vals = {}

    # If HbA1c clearly diabetic -> force Diabetes with very high confidence
    if "HbA1c" in numeric_vals:
        try:
            a1c = float(numeric_vals["HbA1c"])
            if a1c >= 6.5:
                if debug:
                    print(f"[CLASSIFIER OVERRIDE] HbA1c={a1c} >=6.5 => Diabetes (forced).")
                return "Diabetes", 0.99
            if a1c < 5.7:
                # strong indicator of normal glycemic control
                # but don't force full normal: we'll increase confidence for 'General'
                override_general = True
            else:
                override_general = False
        except Exception:
            override_general = False
    else:
        override_general = False

    # If BP strongly hypertensive -> increase weight for Blood Pressure
    bp_override = None
    if "BP" in numeric_vals:
        try:
            sys, dia = numeric_vals["BP"]
            if sys is None:
                sys = 0
            if dia is None:
                dia = 0
            if sys >= 140 or dia >= 90:
                bp_override = ("Blood Pressure", 0.95)
                if debug:
                    print(f"[CLASSIFIER OVERRIDE] BP={sys}/{dia} => Blood Pressure (forced-ish).")
        except:
            bp_override = None

    # -------------------------
    # Model-based predictions
    # -------------------------
    try:
        # Document-level
        X_doc = vectorizer.transform([normalized_full])
        doc_probs = np.asarray(model.predict_proba(X_doc)[0])  # shape (n_classes,)
    except Exception as e:
        if debug:
            print("Error in doc-level prediction:", e)
        return None, None

    # Line-level: predict for each line, get per-class mean
    try:
        X_lines = vectorizer.transform(normalized_lines)
        line_probs = np.asarray(model.predict_proba(X_lines))  # shape (n_lines, n_classes)
        # average across lines (gives weight to repeated signals across lines)
        mean_line_probs = line_probs.mean(axis=0)
    except Exception as e:
        if debug:
            print("Error in line-level prediction:", e)
        mean_line_probs = np.zeros_like(doc_probs)

    # Combine document and line signals (simple average)
    combined_probs = (doc_probs + mean_line_probs) / 2.0
    # Normalize (should already sum to 1-ish but ensure numeric safety)
    combined_probs = combined_probs / (combined_probs.sum() + 1e-12)

    # pick predicted class and confidence
    best_idx = int(np.argmax(combined_probs))
    predicted = classes[best_idx]
    confidence = float(combined_probs[best_idx])

    # Debug prints: show per-class probs for doc, line, combined
    if debug:
        def show_probs(name, probs_arr):
            pairs = sorted(zip(classes, [float(x) for x in probs_arr]), key=lambda x: -x[1])
            print(f"[{name} top3] " + ", ".join([f"{p[0]}:{p[1]:.2f}" for p in pairs[:3]]))
        print("---- CLASSIFIER DEBUG ----")
        print("Lines (first 6):")
        for i, ln in enumerate(normalized_lines[:6]):
            print(f" L{i}: {ln}")
        show_probs("DOC", doc_probs)
        show_probs("LINE_MEAN", mean_line_probs)
        show_probs("COMBINED", combined_probs)
        print("---------------------------")

    # Apply numeric overrides if present and stronger than combined_probs
    if bp_override:
        # if BP override exists and combined does not already strongly contradict, use it
        ov_label, ov_conf = bp_override
        ov_idx = classes.index(ov_label) if ov_label in classes else None
        if ov_idx is not None and ov_conf > combined_probs[ov_idx]:
            if debug:
                print(f"[APPLY OVERRIDE] Using BP override {ov_label} {ov_conf}")
            return ov_label, float(ov_conf)

    if override_general:
        # If HbA1c indicates normal, bump General if it's in classes
        if "General" in classes:
            gen_idx = classes.index("General")
            # if combined General prob < 0.5 but HbA1c is normal, increase it
            bumped_conf = max(combined_probs[gen_idx], 0.85)
            if debug:
                print(f"[APPLY OVERRIDE] HbA1c normal -> boosting 'General' to {bumped_conf}")
            return "General", float(bumped_conf)

    # Finally apply confidence threshold: if below min_confidence, return None
    if confidence < min_confidence:
        if debug:
            print(f"[LOW CONFIDENCE] best={predicted} conf={confidence:.2f} < min_conf={min_confidence}")
        return None, None

    return predicted, confidence



# ---------- Routes ----------

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        flash("No file part found.")
        return redirect(url_for('home'))

    file = request.files['file']
    if file.filename == '':
        flash("No selected file.")
        return redirect(url_for('home'))

    filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        extracted_text = extract_text_from_file(filepath, filename)

        # DEBUG: print OCR output to console for inspection (remove in production)
        print("\n--- OCR BEGIN ---")
        print(extracted_text)
        print("--- OCR END ---\n")

        # ---- NEW: run classifier on extracted_text (Phase 1) ----
        predicted_category, confidence = classify_text_block(extracted_text)

        # ---- Existing numeric extraction and advice ----
        values = extract_numeric_values(extracted_text)
        advice = generate_professional_advice(values)

        return render_template(
            'result.html',
            filename=filename,
            extracted_text=extracted_text,
            advice=advice,
            predicted_category=predicted_category,
            predicted_confidence=confidence,
            model_loaded=model_loaded
        )


    except Exception as e:
        flash(f"Error processing file: {e}")
        return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)

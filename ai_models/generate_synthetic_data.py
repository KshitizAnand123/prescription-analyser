# ai_models/generate_synthetic_data.py
import csv, random
templates = {
 "Diabetes": [
    "Fasting Blood Sugar: {fs} mg/dL\nPostprandial Blood Sugar: {pp} mg/dL\nHbA1c: {a1c}%",
    "Fasting sugar is {fs} mg/dL; postprandial {pp} mg/dL; HbA1c {a1c}%"
 ],
 "Blood Pressure": [
    "Blood Pressure: {sys}/{dia} mmHg",
    "BP recorded as {sys} over {dia}"
 ],
 "Liver": [
    "ALT: {alt} U/L\nAST: {ast} U/L\nBilirubin: {bil} mg/dL",
 ],
 "Cardiac": [
    "LDL Cholesterol: {ldl} mg/dL\nECG: {ecg}",
 ],
 "General": [
    "Routine health check: no significant abnormality",
    "General checkup normal; vitals stable"
 ]
}

def sample_vals(label):
    if label=="Diabetes":
        return {"fs": random.randint(70,260), "pp": random.randint(90,320), "a1c": round(random.uniform(4.5,10.5),1)}
    if label=="Blood Pressure":
        return {"sys": random.randint(100,180), "dia": random.randint(60,110)}
    if label=="Liver":
        return {"alt": random.randint(20,200), "ast": random.randint(15,180), "bil": round(random.uniform(0.3,4.0),1)}
    if label=="Cardiac":
        return {"ldl": random.randint(60,240), "ecg": random.choice(["normal", "arrhythmia", "ischemic changes"])}
    return {}

rows = []
for label in templates:
    for i in range(300):  # 300 samples per class
        vals = sample_vals(label)
        t = random.choice(templates[label]).format(**vals)
        # Add header lines to mimic report
        t = f"Patient Name: Test User\nDate: 12/11/2025\n{t}\n"
        rows.append([t, label])

with open("ai_models/text_classification_data_large.csv", "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["text","category"])
    writer.writerows(rows)

print("Wrote ai_models/text_classification_data_large.csv with", len(rows), "rows.")

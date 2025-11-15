# ai_models/augment_general_data.py
import csv

new_rows = [
    ["HbA1c: 5.4% (within normal range). Hemoglobin: 14.5 g/dL. All results normal.", "General"],
    ["Blood Pressure: 118/76 mmHg. Sugar normal. No abnormal findings detected.", "General"],
    ["All parameters within normal range. Routine health check normal.", "General"],
    ["Normal report: HbA1c 5.3%, BP 120/80, Hemoglobin 14.2. Patient healthy.", "General"],
    ["Healthy individual. No signs of diabetes or hypertension.", "General"],
]

with open("ai_models/text_classification_data_large.csv", "a", newline='', encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerows(new_rows)

print("✅ Added", len(new_rows), "new 'General' examples.")

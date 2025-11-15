from pathlib import Path
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:/Program Files/Tesseract-OCR/tesseract.exe"

directory = 'C:/Users/hp/OneDrive/Pictures/Screenshots 1'
files = Path(directory).glob('*.png')
for file in files:
     print(file)
     print(pytesseract.image_to_string(Image.open(file)))
     print("\n------------------------\n")
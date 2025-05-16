import docx
import json

# Load the Word document
doc = docx.Document('/Users/rodrigogarciaresendiz/new_pressroom/data/Clasificación_Notas_97_VALIDADO_FINAL.docx')

# Extract text from paragraphs
data = [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]

# Save as JSON
with open('train.json', 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("Conversion completed. JSON file created as 'train.json'.") 
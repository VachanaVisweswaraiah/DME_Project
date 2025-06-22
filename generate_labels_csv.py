import os
import pandas as pd

# Path to your images folder
image_dir = 'train'
output_csv = "labels.csv"

# Mapping prefixes to class labels
prefix_to_label = {
    "m": "Mild",
    "d": "Moderate",
    "n": "Normal",
    "p": "Proliferate",
    "s": "Severe"
}

# List image files and create label entries
data = []
for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        prefix = filename[0].lower()
        label = prefix_to_label.get(prefix)
        if label:
            data.append({"filename": filename, "label": label})

# Save to CSV
df = pd.DataFrame(data)
df.to_csv(output_csv, index=False)
print(f"✅ Created {output_csv} with {len(df)} entries.")

import csv
import os
import requests

# Input CSV path and output folder
csv_file = r"C:\Users\camposmo\Downloads\sbb_tiffs.csv"
output_dir = r"C:\Users\camposmo\Downloads\swissalti3d_tiles"
os.makedirs(output_dir, exist_ok=True)

# Load and extract URLs
with open(csv_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f)
    urls = [row[0].strip() for row in reader if row and row[0].startswith("https://")]

print(f"Found {len(urls)} URLs. Starting download...\n")

# Download each file
for i, url in enumerate(urls, 1):
    filename = os.path.join(output_dir, os.path.basename(url))
    try:
        print(f"[{i}/{len(urls)}] Downloading {os.path.basename(url)}...")
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(filename, 'wb') as f_out:
                for chunk in r.iter_content(chunk_size=8192):
                    f_out.write(chunk)
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")

print("\n✅ Download complete.")

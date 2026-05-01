import json
from pathlib import Path
from science_parse_api.api import parse_pdf

host = "http://127.0.0.1"
port = "8080"
pdf_folder = Path("acl_papers_2025")          # folder containing all PDFs
output_folder = Path("acl_json_files")        # folder for JSON files
output_folder.mkdir(exist_ok=True)

pdf_files = list(pdf_folder.glob("*.pdf"))

if not pdf_files:
    print(" No PDFs found in", pdf_folder)
else:
    print(f" Found {len(pdf_files)} PDFs in '{pdf_folder}'")

for pdf_file in pdf_files:
    json_path = output_folder / (pdf_file.stem + ".json")

    #  Skip already processed files
    if json_path.exists():
        print(f"  Skipping {pdf_file.name}, already converted.")
        continue

    print(f"  Processing: {pdf_file.name}")
    try:
        parsed_data = parse_pdf(host, pdf_file, port=port)

        if parsed_data and any(parsed_data.values()):
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(parsed_data, f, indent=2, ensure_ascii=False)
            print(f" Saved as: {json_path.name}")
        else:
            print(f" No valid data parsed for: {pdf_file.name}")

    except Exception as e:
        print(f" Failed: {pdf_file.name} -> {e}")

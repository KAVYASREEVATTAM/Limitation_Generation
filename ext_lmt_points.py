import json
import os
from groq import Groq
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load the Groq API Key
GROQ_KEY = os.getenv("enter your key")

client = Groq(api_key=GROQ_KEY)

# Extract limitation points using Groq
def extract_limitations(text):

    prompt = f"""
You are extracting limitation points from a text.

The input already contains the limitations inside it. 
Your task is ONLY to convert the given text into a clear list of numbered limitation points.

STRICT RULES:
- Do NOT add new limitations.
- Do NOT infer anything outside the text.
- Do NOT summarize the paper.
- Only break down the provided limitation paragraph into limitation points.
- Output ONLY bullet points.

TEXT:
{text}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",   # ✔ ACTIVE MODEL
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=500
        )

        #  Correct message access
        return response.choices[0].message.content

    except Exception as e:
        return f"ERROR: {str(e)}"


# MAIN EXECUTION
input_file = "extract_limitations_taxonomy_test.json"       # <-- YOUR UPLOADED FILE
output_file = "extract_taxonomy_test_lmts_points.json" # <-- OUTPUT FILE

print(f"Reading: {input_file}")
with open(input_file, "r", encoding="utf-8") as f:
    papers = json.load(f)

results = []

print("\nExtracting limitation points...\n")

for paper in papers:
    title_short = paper["title"][:60] + "..."
    print("Processing:", title_short)

    text = paper.get("extracted_limitations")

    # If paper has no extracted text, skip
    if not text:
        results.append({
            "title": paper["title"],
            "paper_source": paper["paper_source"],
            "limitation_points": "NO LIMITATIONS FOUND"
        })
        continue

    lmts = extract_limitations(text)

    results.append({
        "title": paper["title"],
        "paper_source": paper["paper_source"],
        "limitation_points": lmts
    })

# Save output file
with open(output_file, "w", encoding="utf-8") as out:
    json.dump(results, out, ensure_ascii=False, indent=4)

print("\nDONE! Saved to:", output_file)

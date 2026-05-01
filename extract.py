import os
import json
import re

#  Path to your folder containing the JSON files
folder_path = r"data_files_taxonomy_test"  # <-- change this
output_file = os.path.join("extract_limitations_taxonomy_test.json")

# Regex pattern to match "limitation" or "limitations" (case-insensitive)
limitation_pattern = re.compile(r'\blimitations?\b', re.IGNORECASE)

# Regex pattern to match "conclusion" or "conclusions" (case-insensitive)
conclusion_pattern = re.compile(r'\bconclusions?\b', re.IGNORECASE)

# Regex pattern to match "acknowledgement" or "acknowledgements" (case-insensitive)
acknowledgement_pattern = re.compile(r'\backnowledgements?\b', re.IGNORECASE)

# Regex to find a "Limitations" sub-heading *inside* text.
# Matches: newline, optional whitespace, optional numbering (e.g., "V.", "5.1"), "limitation(s)"
limitation_subheading_pattern = re.compile(
    r'\n\s*(?:[IVX\d\.]+\s*)*\blimitations?\b', 
    re.IGNORECASE
)

# Regex to find the *next* sub-heading that would *end* a limitations section
# Matches: newline, optional whitespace, and common follow-up sections
next_subheading_pattern = re.compile(
    r'\n\s*(?:(?:future\s+work)|(?:acknowledgements?)|(?:appendix)|(?:conclusions?))\b',
    re.IGNORECASE
)

# List to store extracted results
all_results = []

# Loop through all files in the folder
for filename in os.listdir(folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(folder_path, filename)
        # Get paper_source from filename, e.g., "2025.realm-1.1"
        paper_source = os.path.splitext(filename)[0] 
        
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Handle files that are a list of papers or a single paper object
            papers = data if isinstance(data, list) else [data]

            # Process each paper in the file
            for paper in papers:
                if not isinstance(paper, dict):
                    print(f" Skipping non-dict item in {filename}")
                    continue

                # Get title from paper data
                title = paper.get("title", paper.get("Title", "Unknown Title"))
                
                sections = paper.get("sections", [])
                limitation_texts = [] # Store texts found for *this* paper

                for section in sections:
                    heading = section.get("heading", "")
                    text = section.get("text", "")
                    
                    if not text:
                        continue

                    heading_lower = heading.lower()

                    # Scenario 1: Section heading *is* "Limitations"
                    if limitation_pattern.search(heading_lower):
                        limitation_texts.append(text.strip())

                    # Scenario 2: Section heading is "Conclusion" OR "Acknowledgements"
                    elif conclusion_pattern.search(heading_lower) or acknowledgement_pattern.search(heading_lower):
                        # Now, search *inside* this section's text for a "Limitations" sub-section
                        parts = re.split(limitation_subheading_pattern, text, maxsplit=1)
                        
                        if len(parts) > 1:
                            # We found a limitations sub-section.
                            limitation_part = parts[1]
                            
                            next_heading_match = re.search(next_subheading_pattern, limitation_part)
                            
                            final_limitation_text = ""
                            if next_heading_match:
                                final_limitation_text = limitation_part[:next_heading_match.start()]
                            else:
                                final_limitation_text = limitation_part
                            
                            limitation_texts.append(final_limitation_text.strip())

                # Format the output for this specific paper
                extraction_method = "none"
                extracted_limitations = None
                
                if limitation_texts:
                    extraction_method = "structured_extract"
                    # Join all found limitation texts into one string
                    extracted_limitations = "\n\n".join(limitation_texts) 

                # Add the formatted result to our master list
                all_results.append({
                    "title": title,
                    "extraction_method": extraction_method,
                    "extracted_limitations": extracted_limitations,
                    "paper_source": paper_source
                })

        except Exception as e:
            print(f" Error reading {filename}: {e}")

# Save all extracted results into one JSON file
with open(output_file, "w", encoding="utf-8") as out:
    json.dump(all_results, out, ensure_ascii=False, indent=4)

print(f" All limitation sections extracted and saved to:\n{output_file}")

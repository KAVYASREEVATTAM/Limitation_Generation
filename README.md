# Limitation_Generation
# LLM Limitations Taxonomy — Dataset, Pipeline & Experiments 


This project builds and evaluates an automated **two-level taxonomy of research limitations** stated in NLP/LLM papers. The project has two components:

1. **Pipeline** (`limitation_taxonomy.py`): Starting from a cleaned set of 3,000 extracted limitation sentences, the pipeline uses GPT-based models to label, cluster, and hierarchically organize every limitation into a structured coarse-to-fine taxonomy (11 coarse categories, 36 fine labels). The result is a reproducible, richly annotated dataset useful for meta-research, bias auditing, and systematic review of LLM research.

2. **Experiments** (`taxonomy_experiments.py`): Evaluates the custom taxonomy against **LIMITGEN** — a four-aspect taxonomy from the literature — across four complementary experiments using a shared dataset of 500 annotated limitation sentences. Experiments measure coverage, distribution balance, generative quality via a GPT judge, and embedding-based cluster coherence.


Part 1: Taxonomy Pipeline

### Input: `cleaned_limitations.csv`

#### Description

This is the primary input file containing 3,000 individual limitation statements extracted from NLP/LLM research papers. Each row represents a single, cleaned limitation sentence extracted from a paper's "Limitations" section.


### Pipeline Script: `limitation_taxonomy.py`

#### Purpose

A self-contained, eight-step pipeline that transforms raw limitation sentences into a structured two-level taxonomy using the OpenAI API (GPT models + text embeddings). The pipeline is modular, supports disk caching at every step, and includes structural enforcement logic to guarantee a clean, consistent taxonomy.

#### Dependencies

```bash
pip install openai numpy pandas scipy scikit-learn tqdm
```

#### Environment

```bash
export OPENAI_API_KEY="your-key-here"

#### Pipeline Steps

The script executes eight sequential steps, each with disk caching:

**Step 1 — Extract Type Labels (`gpt-4o-mini`)**
Each limitation sentence is sent in batches of 20 to GPT with a prompt that extracts a short **constraint-type label** (3–6 words, e.g., `"single dataset evaluation"`, `"no human evaluation"`). These labels name the *kind* of limitation, not its topic. A maximum of 30% unspecified labels is enforced — if exceeded, the run fails.

**Step 2 — Embed Type Labels (`text-embedding-3-large`)**
All extracted type labels are embedded using OpenAI's text embedding model, producing a 3,000 × N embedding matrix. Embeddings are cached to `.npy` to avoid recomputation.

**Step 3 — Over-Cluster (Ward Hierarchical, n=24)**
Embeddings are L2-normalized, then clustered using Ward's method (cosine distance) into `over_n` (default 24) clusters. This deliberately over-segments the space to ensure no real distinction is lost before the semantic consolidation step.

**Step 4 — Semantic Consolidation (`gpt-4o`)**
The 24 over-clusters are shown to GPT-4o (with their top type labels and example sentences), which is asked to group them into exactly **10 coarse categories**. The prompt enforces canonical constraint dimension names (e.g., `Data Coverage`, `Evaluation Design`, `Computational Resources`) and forbids symptom-based names. A structural enforcement loop then forces merges if GPT returns more than the allowed maximum.

**Step 5 — Generate Fine Labels (`gpt-4o`)**
For each coarse category, GPT-4o generates 2–12 **fine-grained sub-labels** (Title Case, 2–5 words each). Each fine label includes a definition, a decision rule, a not-clause (to disambiguate from siblings), and a real example sentence drawn from the data.

**Step 6 — Verify Example Sentences**
Each generated example sentence is fuzzy-matched against real sentences in the corpus. If a close match is found (threshold 0.7), the verified sentence replaces the generated one and is flagged `_verified=True`.

**Step 7 — Global Distinctness Pass (`gpt-4o`)**
A taxonomy auditor prompt reviews the full taxonomy for three classes of issues:
- **Type A:** Within-category fine labels that may be confused with each other
- **Type B:** Cross-category overlaps with the same root cause (one is dropped and rows are reassigned)
- **Type C:** Duplicate fine label names across categories (renamed)

**Step 8 — Assign Fine Labels (`gpt-4o-mini`)**
Each of the 3,000 rows is assigned a fine label within its coarse category. GPT receives the sentence, the coarse label, and all fine label options with their decision rules and not-clauses, and returns the single best match.


Output Artifacts
limitations_annotated.csv: The fully annotated dataset mapping each of the 3,000 rows to its coarse_label and fine_label.  

taxonomy_coarse_fine.json: The authoritative JSON reference for the taxonomy's definitions, decision rules, example sentences, and cluster counts.  

taxonomy_table.csv: A flat, human-readable CSV representation of the taxonomy, optimized for annotation guidelines or spreadsheet analysis.



## Part 2: Evaluation Experiments
Input: `annotated_limitations_complete.csv`

#### Description

A dataset of **500 real limitation sentences** extracted from NLP/LLM research papers, each annotated under two taxonomies simultaneously: the custom taxonomy and LIMITGEN.


Experiment Script: `taxonomy_experiments.py`

#### Purpose

A self-contained Python script that runs four evaluation experiments comparing the custom taxonomy against LIMITGEN. It supports both GPT-powered evaluation (primary mode) and a TF-IDF fallback when no API key is provided, enabling reproducible results without OpenAI access.

#### Dependencies

```bash
pip install openai numpy pandas matplotlib scikit-learn sentence-transformers
```

#### Usage

```bash
# Full run with GPT evaluation (recommended)
python taxonomy_experiments.py \
  --csv annotated_limitations_complete.csv \
  --openai_key sk-... \
  --model gpt-4o-mini \
  --out experiment_outputs \
  --sample_size 30



### Experiments


Experiment 1: Coverage Rate:
Measures what fraction of real limitation sentences each taxonomy can confidently categorize without relying on "Other".  

Result: Custom Taxonomy achieved 100.0% coverage, while LIMITGEN achieved 95.6% in-data (with paper-reported out-of-scope rates as high as 36%).

Experiment 2: Distribution Faithfulness: 
Computes normalized Shannon entropy to evaluate how evenly sentences are distributed across categories.  

Result: Custom Taxonomy (0.862) significantly outperformed LIMITGEN (0.577), which largely collapsed limitations into just two buckets.  

Experiment 3: LLM Generation Quality: 
Tests whether a taxonomy's categories are specific enough to guide LLMs in generating realistic, human-like limitation sentences, using a GPT-judge to calculate Soft Recall, Precision, and Jaccard scores.  

Result: Custom Taxonomy outperformed LIMITGEN across all metrics, with "Bias and Representation" proving exceptionally effective for grounded generation.  

Experiment 4: Embedding-Based Coherence: 
Uses all-MiniLM-L6-v2 embeddings to measure intra-category similarity, inter-category separation, and Silhouette scores.  

Result: Custom Taxonomy exhibited higher intra-category coherence (0.266 vs 0.217) and a Silhouette score closer to 0 (-0.013 vs -0.037), indicating finer-grained and more meaningful clustering.



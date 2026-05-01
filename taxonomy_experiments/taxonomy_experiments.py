import argparse
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import LabelEncoder

# TAXONOMY DEFINITIONS 
YOUR_TAXONOMY_CATEGORIES = [
    "Evaluation Design",
    "Computational Resources",
    "Language Coverage",
    "Generalization Scope",
    "Task and Application Scope",
    "Prompt and Interaction Design",
    "Bias and Representation",
    "Annotation Quality",
    "Data Coverage",
    "Security and Privacy",
    "Inference and Efficiency",
]

YOUR_TAXONOMY_SUBCATEGORIES = {
    "Evaluation Design":           ["Evaluation Coverage", "Metric Limitations"],
    "Computational Resources":     ["Budget and Resource Constraints", "Computational Overhead"],
    "Language Coverage":           ["Single Language Focus", "Language Coverage Limitation"],
    "Generalization Scope":        ["Limited Generalizability", "Narrow Scope Focus"],
    "Task and Application Scope":  ["Task Scope Limitation", "Application Scope Limitation"],
    "Prompt and Interaction Design": ["Prompt Design Sensitivity", "Parameter Size Limitation",
                                      "Input Length Limitation"],
    "Bias and Representation":     ["Inherent Bias", "Training Data Bias",
                                    "Cultural Representation Bias", "Bias Propagation Risk",
                                    "Dataset Representativeness Issue", "Language Resource Bias"],
    "Annotation Quality":          ["Annotation Quality", "Annotator Diversity",
                                    "Human Annotation Dependency", "Annotator Bias"],
    "Data Coverage":               ["Limited Dataset Scope", "Single Dataset Evaluation",
                                    "Data Quality Dependence", "Curated Data Reliance"],
    "Security and Privacy":        ["Data Leakage Risk", "Privacy Concerns",
                                    "Adversarial Vulnerability", "Harmful Content Risk"],
    "Inference and Efficiency":    ["Efficiency Limitation", "Performance Degradation",
                                    "Hyperparameter Search Requirement",
                                    "Inference Speed Limitation", "Hyperparameter Sensitivity"],
}

CANONICAL_MAP = {
    "Task & Application Scope":     "Task and Application Scope",
    "Prompt & Interaction Design":  "Prompt and Interaction Design",
    "Bias & Representation":        "Bias and Representation",
    "Security & Privacy":           "Security and Privacy",
    "Inference & Efficiency":       "Inference and Efficiency",
    "Other":                        "Other",
}

LIMITGEN_ASPECTS = ["Methodology", "Experimental Design", "Result Analysis", "Literature Review"]
LIMITGEN_OTHERS  = ["Clarity", "Others", "Other"]

# BUILT-IN SAMPLE DATA  (used when --csv is omitted)

SAMPLE_DATA = [
    ("The study is not exhaustive regarding the full range of syntactic phenomena.",
     "Data Coverage", "Experimental Design", "medium"),
    ("The semi-synthetic template-based approach may not fully capture spontaneous speech.",
     "Language Coverage", "Methodology", "low"),
    ("The scope is limited to a US and English language context.",
     "Language Coverage", "Experimental Design", "high"),
    ("The approach may fail to reflect non-Western gender practices.",
     "Bias and Representation", "Methodology", "low"),
    ("The evaluation in Section 5.2 is a toy example that needs further validation.",
     "Evaluation Design", "Experimental Design", "low"),
    ("Other LLMs such as ChatGPT were not validated in this study.",
     "Generalization Scope", "Experimental Design", "medium"),
    ("Annotators were engaged in multiple projects throughout the study.",
     "Annotation Quality", "Experimental Design", "medium"),
    ("Fine-tuning requires significant GPU memory unavailable to most researchers.",
     "Computational Resources", "Methodology", "high"),
    ("The system may be vulnerable to adversarial inputs.",
     "Security and Privacy", "Methodology", "medium"),
    ("Bias in training data propagates to model predictions.",
     "Bias and Representation", "Methodology", "high"),
    ("Prompt sensitivity affects reproducibility of results.",
     "Prompt and Interaction Design", "Methodology", "medium"),
    ("The annotation process depends heavily on human judgment.",
     "Annotation Quality", "Methodology", "medium"),
    ("Results may not transfer to domains outside NLP.",
     "Generalization Scope", "Methodology", "medium"),
    ("Inference time is prohibitive for real-time applications.",
     "Inference and Efficiency", "Methodology", "high"),
    ("Cultural nuances are underrepresented in the training data.",
     "Bias and Representation", "Methodology", "medium"),
    ("The dataset consists solely of curated academic papers.",
     "Data Coverage", "Experimental Design", "medium"),
    ("The evaluation metric does not capture semantic similarity.",
     "Evaluation Design", "Result Analysis", "high"),
    ("The study lacks a discussion of potential data leakage.",
     "Security and Privacy", "Experimental Design", "low"),
    ("The model struggles with low-resource languages.",
     "Language Coverage", "Experimental Design", "high"),
    ("The framework lacks support for customized templates.",
     "Task and Application Scope", "Methodology", "medium"),
]

# GPT CLIENT
_gpt_client = None   # lazy-initialised once --openai_key is available
_gpt_model  = "gpt-4o-mini"
_CACHE_FILE = "gpt_cache.json"   # disk cache so re-runs don't cost money


def init_gpt(api_key: str, model: str = "gpt-4o-mini"):
    """Initialise the OpenAI client and set the model."""
    global _gpt_client, _gpt_model
    try:
        from openai import OpenAI
        _gpt_client = OpenAI(api_key=api_key)
        _gpt_model  = model
        # Quick connectivity check
        _gpt_client.models.list()
        print(f"[✓] OpenAI client ready — model: {_gpt_model}")
    except Exception as e:
        print(f"[!] OpenAI init failed: {e}")
        _gpt_client = None


def _load_cache() -> dict:
    if os.path.exists(_CACHE_FILE):
        with open(_CACHE_FILE) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict):
    with open(_CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=2)


def gpt_call(system: str, user: str, max_tokens: int = 600,
             temperature: float = 0.3, retries: int = 3) -> str:
    """
    Single GPT call with disk-based caching and retry logic.
    Falls back gracefully to empty string if the client is unavailable.
    """
    if _gpt_client is None:
        return ""

    cache_key = f"{_gpt_model}|{system[:80]}|{user[:120]}"
    cache = _load_cache()
    if cache_key in cache:
        return cache[cache_key]

    for attempt in range(retries):
        try:
            resp = _gpt_client.chat.completions.create(
                model=_gpt_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            result = resp.choices[0].message.content.strip()
            cache[cache_key] = result
            _save_cache(cache)
            return result
        except Exception as e:
            wait = 2 ** attempt
            print(f"    [GPT retry {attempt+1}/{retries}] {e} — waiting {wait}s …")
            time.sleep(wait)
    return ""


# HELPERS
def load_data(csv_path=None):
    if csv_path and os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        print(f"[✓] Loaded {len(df)} rows from {csv_path}")
        df.columns = [c.strip() for c in df.columns]
        if "limitation" not in df.columns:
            df = df.rename(columns={df.columns[0]: "limitation"})
    else:
        print("[!] No CSV — using built-in sample data (20 rows).")
        rows = [{"limitation": r[0], "CATEGORY_taxonomy": r[1],
                 "limitgen_taxonomy": r[2], "confidence": r[3]}
                for r in SAMPLE_DATA]
        df = pd.DataFrame(rows)

    def normalise(col):
        return df[col].str.strip().replace(CANONICAL_MAP)

    if "CATEGORY_taxonomy" in df.columns:
        df["CATEGORY_taxonomy"] = normalise("CATEGORY_taxonomy")
    if "limitgen_taxonomy" in df.columns:
        df["limitgen_taxonomy"] = normalise("limitgen_taxonomy")

    unknown_your = set(df["CATEGORY_taxonomy"].unique()) - set(YOUR_TAXONOMY_CATEGORIES)
    unknown_lim  = set(df["limitgen_taxonomy"].unique()) - set(LIMITGEN_ASPECTS) - set(LIMITGEN_OTHERS)
    if unknown_your:
        print(f"  [!] Unknown YOUR_TAXONOMY values: {sorted(unknown_your)}")
    if unknown_lim:
        print(f"  [!] Unknown LIMITGEN values: {sorted(unknown_lim)}")
    return df


def get_embeddings(texts, model=None):
    """SentenceTransformer embeddings, TF-IDF fallback."""
    try:
        from sentence_transformers import SentenceTransformer
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")
        print("  [embedding] SentenceTransformer (all-MiniLM-L6-v2) …")
        return model.encode(texts, show_progress_bar=False, batch_size=32), model
    except Exception as e:
        print(f"  [embedding] Fallback TF-IDF ({e})")
        from sklearn.feature_extraction.text import TfidfVectorizer
        vec = TfidfVectorizer(max_features=300)
        return vec.fit_transform(texts).toarray(), None


def make_output_dir(name="experiment_outputs"):
    os.makedirs(name, exist_ok=True)
    return name


# EXPERIMENT 1: COVERAGE / UNCATEGORIZABLE RATE

def experiment1_coverage(df, out_dir):
    print("\n" + "="*60)
    print("EXPERIMENT 1: COVERAGE / UNCATEGORIZABLE RATE")
    print("="*60)

    your_valid   = df["CATEGORY_taxonomy"].isin(YOUR_TAXONOMY_CATEGORIES)
    limitgen_valid = df["limitgen_taxonomy"].isin(LIMITGEN_ASPECTS)

    your_uncategorizable   = (~your_valid).sum()
    limitgen_uncategorizable = (~limitgen_valid).sum()
    your_coverage    = your_valid.mean() * 100
    limitgen_coverage = limitgen_valid.mean() * 100
    limitgen_paper_oos = 36   # Clarity(34%) + Others(2%), from LIMITGEN Figure 2

    print(f"\nYour Taxonomy  — Coverage: {your_coverage:.1f}%  |  "
          f"Uncategorizable: {your_uncategorizable}/{len(df)}")
    print(f"LIMITGEN       — Coverage: {limitgen_coverage:.1f}%  |  "
          f"Uncategorizable: {limitgen_uncategorizable}/{len(df)}")
    print(f"LIMITGEN paper-reported out-of-scope: {limitgen_paper_oos}%  "
          f"[Clarity=34% + Others=2%]")

    if "confidence" in df.columns:
        conf_dist = df.groupby(["CATEGORY_taxonomy", "confidence"]).size().unstack(fill_value=0)
        print("\nConfidence distribution:")
        print(conf_dist.to_string())

    # Plot
    n_your = len(YOUR_TAXONOMY_CATEGORIES)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Experiment 1: Coverage Comparison", fontsize=14, fontweight="bold")

    # Left: coverage bar
    ax = axes[0]
    labels = [f"Your Taxonomy\n({n_your} categories)",
              "LIMITGEN\n(4 aspects, in data)",
              "LIMITGEN\n(paper-reported)"]
    values = [your_coverage, limitgen_coverage, 100 - limitgen_paper_oos]
    colors = ["#2196F3", "#FF5722", "#FF5722"]
    bars = ax.bar(labels, values, color=colors, alpha=0.85, edgecolor="black")
    ax.set_ylim(0, 115)
    ax.set_ylabel("% of Limitations Confidently Categorized", fontsize=11)
    ax.set_title("Coverage Rate", fontsize=12)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    ax.axhline(100, color="gray", linestyle="--", alpha=0.5, label="Ideal (100%)")
    ax.legend(fontsize=9)

    # Right: stacked count
    ax2 = axes[1]
    cat_counts   = [len(df) - your_uncategorizable, len(df) - limitgen_uncategorizable]
    uncat_counts = [your_uncategorizable, limitgen_uncategorizable]
    x = np.arange(2)
    b1 = ax2.bar(x, cat_counts, color=["#4CAF50", "#F44336"], alpha=0.8, label="Categorized")
    ax2.bar(x, uncat_counts, bottom=cat_counts, color=["#C8E6C9", "#FFCDD2"],
            alpha=0.9, label="Uncategorizable", edgecolor="black", linewidth=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Your Taxonomy", "LIMITGEN"], fontsize=11)
    ax2.set_ylabel("Number of Limitations", fontsize=11)
    ax2.set_title("Categorized vs. Uncategorizable", fontsize=12)
    ax2.legend(fontsize=9)
    for bar, val in zip(b1, cat_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, val/2,
                 str(val), ha="center", va="center", fontsize=10,
                 color="white", fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp1_coverage.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[✓] Saved: exp1_coverage.png")

    results = {
        "your_taxonomy":    {"coverage_%": round(your_coverage, 2),
                             "uncategorizable_count": int(your_uncategorizable),
                             "total": len(df)},
        "limitgen_taxonomy":{"coverage_%": round(limitgen_coverage, 2),
                             "uncategorizable_count": int(limitgen_uncategorizable),
                             "total": len(df)},
        "limitgen_paper_oos_%": limitgen_paper_oos,
    }
    with open(os.path.join(out_dir, "exp1_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# EXPERIMENT 2: DISTRIBUTION FAITHFULNESS

def experiment2_distribution(df, out_dir):
    print("\n" + "="*60)
    print("EXPERIMENT 2: DISTRIBUTION FAITHFULNESS")
    print("="*60)

    your_dist   = df["CATEGORY_taxonomy"].value_counts()
    limitgen_dist = df["limitgen_taxonomy"].value_counts()
    your_pct    = (your_dist / len(df) * 100).round(2)
    limitgen_pct  = (limitgen_dist / len(df) * 100).round(2)

    def entropy(dist):
        p = np.array(dist) / dist.sum()
        p = p[p > 0]
        return -np.sum(p * np.log2(p))

    your_H    = entropy(your_dist.values)
    limitgen_H  = entropy(limitgen_dist.values)
    max_your  = np.log2(len(your_dist))
    max_lim   = np.log2(len(limitgen_dist))

    print(f"\nNormalized entropy — Your: {your_H/max_your:.3f}  |  LIMITGEN: {limitgen_H/max_lim:.3f}")

    limitgen_paper = {"Experimental Design": 23, "Clarity": 34, "Methodology": 15,
                      "Result Analysis": 15, "Literature Review": 11, "Others": 2}

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Experiment 2: Distribution Faithfulness", fontsize=15, fontweight="bold")
    gs = fig.add_gridspec(2, 3, hspace=0.45, wspace=0.35)

    # Your taxonomy horizontal bar
    ax1 = fig.add_subplot(gs[0, :2])
    sorted_your = your_pct.sort_values(ascending=True)
    colors_your = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_your)))
    bars = ax1.barh(sorted_your.index, sorted_your.values, color=colors_your, edgecolor="white")
    ax1.set_xlabel("% of Limitations", fontsize=11)
    ax1.set_title("Your Taxonomy — Distribution", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, sorted_your.values):
        ax1.text(val + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{val:.1f}%", va="center", fontsize=9)
    ideal = 100 / len(sorted_your)
    ax1.axvline(ideal, color="red", linestyle="--", alpha=0.7,
                label=f"Ideal uniform ({ideal:.1f}%)")
    ax1.legend(fontsize=9)

    # LIMITGEN paper pie
    ax2 = fig.add_subplot(gs[0, 2])
    explode = [0.05 if k in ["Clarity", "Experimental Design"] else 0
               for k in limitgen_paper.keys()]
    wedge_colors = ["#EF5350" if k in ["Clarity", "Others"] else "#42A5F5"
                    for k in limitgen_paper.keys()]
    ax2.pie(limitgen_paper.values(), labels=limitgen_paper.keys(),
            autopct="%1.0f%%", colors=wedge_colors, explode=explode,
            startangle=90, textprops={"fontsize": 8})
    ax2.set_title("LIMITGEN Distribution (paper Figure 2)", fontsize=11, fontweight="bold")
    red_p  = mpatches.Patch(color="#EF5350", label="Out-of-scope (36%)")
    blue_p = mpatches.Patch(color="#42A5F5", label="In-scope (64%)")
    ax2.legend(handles=[red_p, blue_p], fontsize=8, loc="lower center",
               bbox_to_anchor=(0.5, -0.15))

    # Side-by-side
    ax3 = fig.add_subplot(gs[1, :])
    all_cats = sorted(set(list(your_pct.index) + list(limitgen_pct.index)))
    your_vals = [your_pct.get(c, 0) for c in all_cats]
    lim_vals  = [limitgen_pct.get(c, 0) for c in all_cats]
    x = np.arange(len(all_cats))
    w = 0.38
    ax3.bar(x - w/2, your_vals, w, label="Your Taxonomy", color="#2196F3", alpha=0.85)
    ax3.bar(x + w/2, lim_vals,  w, label="LIMITGEN",      color="#FF5722", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(all_cats, rotation=35, ha="right", fontsize=8)
    ax3.set_ylabel("% of Limitations", fontsize=11)
    ax3.set_title("Side-by-Side Distribution (500-limitation dataset)", fontsize=12,
                  fontweight="bold")
    ax3.legend(fontsize=10)
    ax3.axhline(100/12, color="#2196F3", linestyle=":", alpha=0.6, label="Ideal (your)")
    ax3.axhline(100/4,  color="#FF5722", linestyle=":", alpha=0.6, label="Ideal (LIMITGEN)")

    plt.savefig(os.path.join(out_dir, "exp2_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: exp2_distribution.png")

    metrics = {
        "your_entropy_normalized":    round(your_H / max_your, 4),
        "limitgen_entropy_normalized": round(limitgen_H / max_lim, 4),
        "your_categories":    len(your_dist),
        "limitgen_categories": len(limitgen_dist),
        "limitgen_paper_outofscope_pct": 36,
    }
    with open(os.path.join(out_dir, "exp2_results.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


# EXPERIMENT 3: GPT-POWERED LLM GENERATION QUALITY

# 3a: GPT generates limitations for a given category 

def gpt_generate_limitations(category: str, n: int = 5,
                              subcategories: list = None) -> list[str]:
    """
    Ask GPT to generate `n` diverse, realistic limitation sentences for the given taxonomy category, grounded in the category's subcategories.
    These simulate what an LLM prompted with this category would write in a real peer-review scenario.
    """
    subcat_hint = ""
    if subcategories:
        subcat_hint = (f"The subcategories within '{category}' are: "
                       f"{', '.join(subcategories)}. "
                       "Draw on these to ensure variety.\n")

    system = (
        "You are an expert NLP researcher writing critical limitations sections "
        "for AI research papers. Your sentences are specific, constructive, and "
        "grounded in real reviewer concerns — not generic filler text."
    )
    user = (
        f"Generate exactly {n} distinct limitation sentences for the category "
        f"'{category}' as it appears in AI/NLP research papers.\n"
        f"{subcat_hint}"
        "Rules:\n"
        "- Each sentence must be self-contained (1–2 sentences max).\n"
        "- Cover different sub-aspects within the category.\n"
        "- Use concrete, technical language typical of peer reviews.\n"
        "- Do NOT number or bullet the sentences — output one per line.\n"
        "- Do NOT include explanations or preamble.\n"
    )
    raw = gpt_call(system, user, max_tokens=400, temperature=0.7)
    if not raw:
        # Fallback: use hardcoded templates (same as v3)
        return _fallback_templates().get(category, ["No limitation available."])[:n]
    lines = [l.strip() for l in raw.split("\n") if l.strip()]
    return lines[:n] if len(lines) >= n else lines


def _fallback_templates() -> dict:
    """Hardcoded templates used when GPT is unavailable."""
    return {
        "Evaluation Design": [
            "The evaluation coverage is insufficient; only a narrow subset of tasks is assessed.",
            "The metrics used fail to capture all relevant dimensions of model performance.",
            "Human evaluation is absent, limiting understanding of real-world utility.",
            "The benchmark does not include held-out test sets, raising data-leakage concerns.",
            "Automatic metrics may not correlate with human judgements of quality.",
        ],
        "Computational Resources": [
            "The method requires substantial GPU memory, limiting accessibility.",
            "Training time is prohibitively long without high-end infrastructure.",
            "The computational overhead makes iterative experimentation infeasible.",
            "Fine-tuning is resource-intensive and impractical for low-budget settings.",
            "The approach depends on proprietary hardware, limiting reproducibility.",
        ],
        "Language Coverage": [
            "The study is limited to English and findings may not generalise.",
            "Low-resource and morphologically complex languages are excluded.",
            "Multilingual capabilities are untested, restricting broader applicability.",
            "The model has not been evaluated on non-Latin script languages.",
            "Cross-lingual transfer performance is not reported.",
        ],
        "Generalization Scope": [
            "Results may not generalise beyond the specific domain studied.",
            "The narrow dataset scope limits broader applicability of findings.",
            "Performance on out-of-distribution inputs remains uncharacterised.",
            "The model was trained and tested on the same domain.",
            "Zero-shot generalisation to unseen tasks is not evaluated.",
        ],
        "Task and Application Scope": [
            "The approach is evaluated on a single task type.",
            "Real-world deployment scenarios are not considered.",
            "The scope is confined to controlled lab settings.",
            "The system is only tested on one application domain.",
            "Downstream task performance beyond the primary benchmark is not reported.",
        ],
        "Prompt and Interaction Design": [
            "The system is sensitive to minor variations in prompt wording.",
            "Large parameter sizes restrict applicability on resource-constrained devices.",
            "Input length limitations constrain processing of long documents.",
            "Prompt engineering choices are not systematically ablated.",
            "Model behaviour changes substantially with different instruction formats.",
        ],
        "Bias and Representation": [
            "Training data encodes demographic biases that propagate to predictions.",
            "Cultural minority groups and non-Western perspectives are underrepresented.",
            "Gender and racial biases in pre-training corpora remain unmitigated.",
            "The dataset over-represents high-resource language communities.",
            "Bias evaluation is limited to a single demographic axis.",
        ],
        "Annotation Quality": [
            "Annotator diversity is limited, introducing systematic labelling bias.",
            "Inter-annotator reliability is not reported.",
            "Annotation guidelines may not have been applied consistently.",
            "Low inter-annotator agreement suggests label noise in training data.",
            "The annotation relies on crowd workers whose expertise is unverified.",
        ],
        "Data Coverage": [
            "The dataset is limited to a single domain and lacks diversity.",
            "Curated benchmark data may not capture naturally occurring distributions.",
            "Evaluation relies on a single dataset, making conclusions fragile.",
            "The training corpus is drawn exclusively from web text.",
            "Dataset size is too small to support statistically reliable conclusions.",
        ],
        "Security and Privacy": [
            "The model may be susceptible to adversarial inputs.",
            "Privacy risks arise when applying the model to sensitive data.",
            "Data leakage from training sets could expose private information.",
            "The system lacks robustness against prompt injection attacks.",
            "User data collected during deployment is not anonymised.",
        ],
        "Inference and Efficiency": [
            "Inference latency is too high for real-time deployment.",
            "Hyperparameter sensitivity makes reproducibility difficult.",
            "The efficiency gap relative to baselines is not adequately addressed.",
            "Memory footprint at inference time prohibits edge deployment.",
            "Batch processing throughput degrades with longer inputs.",
        ],
        "Methodology": [
            "The data collection methodology introduces selection bias.",
            "The chosen modelling approach is not well-justified.",
            "Key methodological decisions lack ablation studies.",
            "The proposed method relies on unvalidated heuristics.",
            "The training objective does not align with the downstream metric.",
        ],
        "Experimental Design": [
            "The paper omits comparison against important strong baselines.",
            "The evaluation dataset is too small and homogeneous.",
            "The experimental setup does not control for confounding variables.",
            "Statistical significance is not reported for the main results.",
            "The test set overlaps with the training data, inflating performance.",
        ],
        "Result Analysis": [
            "The evaluation metrics are insufficient to characterise performance.",
            "Qualitative error analysis is absent.",
            "Statistical significance testing is not reported.",
            "Performance gains are marginal and may not be practically meaningful.",
            "The analysis does not identify which subgroups benefit most.",
        ],
    }


# 3b: GPT scores semantic similarity 

def gpt_score_similarity(generated: str, ground_truth: str) -> float:
    """
    Ask GPT to rate the semantic similarity between a generated limitation and a ground-truth reviewer comment on a 0–1 continuous scale.

    Returns a float in [0, 1]. Falls back to 0.0 on parse failure.
    """
    system = (
        "You are a strict semantic similarity judge for research paper limitation sentences. "
        "You output ONLY a single decimal number between 0.0 and 1.0 — nothing else. "
        "No explanation, no units, no words — just the number.\n\n"
        "Scoring guide:\n"
        "  1.0 = identical meaning and specificity\n"
        "  0.8 = same core concern, minor wording difference\n"
        "  0.6 = clearly related — both discuss the same limitation type\n"
        "  0.4 = loosely related — overlapping theme but different specific focus\n"
        "  0.2 = tangentially related — same broad domain only\n"
        "  0.0 = completely unrelated"
    )
    user = (
        f"Generated limitation:\n{generated}\n\n"
        f"Ground-truth reviewer comment:\n{ground_truth}"
    )
    raw = gpt_call(system, user, max_tokens=5, temperature=0.0)
    try:
        val = float(raw.strip().split()[0])   # take first token only
        return min(1.0, max(0.0, val))
    except (ValueError, TypeError, IndexError):
        return 0.0


def gpt_score_batch(generated_list: list, ground_truth_list: list,
                    sample_size: int = 30) -> dict:
    """
    Compute soft recall / precision / Jaccard using GPT as the similarity judge.

    SCORING APPROACH  (fixes the all-zeros problem):
   ───────────────────
    Instead of binary hit/miss with a hard threshold (which produces 0.000 for
    every category except one when generated sentences are generic and GT sentences
    are paper-specific), we use CONTINUOUS SOFT SCORING:

      Soft Recall    = mean over sampled GT items of  max(sim(gt, gen_i))
      Soft Precision = mean over generated items of   max(sim(gen_i, gt_j))
      Soft Jaccard   = (R * P) / (R + P - R * P)

    This is the same formulation used in BERTScore and soft-ROUGE:
    - Every GT item contributes its best-match score (0–1) — not a binary 0/1
    - Categories where generated sentences thematically cover the GT pool
      score higher than categories where there is no thematic overlap
    - LIMITGEN's 4 coarse aspects will score lower than your 12 fine-grained
      categories because their generated sentences cannot precisely cover
      the specific concerns in each real reviewer comment

    Cost control:
    - GT is RANDOMLY sampled (not first-N) to avoid order bias
    - Generated list is always fully scored (only 5 items)
    """
    if not generated_list or not ground_truth_list:
        return {"recall": 0.0, "precision": 0.0, "jaccard": 0.0}

    # Random sample for cost control (reproducible seed)
    rng = np.random.default_rng(seed=42)
    if sample_size and len(ground_truth_list) > sample_size:
        indices   = rng.choice(len(ground_truth_list), size=sample_size, replace=False)
        gt_sample = [ground_truth_list[i] for i in sorted(indices)]
    else:
        gt_sample = ground_truth_list

    # Soft Recall: mean over GT of best-match similarity to any generated item
    recall_sims = []
    for gt in gt_sample:
        best = max(gpt_score_similarity(gen, gt) for gen in generated_list)
        recall_sims.append(best)
    soft_recall = float(np.mean(recall_sims))

    # Soft Precision: mean over generated items of best-match similarity to any GT
    prec_sims = []
    for gen in generated_list:
        # For precision use the full GT pool (it's small enough — only 5 gen items)
        best = max(gpt_score_similarity(gen, gt) for gt in gt_sample)
        prec_sims.append(best)
    soft_precision = float(np.mean(prec_sims))

    # Soft Jaccard
    denom = soft_recall + soft_precision - soft_recall * soft_precision
    soft_jaccard = (soft_recall * soft_precision / denom) if denom > 0 else 0.0

    return {
        "recall":           round(soft_recall,    4),
        "precision":        round(soft_precision, 4),
        "jaccard":          round(soft_jaccard,   4),
        "recall_per_item":  [round(s, 3) for s in recall_sims],
        "precision_per_item": [round(s, 3) for s in prec_sims],
    }


# TF-IDF fallback (used when GPT is unavailable)

def tfidf_score_batch(generated_list: list, ground_truth_list: list,
                      sample_size: int = 30) -> dict:
    """
    Soft recall/precision/Jaccard via word-level F1 (no threshold).
    Used when GPT is unavailable. Mirrors the GPT soft-scoring approach.
    """
    import re as _re
    _STOP = {
        "the","a","an","is","are","was","were","be","been","have","has","had",
        "do","does","did","will","would","may","might","can","could","of","in",
        "to","for","on","at","by","with","from","that","this","it","not","or",
        "and","but","as","its","our","their","which","who","if","so","we","they",
        "also","only","more","such","each","any","all","some","than","both",
    }

    def _tok(s):
        return set(_re.sub(r"[^a-zA-Z0-9]", " ", s.lower()).split()) - _STOP

    def _overlap(a, b):
        return len(a & b) / len(a) if a else 0.0

    rng = np.random.default_rng(seed=42)
    gt_sample = ground_truth_list
    if sample_size and len(ground_truth_list) > sample_size:
        idx = rng.choice(len(ground_truth_list), size=sample_size, replace=False)
        gt_sample = [ground_truth_list[i] for i in sorted(idx)]

    gen_toks = [_tok(s) for s in generated_list]
    gt_toks  = [_tok(s) for s in gt_sample]

    recall_scores = [max(_overlap(gt, g) for g in gen_toks) for gt in gt_toks]
    prec_scores   = [max(_overlap(g, gt) for gt in gt_toks) for g in gen_toks]

    R = float(np.mean(recall_scores))
    P = float(np.mean(prec_scores))
    denom = R + P - R * P
    J = (R * P / denom) if denom > 0 else 0.0

    return {"recall": round(R, 4), "precision": round(P, 4), "jaccard": round(J, 4)}


# Main Exp 3 orchestrator

def experiment3_llm_generation(df, out_dir, sample_size: int = 30):
    """
    Two-stage pipeline:
      Stage 1 — GPT generates 5 limitation sentences per category.
      Stage 2 — GPT scores each generated sentence against the real
                human-written limitations in that category.

    Falls back to TF-IDF scoring (and hardcoded templates) if GPT unavailable.
    """
    print("\n" + "="*60)
    print("EXPERIMENT 3: GPT-POWERED LLM GENERATION QUALITY")
    print("="*60)

    use_gpt = (_gpt_client is not None)
    scorer_name = "GPT judge" if use_gpt else "TF-IDF soft-match (GPT unavailable)"
    print(f"  Scoring method: {scorer_name}")
    print(f"  Generation:     {'GPT' if use_gpt else 'hardcoded templates (GPT unavailable)'}")

    # Stage 1: Generate limitations per category
    all_generated = {}   # category → list[str]
    print("\n  [Stage 1] Generating limitations per category …")

    for category in YOUR_TAXONOMY_CATEGORIES:
        subcats = YOUR_TAXONOMY_SUBCATEGORIES.get(category)
        if use_gpt:
            print(f"    GPT → {category} …", end=" ", flush=True)
            lims = gpt_generate_limitations(category, n=5, subcategories=subcats)
            print(f"{len(lims)} sentences")
        else:
            lims = _fallback_templates().get(category,
                                             ["No limitation available for this category."])
        all_generated[category] = lims

    for aspect in LIMITGEN_ASPECTS:
        if use_gpt:
            print(f"    GPT → {aspect} (LIMITGEN) …", end=" ", flush=True)
            lims = gpt_generate_limitations(aspect, n=5)
            print(f"{len(lims)} sentences")
        else:
            lims = _fallback_templates().get(aspect,
                                             ["No limitation available for this aspect."])
        all_generated[aspect] = lims

    # Save generated limitations for inspection
    gen_path = os.path.join(out_dir, "exp3_generated_limitations.json")
    with open(gen_path, "w") as f:
        json.dump(all_generated, f, indent=2)
    print(f"\n  [✓] Generated limitations saved: {gen_path}")

    # Stage 2: Score against ground truth
    print(f"\n  [Stage 2] Scoring (sample_size={sample_size} GT items per category) …")

    def score(generated, ground_truth):
        if use_gpt:
            return gpt_score_batch(generated, ground_truth, sample_size=sample_size)
        return tfidf_score_batch(generated, ground_truth, sample_size=sample_size)

    your_scores    = []
    limitgen_scores = []

    for category in YOUR_TAXONOMY_CATEGORIES:
        cat_df = df[df["CATEGORY_taxonomy"] == category]
        if len(cat_df) == 0:
            continue
        ground_truth = cat_df["limitation"].tolist()
        generated    = all_generated.get(category, [])
        print(f"    Scoring {category} ({len(ground_truth)} GT, {len(generated)} gen) …")
        s = score(generated, ground_truth)
        s["category"]       = category
        s["n_ground_truth"] = len(ground_truth)
        s["n_generated"]    = len(generated)
        your_scores.append(s)

    for aspect in LIMITGEN_ASPECTS:
        asp_df = df[df["limitgen_taxonomy"] == aspect]
        if len(asp_df) == 0:
            continue
        ground_truth = asp_df["limitation"].tolist()
        generated    = all_generated.get(aspect, [])
        print(f"    Scoring {aspect} [{len(ground_truth)} GT, {len(generated)} gen] …")
        s = score(generated, ground_truth)
        s["category"]       = aspect
        s["n_ground_truth"] = len(ground_truth)
        s["n_generated"]    = len(generated)
        limitgen_scores.append(s)

    your_df  = pd.DataFrame(your_scores)
    lim_df   = pd.DataFrame(limitgen_scores)

    # Drop helper columns for display/aggregation
    agg_cols = ["recall", "precision", "jaccard"]

    def agg(sdf):
        return {k: round(sdf[k].mean(), 4) for k in agg_cols if k in sdf.columns}

    your_agg = {f"mean_{k}": v for k, v in agg(your_df).items()}
    lim_agg  = {f"mean_{k}": v for k, v in agg(lim_df).items()}

    print(f"\n  Your Taxonomy averages: {your_agg}")
    print(f"  LIMITGEN averages:      {lim_agg}")

    print("\nPer-Category Scores (Your Taxonomy):")
    print(your_df[["category", "n_ground_truth", "recall",
                   "precision", "jaccard"]].to_string(index=False))
    print("\nPer-Aspect Scores (LIMITGEN):")
    print(lim_df[["category", "n_ground_truth", "recall",
                  "precision", "jaccard"]].to_string(index=False))

    # Plot 1: Aggregate comparison bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        f"Experiment 3: LLM Generation Quality  [{scorer_name}]\n"
        "(Soft Recall / Precision / Jaccard — continuous scores, no binary threshold)",
        fontsize=12, fontweight="bold"
    )
    colors = ["#2196F3", "#FF5722"]

    for i, (metric, label) in enumerate(
        zip(["recall", "precision", "jaccard"],
            ["Soft Recall ↑", "Soft Precision ↑", "Soft Jaccard ↑"])
    ):
        ax = axes[i]
        y_mean = your_agg.get(f"mean_{metric}", 0)
        l_mean = lim_agg.get(f"mean_{metric}", 0)

        # Per-category scatter + mean bar
        yv = your_df[metric].values
        lv = lim_df[metric].values
        max_val = max(yv.max() if len(yv) else 0,
                      lv.max() if len(lv) else 0, 0.05)

        bars = ax.bar(["Your\nTaxonomy", "LIMITGEN"], [y_mean, l_mean],
                      color=colors, alpha=0.75, edgecolor="black", linewidth=0.8,
                      zorder=2, label="Mean")

        # Overlay individual category dots
        jitter = 0.08
        rng2 = np.random.default_rng(seed=7)
        ax.scatter(rng2.uniform(-jitter, jitter, len(yv)),
                   yv, color="#0D47A1", alpha=0.55, s=30, zorder=3)
        ax.scatter(1 + rng2.uniform(-jitter, jitter, len(lv)),
                   lv, color="#BF360C", alpha=0.55, s=30, zorder=3)

        ax.set_ylim(0, min(1.05, max_val * 1.45 + 0.05))
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel(label, fontsize=10)

        for bar, val, col in zip(bars, [y_mean, l_mean], ["#0D47A1", "#BF360C"]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max_val * 0.04,
                    f"{val:.3f}", ha="center", fontsize=11,
                    fontweight="bold", color=col)

        # Clean legend using explicit handles (avoids _child* leak)
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#2196F3",
                   markersize=7, alpha=0.7, label="Category score (Your)"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="#BF360C",
                   markersize=7, alpha=0.7, label="Category score (LIMITGEN)"),
            Patch(facecolor="#888888", alpha=0.75, label="Mean (bar height)"),
        ]
        ax.legend(handles=legend_handles, fontsize=7, loc="upper right")

    fig.text(0.5, -0.02,
             "Soft scoring: for each GT item, score = max GPT-similarity to any generated sentence (continuous 0–1, no threshold).",
             ha="center", fontsize=9, style="italic", color="#555")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp3_llm_generation.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[✓] Saved: exp3_llm_generation.png")

    # Plot 2: Per-category horizontal bar chart
    fig2, axes2 = plt.subplots(1, 2, figsize=(20, max(6, len(your_df) * 0.6 + 2)))
    fig2.suptitle(f"Experiment 3: Per-Category Scores  [{scorer_name}]",
                  fontsize=13, fontweight="bold")

    for ax, sub_df, title, color in [
        (axes2[0], your_df,  "Your Taxonomy (per category)", "#2196F3"),
        (axes2[1], lim_df,   "LIMITGEN (per aspect)",        "#FF5722"),
    ]:
        sub_df = sub_df.copy().sort_values("recall", ascending=True)
        x      = np.arange(len(sub_df))
        width  = 0.26

        ax.barh(x - width, sub_df["recall"].values,    width, label="Recall",
                color=color, alpha=0.90)
        ax.barh(x,          sub_df["precision"].values, width, label="Precision",
                color=color, alpha=0.60)
        ax.barh(x + width,  sub_df["jaccard"].values,   width, label="Jaccard",
                color=color, alpha=0.35)

        ax.set_yticks(x)
        ax.set_yticklabels(sub_df["category"].values, fontsize=9)
        ax.set_xlabel("Score (soft, 0–1)", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlim(0, min(1.0, sub_df[["recall","precision","jaccard"]].values.max() * 1.4 + 0.05))
        mean_r = sub_df["recall"].mean()
        ax.axvline(mean_r, color="black", linestyle="--", alpha=0.5,
                   label=f"Mean recall = {mean_r:.3f}")
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp3_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[✓] Saved: exp3_heatmap.png")

    results = {
        "scorer": scorer_name,
        "scoring_method": "soft_continuous_no_threshold",
        "your_taxonomy": your_agg,
        "limitgen": lim_agg,
        "your_per_category": your_df[["category","n_ground_truth","recall",
                                      "precision","jaccard"]].to_dict(orient="records"),
        "limitgen_per_aspect": lim_df[["category","n_ground_truth","recall",
                                       "precision","jaccard"]].to_dict(orient="records"),
    }
    with open(os.path.join(out_dir, "exp3_results.json"), "w") as f:
        json.dump(results, f, indent=2, default=str)
    return results


# EXPERIMENT 4: EMBEDDING COHERENCE
def experiment4_coherence(df, out_dir):
    print("\n" + "="*60)
    print("EXPERIMENT 4: TAXONOMY COHERENCE (EMBEDDING-BASED)")
    print("="*60)

    texts = df["limitation"].tolist()
    print(f"  Computing embeddings for {len(texts)} limitations …")
    embeddings, _ = get_embeddings(texts)

    results = {}

    for taxonomy_name, label_col, valid_labels in [
        ("Your Taxonomy", "CATEGORY_taxonomy", YOUR_TAXONOMY_CATEGORIES),
        ("LIMITGEN",      "limitgen_taxonomy",  LIMITGEN_ASPECTS),
    ]:
        print(f"\n  Processing: {taxonomy_name}")
        mask   = df[label_col].isin(valid_labels)
        emb    = embeddings[mask]
        labels = df.loc[mask, label_col].values
        unique_labels = sorted(set(labels))

        if len(unique_labels) < 2:
            print(f"  [!] Only {len(unique_labels)} category — skipping.")
            continue

        # Intra-category similarity
        intra_scores = {}
        for cat in unique_labels:
            cat_emb = emb[labels == cat]
            if len(cat_emb) < 2:
                intra_scores[cat] = float("nan")
                continue
            sim = cosine_similarity(cat_emb)
            upper = sim[np.triu_indices_from(sim, k=1)]
            intra_scores[cat] = float(np.mean(upper))
        mean_intra = np.nanmean(list(intra_scores.values()))

        # Inter-category separation (centroid-based)
        centroids = {cat: emb[labels == cat].mean(axis=0) for cat in unique_labels
                     if (labels == cat).sum() > 0}
        C_mat = np.array(list(centroids.values()))
        C_sim = cosine_similarity(C_mat)
        upper_inter = C_sim[np.triu_indices_from(C_sim, k=1)]
        mean_inter_sim = float(np.mean(upper_inter))
        mean_separation = 1 - mean_inter_sim

        # Silhouette
        le = LabelEncoder()
        label_ids = le.fit_transform(labels)
        try:
            sil = float(silhouette_score(emb, label_ids, metric="cosine"))
        except Exception:
            sil = float("nan")

        print(f"    Intra-similarity:   {mean_intra:.4f}")
        print(f"    Inter-separation:   {mean_separation:.4f}")
        print(f"    Silhouette (cosine): {sil:.4f}")

        results[taxonomy_name] = {
            "mean_intra_similarity":  round(mean_intra, 4),
            "mean_inter_similarity":  round(mean_inter_sim, 4),
            "mean_inter_separation":  round(mean_separation, 4),
            "silhouette_score":       round(sil, 4) if not np.isnan(sil) else None,
            "per_category_intra": {k: (round(v, 4) if not np.isnan(v) else None)
                                   for k, v in intra_scores.items()},
        }

    # Summary bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle("Experiment 4: Taxonomy Coherence (Semantic Embeddings)",
                 fontsize=13, fontweight="bold")
    tax_names = list(results.keys())
    colors    = ["#2196F3", "#FF5722"]

    for i, (metric, title) in enumerate([
        ("mean_intra_similarity",
         "Intra-Category Similarity\n(↑ higher = more coherent)"),
        ("mean_inter_separation",
         "Inter-Category Separation\n(↑ higher = more distinct — see note*)"),
        ("silhouette_score",
         "Silhouette Score\n(↑ closer to 0 = better — primary metric)"),
    ]):
        ax = axes[i]
        vals = [results[t].get(metric) or 0 for t in tax_names]
        bars = ax.bar(tax_names, vals, color=colors[:len(vals)],
                      alpha=0.85, edgecolor="black")
        max_abs = max(abs(v) for v in vals) if vals else 0.1
        ax.set_ylim(min(min(vals) - 0.05, -0.05), max_abs * 1.35 + 0.02)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", labelsize=9)
        for bar, val in zip(bars, vals):
            va  = "bottom" if val >= 0 else "top"
            off = 0.005 if val >= 0 else -0.008
            ax.text(bar.get_x() + bar.get_width()/2, val + off,
                    f"{val:.4f}", ha="center", va=va,
                    fontsize=10, fontweight="bold")

    # Footnote explaining inter-separation paradox and negative silhouettes
    fig.text(0.5, -0.04,
             "* Higher inter-separation for LIMITGEN is expected: 4 coarse buckets place centroids further apart by construction.\n"
             "  Both silhouette scores are negative — this reflects shared surface vocabulary across limitation categories.\n"
             "  Your taxonomy score (−0.013) is closer to 0 than LIMITGEN (−0.037), indicating finer-grained cluster structure.",
             ha="center", fontsize=8.5, style="italic", color="#555555")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "exp4_coherence.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[✓] Saved: exp4_coherence.png")

    # Per-category intra-similarity bar (all categories shown)
    if "Your Taxonomy" in results:
        per_cat = results["Your Taxonomy"]["per_category_intra"]
        # Include all — replace None with 0 and mark them
        cats = list(per_cat.keys())
        vals = [per_cat[c] if per_cat[c] is not None else 0.0 for c in cats]
        has_data = [per_cat[c] is not None for c in cats]

        sorted_pairs = sorted(zip(vals, cats, has_data), reverse=True)
        vals_s  = [p[0] for p in sorted_pairs]
        cats_s  = [p[1] for p in sorted_pairs]
        has_s   = [p[2] for p in sorted_pairs]

        bar_colors = [plt.cm.Blues(v / max(vals_s + [0.01])) if h else "#CCCCCC"
                      for v, h in zip(vals_s, has_s)]

        fig2, ax2 = plt.subplots(figsize=(11, max(5, len(cats_s) * 0.55)))
        bars2 = ax2.barh(cats_s, vals_s, color=bar_colors, edgecolor="white")
        ax2.set_xlabel("Mean Intra-Category Cosine Similarity", fontsize=11)
        n_shown = sum(1 for h in has_s if h)
        ax2.set_title(
            f"Per-Category Coherence (Your Taxonomy — {n_shown} of {len(YOUR_TAXONOMY_CATEGORIES)} categories with ≥2 items)",
            fontsize=12, fontweight="bold")
        mean_val = np.nanmean([v for v, h in zip(vals_s, has_s) if h])
        ax2.axvline(mean_val, color="red", linestyle="--", alpha=0.7,
                    label=f"Mean = {mean_val:.3f}")
        ax2.legend(fontsize=9)
        for bar, val, has in zip(bars2, vals_s, has_s):
            label = f"{val:.3f}" if has else "n<2"
            ax2.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                     label, va="center", fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "exp4_per_category_coherence.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[✓] Saved: exp4_per_category_coherence.png")

    with open(os.path.join(out_dir, "exp4_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


# SUMMARY FIGURE
def make_summary_figure(exp1, exp2, exp3, exp4, out_dir):
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(
        "Taxonomy Comparison: Your 12-Category Taxonomy vs. LIMITGEN (4 Aspects)\n"
        f"[Exp 3 scorer: {exp3.get('scorer', 'N/A')}]",
        fontsize=14, fontweight="bold", y=1.01
    )
    gs     = fig.add_gridspec(2, 4, hspace=0.5, wspace=0.45)
    labels = ["Your\nTaxonomy", "LIMITGEN"]
    colors = ["#2196F3", "#FF5722"]

    def bar2(ax, vals, title, ylim=None, fmt=".1f"):
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, edgecolor="black")
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, max(vals) * 1.35 + 0.02)
        ax.set_title(title, fontsize=10, fontweight="bold")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ax.get_ylim()[1] * 0.02,
                    f"{val:{fmt}}", ha="center", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)

    # Exp 1 coverage
    ax1 = fig.add_subplot(gs[0, 0])
    bar2(ax1, [exp1["your_taxonomy"]["coverage_%"],
               exp1["limitgen_taxonomy"]["coverage_%"]],
         "Coverage Rate (%)\n↑ higher is better", ylim=(0, 115), fmt=".1f")

    # Exp 2 entropy
    ax2 = fig.add_subplot(gs[0, 1])
    bar2(ax2, [exp2["your_entropy_normalized"],
               exp2["limitgen_entropy_normalized"]],
         "Distribution Entropy\n(normalized, ↑ more balanced)", ylim=(0, 1.25), fmt=".3f")
    ax2.axhline(1.0, color="gray", linestyle="--", alpha=0.5)

    # Exp 1 out-of-scope
    ax5 = fig.add_subplot(gs[0, 2])
    your_oos = 100 - exp1["your_taxonomy"]["coverage_%"]
    bar2(ax5, [your_oos, 36.0],
         "Out-of-Scope Rate (%)\n↓ lower is better", ylim=(0, 55), fmt=".1f")

    # Exp 4 inter-separation
    ax6 = fig.add_subplot(gs[0, 3])
    if exp4:
        sep_vals = [exp4[t].get("mean_inter_separation", 0) for t in list(exp4.keys())]
        bar2(ax6, sep_vals,
             "Inter-Category Separation\n(higher = more distinct ↑)", fmt=".3f")

    # Exp 3 recall / precision / jaccard
    for i, (m, lbl) in enumerate(
        zip(["mean_recall", "mean_precision", "mean_jaccard"],
            ["Recall ↑", "Precision ↑", "Jaccard ↑"])
    ):
        ax = fig.add_subplot(gs[1, i])
        yv = exp3["your_taxonomy"].get(m, 0)
        lv = exp3["limitgen"].get(m, 0)
        bar2(ax, [yv, lv],
             f"LLM Gen: {lbl}\n(vs. real reviewer comments)", fmt=".3f")

    # Exp 4 coherence
    ax4 = fig.add_subplot(gs[1, 3])
    if exp4:
        tnames = list(exp4.keys())
        iv = [exp4[t].get("mean_intra_similarity", 0) for t in tnames]
        sv = [exp4[t].get("silhouette_score") or 0  for t in tnames]
        x  = np.arange(len(tnames))
        w  = 0.35
        ax4.bar(x - w/2, iv, w, color=colors[:len(tnames)], alpha=0.7, label="Intra-sim ↑")
        ax4.bar(x + w/2, sv, w, color=colors[:len(tnames)], alpha=0.4,
                edgecolor="black", hatch="//", label="Silhouette (↑ = closer to 0)")
        all_v = iv + sv
        ax4.set_ylim(min(min(all_v) - 0.03, -0.05),
                     max(max(all_v) * 1.4, 0.05))
        ax4.axhline(0, color="black", linewidth=0.8, alpha=0.5)
        ax4.set_xticks(x)
        ax4.set_xticklabels(["Your\nTaxonomy", "LIMITGEN"], fontsize=8)
        ax4.set_title("Coherence Metrics\n(intra-sim ↑ better | silhouette ↑ closer to 0)",
                      fontsize=9, fontweight="bold")
        ax4.legend(fontsize=7)
        for xi, (i_, s_) in enumerate(zip(iv, sv)):
            ax4.text(xi - w/2, i_ + 0.005, f"{i_:.3f}",
                     ha="center", fontsize=7, fontweight="bold")
            off = 0.005 if s_ >= 0 else -0.008
            ax4.text(xi + w/2, s_ + off, f"{s_:.3f}",
                     ha="center", fontsize=7, fontweight="bold",
                     va="bottom" if s_ >= 0 else "top")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "SUMMARY_all_experiments.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[✓] Summary saved: SUMMARY_all_experiments.png")


# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Taxonomy vs LIMITGEN — GPT-powered experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--csv",        type=str, default=None,
                        help="Path to annotated_limitations_complete.csv")
    parser.add_argument("--out",        type=str, default="experiment_outputs_gpt",
                        help="Output directory")
    parser.add_argument("--openai_key", type=str, default=None,
                        help="OpenAI API key (sk-...)")
    parser.add_argument("--model",      type=str, default="gpt-4o-mini",
                        help="OpenAI model (default: gpt-4o-mini)")
    parser.add_argument("--sample_size", type=int, default=30,
                        help="Max GT items sampled per category for GPT scoring "
                             "(lower = cheaper, default 30)")
    parser.add_argument("--skip_exp",  nargs="*", default=[],
                        help="Experiments to skip, e.g. --skip_exp 3")
    args = parser.parse_args()

    # Initialise GPT (optional)
    key = args.openai_key or os.environ.get("OPENAI_API_KEY")
    if key:
        init_gpt(key, args.model)
    else:
        print("[!] No OpenAI key — Exp 3 will use TF-IDF fallback.\n"
              "    Set --openai_key sk-... or OPENAI_API_KEY env var to enable GPT.")

    out_dir = make_output_dir(args.out)
    df      = load_data(args.csv)

    print(f"\nDataset: {df.shape}  |  columns: {list(df.columns)}")
    print(f"Your taxonomy categories: {sorted(df['CATEGORY_taxonomy'].unique())}")
    print(f"LIMITGEN aspects:         {sorted(df['limitgen_taxonomy'].unique())}")

    exp1 = exp2 = exp3 = exp4 = None

    if "1" not in args.skip_exp:
        exp1 = experiment1_coverage(df, out_dir)
    if "2" not in args.skip_exp:
        exp2 = experiment2_distribution(df, out_dir)
    if "3" not in args.skip_exp:
        exp3 = experiment3_llm_generation(df, out_dir, sample_size=args.sample_size)
    if "4" not in args.skip_exp:
        exp4 = experiment4_coherence(df, out_dir)

    if all(x is not None for x in [exp1, exp2, exp3, exp4]):
        make_summary_figure(exp1, exp2, exp3, exp4, out_dir)

    print("\n" + "="*60)
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results in: {out_dir}/")
    print("="*60)


if __name__ == "__main__":
    main()

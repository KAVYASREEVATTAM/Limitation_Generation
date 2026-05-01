import os, json, time, re, argparse, logging, difflib, math
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# CONFIG
DEFAULT_MODEL        = "gpt-4o-mini"
DEFAULT_STRONG_MODEL = "gpt-4o"
EMBED_MODEL          = "text-embedding-3-large"
EXTRACT_BATCH        = 20
EMBED_BATCH          = 100
MAX_RETRIES          = 4
RETRY_SLEEP          = 6
OVER_CLUSTER_N       = 24
TARGET_MIN           = 6
TARGET_MAX           = 16
N_CENTRAL            = 12
MAX_FINE_HARD        = 12
MAX_UNSPEC_PCT       = 0.30
np.random.seed(42)

CANONICAL_DIMENSIONS = [
    "Data Coverage",           "Evaluation Design",
    "Generalisation Scope",    "Baseline Comparison",
    "Computational Resources", "Scalability",
    "Annotation Quality",      "Human Evaluation",
    "Language Coverage",       "Modality Coverage",
    "Reproducibility",         "Ethical Scope",
    "Temporal Scope",          "Theoretical Grounding",
    "Real-World Applicability",
]

DISTINCTION_RULES = """
CRITICAL DISTINCTION RULES — do NOT merge these:
  • Annotation Quality (reliability of individual labels) ≠ Sample Size
    (how many items collected) ≠ Language Coverage (which languages)
  • Computational Resources (hardware/cost) ≠ Scalability (growth behaviour)
  • Language Coverage (which languages) ≠ Modality Coverage (text/image/audio)
  • Evaluation Design (how measured) ≠ Baseline Comparison (what compared against)
"""


# HELPERS
def make_client(model: str) -> OpenAI:
    key = os.environ.get("OPENAI_API_KEY", "ENTER API KEY")
    if not key:
        raise EnvironmentError("Set OPENAI_API_KEY before running.")
    client = OpenAI(api_key=key)
    try:
        r = client.chat.completions.create(
            model=model,
            messages=[{"role":"user","content":"Reply with the single word: Ready"}],
            max_tokens=10, temperature=0)
        log.info("Model '%s' OK → '%s'", model, r.choices[0].message.content.strip())
    except Exception as e:
        raise RuntimeError(f"Model '{model}' validation failed: {e}")
    return client


def gpt(client, model: str, system: str, user: str,
        temperature: float = 0.0, json_mode: bool = False,
        max_tokens: int = 1500) -> str:
    kwargs = dict(
        model=model,
        messages=[{"role":"system","content":system},
                  {"role":"user","content":user}],
        temperature=temperature, max_tokens=max_tokens)
    if json_mode:
        kwargs["response_format"] = {"type":"json_object"}
    for attempt in range(1, MAX_RETRIES+1):
        try:
            r = client.chat.completions.create(**kwargs)
            return r.choices[0].message.content.strip()
        except Exception as e:
            log.warning("GPT attempt %d/%d (%s): %s", attempt, MAX_RETRIES, model, e)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_SLEEP * attempt)
    raise RuntimeError("GPT call failed after retries.")


def embed(client, texts: List[str],
          cache_path: Optional[Path] = None) -> np.ndarray:
    if cache_path and cache_path.exists():
        arr = np.load(cache_path)
        if arr.shape[0] == len(texts):
            log.info("Loaded embeddings from cache (%d).", len(texts))
            return arr
    log.info("Embedding %d texts ...", len(texts))
    out = []
    for i in tqdm(range(0, len(texts), EMBED_BATCH), desc="Embed"):
        batch = texts[i:i+EMBED_BATCH]
        resp  = client.embeddings.create(model=EMBED_MODEL, input=batch)
        out.extend([r.embedding for r in sorted(resp.data, key=lambda x: x.index)])
        time.sleep(0.15)
    arr = np.array(out, dtype=np.float32)
    if cache_path:
        np.save(cache_path, arr)
        log.info("Saved embeddings → %s", cache_path)
    return arr


# STEP 1 — EXTRACT TYPE LABELS
EXTRACT_SYS = """You are an expert in research methodology.
For each numbered limitation sentence, extract a SHORT TYPE LABEL (3–6 words) naming the KIND of limitation — NOT the topic.

RULES:
   Name the CONSTRAINT TYPE: "single dataset evaluation", "no human evaluation"
   NOT the subject: "ImageNet limitation", "GPT-4 weakness"
   Noun phrase starting with the constrained dimension
   Never start with: "poor", "weak", "bad", "low", "suboptimal"

Return ONLY valid JSON: {"labels": {"1": "label", "2": "label", ...}}
"""

def _parse_labels(raw: str, batch_keys: List[str]) -> Dict[str,str]:
    parsed: Dict[str,str] = {}
    try:
        data = json.loads(raw)
        if isinstance(data.get("labels"), dict):
            parsed = {str(k): str(v) for k,v in data["labels"].items()}
        elif isinstance(data, dict):
            vals = [str(v) for v in data.values() if isinstance(v,str) and len(v)>2]
            for i, key in enumerate(batch_keys):
                if i < len(vals): parsed[key] = vals[i]
    except Exception:
        pass
    if not parsed:
        for m in re.finditer(r'"(\d+)"\s*:\s*"([^"]{5,80})"', raw):
            parsed[m.group(1)] = m.group(2)
    return parsed

def extract_type_labels(client, model: str, df: pd.DataFrame, col: str,
                         cache_path: Optional[Path]=None) -> List[str]:
    if cache_path and cache_path.exists():
        saved = json.loads(cache_path.read_text())
        if len(saved) == len(df):
            log.info("Loaded type labels from cache (%d).", len(df))
            return saved
    labels  = ["unspecified limitation"] * len(df)
    idx     = df.index.tolist()
    pos_map = {gi: pos for pos, gi in enumerate(idx)}
    first   = False
    for start in tqdm(range(0, len(idx), EXTRACT_BATCH), desc="Extract labels"):
        chunk = idx[start:start+EXTRACT_BATCH]
        batch = {str(li+1): str(df.at[chunk[li], col]).strip()
                 for li in range(len(chunk))
                 if pd.notna(df.at[chunk[li], col]) and str(df.at[chunk[li], col]).strip()}
        if not batch: continue
        user = "Label each:\n" + "\n".join(f'{k}. "{v}"' for k,v in batch.items())
        raw  = gpt(client, model, EXTRACT_SYS, user, json_mode=True, max_tokens=700)
        if not first:
            log.info("First response: %s", raw[:200]); first=True
        parsed = _parse_labels(raw, list(batch.keys()))
        for li, gi in enumerate(chunk):
            label = parsed.get(str(li+1),"").strip().lower()
            labels[pos_map[gi]] = label or "unspecified limitation"
        time.sleep(0.3)
    n_un = sum(1 for l in labels if not l or l=="unspecified limitation")
    pct  = n_un/max(len(labels),1)
    log.info("Extraction: %d/%d unspecified (%.1f%%)", n_un, len(labels), 100*pct)
    if pct > MAX_UNSPEC_PCT:
        raise RuntimeError(f"Extraction failed: {100*pct:.0f}% unspecified. Check model name.")
    if cache_path:
        cache_path.write_text(json.dumps(labels, indent=2))
    return labels


# STEP 2-3 — EMBED + OVER-CLUSTER
def ward_cluster(X_norm: np.ndarray, n: int) -> np.ndarray:
    log.info("Ward clustering (n=%d) ...", n)
    dist = pdist(X_norm, metric="cosine")
    Z    = linkage(dist, method="ward")
    return fcluster(Z, t=n, criterion="maxclust") - 1

def central_sentences(raw_sents: List[str], member_idx: List[int],
                       X_norm: np.ndarray, centroid: np.ndarray,
                       n: int=N_CENTRAL) -> List[str]:
    if not member_idx: return []
    sims = cosine_similarity(X_norm[member_idx], centroid.reshape(1,-1)).flatten()
    top  = sims.argsort()[::-1][:n]
    return [raw_sents[member_idx[i]] for i in top]


# STEP 4 — SEMANTIC CONSOLIDATION
CONSOLIDATE_SYS = """You are a research taxonomy expert.

Group the over-clusters below into EXACTLY {target_n} COARSE CATEGORIES.
This number is a HARD REQUIREMENT — not a suggestion. Count carefully.

A coarse category = one CONSTRAINT DIMENSION (what was not done/tested/available).
NEVER name a category after a symptom ("Performance Issues", "Quality Problems").

CANONICAL DIMENSIONS (examples only, not exhaustive):
{dims}

{distinctions}

QUALITY:
   Every over-cluster ID must appear in exactly one group
   Mutually exclusive — one limitation fits one category only
   No catch-all bins

Return ONLY valid JSON:
{{
  "coarse_categories": [
    {{
      "coarse_name": "2–4 word constraint dimension",
      "over_cluster_ids": [list of integer IDs],
      "rationale": "one sentence: what unifies these",
      "definition": "2–3 sentences: what this covers AND does NOT cover",
      "constraint_dimension": "canonical dimension or 'new: X'"
    }}
  ]
}}
"""

def semantic_consolidation(client, strong_model: str,
                             over_cluster_info: Dict,
                             target_n: int,
                             cache_path: Optional[Path]=None) -> Dict:
    if cache_path and cache_path.exists():
        saved    = json.loads(cache_path.read_text())
        all_cids = set(over_cluster_info.keys())
        mapped   = set()
        for v in saved.values(): mapped.update(v["cids"])
        if all_cids == mapped and len(saved) <= TARGET_MAX:
            log.info("Loaded consolidation from cache (%d cats).", len(saved))
            return saved
        log.warning("Consolidation cache invalid — re-running.")

    lines = []
    for cid, info in sorted(over_cluster_info.items()):
        top = ", ".join(info["top_labels"][:8])
        exs = " | ".join(f'"{s[:90]}"' for s in info["examples"][:2])
        lines.append(f"OC-{cid} (n={info['count']}): {top}\n  Ex: {exs}")

    prompt = CONSOLIDATE_SYS.format(
        target_n=target_n,
        dims="\n".join(f"  • {d}" for d in CANONICAL_DIMENSIONS),
        distinctions=DISTINCTION_RULES)

    user = (f"HARD REQUIREMENT: create EXACTLY {target_n} categories. "
            f"Every OC-ID (0–{max(over_cluster_info.keys())}) must be assigned.\n\n"
            + "\n".join(lines))

    raw = gpt(client, strong_model, prompt, user, json_mode=True, max_tokens=4000)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError(f"Consolidation JSON failed:\n{raw[:300]}")

    result   = {}
    all_cids = set(over_cluster_info.keys())
    mapped   = set()
    for cat in data.get("coarse_categories", []):
        name = cat["coarse_name"]
        cids = [int(c) for c in cat.get("over_cluster_ids", [])]
        result[name] = {
            "cids":                cids,
            "rationale":           cat.get("rationale",""),
            "definition":          cat.get("definition",""),
            "constraint_dimension":cat.get("constraint_dimension",""),
        }
        mapped.update(cids)

    # Assign any unmapped OCs
    unmapped = all_cids - mapped
    if unmapped:
        log.warning("Unmapped OCs after consolidation: %s", unmapped)
        cat_names = list(result.keys())
        for uid in unmapped:
            top = over_cluster_info[uid]["top_labels"][:4]
            sys2 = "Assign this over-cluster to the best category. JSON: {\"best\": \"name\"}"
            u2   = f"OC-{uid} labels: {', '.join(top)}\nCategories: {json.dumps(cat_names)}"
            raw2 = gpt(client, strong_model, sys2, u2, json_mode=True, max_tokens=80)
            try: best = json.loads(raw2).get("best", cat_names[0])
            except: best = cat_names[0]
            if best not in result: best = cat_names[0]
            result[best]["cids"].append(uid)
            mapped.add(uid)
            log.info("  OC-%d → '%s' (fallback)", uid, best)

    if cache_path:
        cache_path.write_text(json.dumps(result, indent=2))
    return result


# STRUCTURAL ENFORCEMENT — FORCED MERGE PASS
# THE KEY FIX: if n_coarse > TARGET_MAX, force a second merge round
FORCE_MERGE_SYS = """You are a research taxonomy expert.

The taxonomy below has too many coarse categories. You must MERGE them down
to EXACTLY {target_n} categories.

Rules:
   Merge categories that share the same ROOT CONSTRAINT DIMENSION
   Keep categories whose concepts are clearly distinct
   The result must have EXACTLY {target_n} categories
   Every category from the input must be assigned to an output category
   Use the name of the larger/broader category as the merged name, or create a new broader name if merging dissimilar ones

Return ONLY valid JSON:
{{
  "merged_categories": [
    {{
      "coarse_name": "final category name",
      "source_names": ["list of input category names merged here"],
      "rationale": "one sentence: why these belong together",
      "definition": "2–3 sentences: what this covers AND does NOT cover",
      "constraint_dimension": "which canonical dimension"
    }}
  ]
}}
"""

def forced_merge_pass(client, strong_model: str,
                       consolidation: Dict, target_n: int,
                       cache_path: Optional[Path]=None) -> Dict:
    """
    If len(consolidation) > target_n, force GPT to merge down.
    Returns a new consolidation dict with target_n categories.
    """
    if len(consolidation) <= target_n:
        return consolidation

    if cache_path and cache_path.exists():
        saved = json.loads(cache_path.read_text())
        if len(saved) <= target_n:
            log.info("Loaded forced merge from cache (%d cats).", len(saved))
            return saved
        log.warning("Forced merge cache has wrong count — re-running.")

    log.info("FORCED MERGE: %d → %d categories.", len(consolidation), target_n)
    prompt = FORCE_MERGE_SYS.format(target_n=target_n)

    # Show all current categories
    lines = []
    for name, meta in consolidation.items():
        lines.append(
            f'[{name}]\n'
            f'  Rationale: {meta.get("rationale","")}\n'
            f'  Definition: {meta.get("definition","")[:150]}\n'
            f'  n_members: {sum(1 for c in meta["cids"])}')

    user = (f"Current categories ({len(consolidation)} total). "
            f"Merge to EXACTLY {target_n}.\n\n" + "\n\n".join(lines))

    raw = gpt(client, strong_model, prompt, user, json_mode=True, max_tokens=3500)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        log.error("Forced merge JSON failed — keeping consolidation as-is.")
        return consolidation

    # Build name→source lookup from old consolidation
    old_name_to_meta = {k: v for k,v in consolidation.items()}

    new_consolidation = {}
    used_old_names    = set()

    for cat in data.get("merged_categories", []):
        new_name    = cat["coarse_name"]
        src_names   = cat.get("source_names", [])
        merged_cids = []
        for src in src_names:
            if src in old_name_to_meta:
                merged_cids.extend(old_name_to_meta[src]["cids"])
                used_old_names.add(src)
            else:
                # fuzzy match
                matches = difflib.get_close_matches(src, list(old_name_to_meta.keys()), n=1, cutoff=0.5)
                if matches:
                    merged_cids.extend(old_name_to_meta[matches[0]]["cids"])
                    used_old_names.add(matches[0])
        new_consolidation[new_name] = {
            "cids":                merged_cids,
            "rationale":           cat.get("rationale",""),
            "definition":          cat.get("definition",""),
            "constraint_dimension":cat.get("constraint_dimension",""),
        }

    # Assign any unmerged source categories
    unassigned = set(old_name_to_meta.keys()) - used_old_names
    if unassigned and new_consolidation:
        first_cat = next(iter(new_consolidation))
        for src in unassigned:
            new_consolidation[first_cat]["cids"].extend(old_name_to_meta[src]["cids"])
            log.warning("  Unassigned '%s' → '%s'", src, first_cat)

    log.info("Forced merge result: %d categories.", len(new_consolidation))
    if cache_path:
        cache_path.write_text(json.dumps(new_consolidation, indent=2))
    return new_consolidation


# STEP 5 — GENERATE GROUNDED FINE LABELS
FINE_LABELS_SYS = """You are a research taxonomy expert.

Coarse category: {coarse_name}
Rationale: {rationale}
Definition: {definition}
n_members: {n_members}
n_sub_clusters: {n_sub_clusters} (number of distinct over-clusters merged here)
max_fine_labels: {max_fine} — HARD MAXIMUM
min_fine_labels: {min_fine} — REQUIRED MINIMUM

UNIQUE TYPE LABELS in this category (frequency-ranked):
{label_lines}

REAL LIMITATION SENTENCES (most representative):
{sent_lines}


FOR EACH FINE LABEL:
  1. label: 2–5 words Title Case
  2. definition: 2–3 sentences — (a) exact constraint, (b) "Unlike [sibling]...",
    (c) evidence from real sentences above
  3. decision_rule: "Assign here if the limitation concerns..." (one sentence)
  4. not_clause: "This label does NOT cover..." (contrast with closest sibling)
  5. example_sentence: EXACT copy of one real sentence above (char-for-char)

Return ONLY valid JSON:
{{"fine_labels": [{{"label":"...","definition":"...","decision_rule":"Assign here if ...","not_clause":"This label does NOT cover ...","example_sentence":"..."}}]}}
"""

def generate_fine_labels(client, strong_model: str,
                          coarse_name: str, rationale: str, definition: str,
                          member_type_labels: List[str], central_sents: List[str],
                          n_members: int, n_sub_clusters: int,
                          cache_key: str, cache_path: Optional[Path]=None) -> List[dict]:
    min_fine = max(2, math.ceil(n_sub_clusters/2)) if n_sub_clusters >= 4 else 2
    max_fine = min(max(min_fine, n_members//3), MAX_FINE_HARD)
    max_fine = max(max_fine, min_fine)

    if cache_path and cache_path.exists():
        try:
            cd = json.loads(cache_path.read_text())
            if cache_key in cd:
                saved = cd[cache_key]
                if len(saved) > max_fine: saved = saved[:max_fine]
                log.info("  Fine labels cached '%s' (%d).", coarse_name, len(saved))
                return saved
        except Exception:
            pass

    label_counts = Counter(member_type_labels).most_common(70)
    label_lines  = "\n".join(f"  [{c:3d}x] {l}" for l,c in label_counts)
    sent_lines   = "\n".join(f"  {i+1}. {s[:190]}"
                              for i,s in enumerate(central_sents[:N_CENTRAL]))

    prompt = FINE_LABELS_SYS.format(
        coarse_name=coarse_name, rationale=rationale, definition=definition,
        n_members=n_members, n_sub_clusters=n_sub_clusters,
        max_fine=max_fine, min_fine=min_fine,
        label_lines=label_lines, sent_lines=sent_lines)

    raw = gpt(client, strong_model, "", prompt, json_mode=True,
              max_tokens=3500, temperature=0.1)
    try:
        fine_labels = json.loads(raw).get("fine_labels", [])
    except json.JSONDecodeError:
        log.warning("Fine label JSON failed for '%s'.", coarse_name)
        fine_labels = []

    if len(fine_labels) > max_fine:
        fine_labels = fine_labels[:max_fine]
    while len(fine_labels) < 2 and central_sents:
        fine_labels.append({
            "label":            f"{coarse_name} — Other",
            "definition":       f"Other limitations within {coarse_name}.",
            "decision_rule":    f"Assign here if the limitation concerns {coarse_name.lower()} but does not fit other sub-types.",
            "not_clause":       "",
            "example_sentence": central_sents[min(len(fine_labels), len(central_sents)-1)],
        })

    cd = {}
    if cache_path and cache_path.exists():
        try: cd = json.loads(cache_path.read_text())
        except: pass
    cd[cache_key] = fine_labels
    if cache_path:
        cache_path.write_text(json.dumps(cd, indent=2))
    return fine_labels


# STEP 6 — VERIFY EXAMPLE SENTENCES
def verify_examples(fine_labels: List[dict], real_sents: List[str]) -> List[dict]:
    for fl in fine_labels:
        ex = fl.get("example_sentence","")
        if not ex: continue
        matches = difflib.get_close_matches(ex, real_sents, n=1, cutoff=0.7)
        if matches:
            fl["example_sentence"] = matches[0]; fl["_verified"] = True
        else:
            best_r, best_s = 0.0, real_sents[0] if real_sents else ex
            for s in real_sents[:60]:
                r = difflib.SequenceMatcher(None, ex.lower(), s.lower()).ratio()
                if r > best_r: best_r, best_s = r, s
            fl["example_sentence"] = best_s; fl["_verified"] = False
    return fine_labels


# STEP 8 — ASSIGN FINE LABELS
ASSIGN_SYS = """Assign the limitation sentence to the SINGLE best fine label.
Pick based on ROOT CAUSE matching the decision rule.
Return ONLY valid JSON: {"fine_label": "exact label text"}
"""

def assign_fine_label(client, model: str, sentence: str,
                       coarse_name: str, fine_options: List[dict]) -> str:
    opts = "\n".join(
        f'  {i+1}. [{fl["label"]}]\n'
        f'     Rule: {fl.get("decision_rule","")[:160]}\n'
        f'     NOT:  {fl.get("not_clause","")[:120]}'
        for i,fl in enumerate(fine_options))
    user = f'Coarse: "{coarse_name}"\nSentence: "{str(sentence)[:280]}"\n\nOptions:\n{opts}'
    raw  = gpt(client, model, ASSIGN_SYS, user, json_mode=True, max_tokens=80)
    try: label = json.loads(raw).get("fine_label","").strip()
    except: label = ""
    valid = [fl["label"] for fl in fine_options]
    if label in valid: return label
    for v in valid:
        if label.lower() in v.lower() or v.lower() in label.lower(): return v
    return valid[0]


# STEP 7 — GLOBAL DISTINCTNESS PASS
DISTINCT_SYS = """You are a taxonomy auditor. Check for:
TYPE A: within-category fine labels that could confuse a coder
TYPE B: cross-category fine labels with the same root cause (specify keep/drop)
TYPE C: duplicate fine label names across categories

Return ONLY valid JSON:
{
  "type_a_rewrites": [{"coarse":"...","fine_label":"...","new_definition":"...","new_decision_rule":"Assign here if ...","new_not_clause":"This label does NOT cover ...","confused_with":"..."}],
  "type_b_overlaps": [{"keep_coarse":"...","keep_fine":"...","drop_coarse":"...","drop_fine":"...","reason":"..."}],
  "type_c_renames": [{"coarse":"...","old_label":"...","new_label":"..."}],
  "no_issues_found": true_or_false
}
"""

def global_distinctness_pass(client, strong_model: str, full_taxonomy: Dict,
                              cache_path: Optional[Path]=None) -> Dict:
    if cache_path and cache_path.exists():
        saved = json.loads(cache_path.read_text())
        log.info("Loaded distinctness pass from cache.")
        return saved
    lines = []
    for coarse, meta in full_taxonomy.items():
        lines.append(f'\n## {coarse}\n{meta.get("definition","")[:150]}')
        for fl in meta["fine_labels"]:
            lines.append(f'  [{fl["label"]}]: {fl.get("definition","")[:200]}\n'
                         f'    Rule: {fl.get("decision_rule","")[:140]}')
    user = "Full taxonomy:\n" + "\n".join(lines)
    raw  = gpt(client, strong_model, DISTINCT_SYS, user, json_mode=True, max_tokens=3500)
    try: data = json.loads(raw)
    except:
        data = {"type_a_rewrites":[],"type_b_overlaps":[],"type_c_renames":[],"no_issues_found":True}

    result = {"type_a":{},"type_b":data.get("type_b_overlaps",[]),"type_c":{}}
    for rw in data.get("type_a_rewrites",[]):
        key = f'{rw.get("coarse","")}__{rw.get("fine_label","")}'
        result["type_a"][key] = {k: rw.get(k,"") for k in
                                  ["new_definition","new_decision_rule","new_not_clause"]}
        log.info("TYPE A: [%s / %s] ← confused with '%s'",
                 rw.get("coarse",""), rw.get("fine_label",""), rw.get("confused_with",""))
    for ov in data.get("type_b_overlaps",[]):
        log.info("TYPE B: DROP [%s / %s], KEEP [%s / %s]",
                 ov.get("drop_coarse",""), ov.get("drop_fine",""),
                 ov.get("keep_coarse",""), ov.get("keep_fine",""))
    for rn in data.get("type_c_renames",[]):
        key = f'{rn.get("coarse","")}__{rn.get("old_label","")}'
        result["type_c"][key] = rn.get("new_label","")
        log.info("TYPE C: [%s / %s] → '%s'",
                 rn.get("coarse",""), rn.get("old_label",""), rn.get("new_label",""))
    if data.get("no_issues_found",False):
        log.info("Distinctness pass: no issues found.")
    if cache_path:
        cache_path.write_text(json.dumps(result, indent=2))
    return result

def apply_distinctness(full_taxonomy: Dict, distincts: Dict, df: pd.DataFrame) -> Dict:
    for coarse, meta in full_taxonomy.items():
        for fl in meta["fine_labels"]:
            key = f'{coarse}__{fl["label"]}'
            if key in distincts["type_a"]:
                rw = distincts["type_a"][key]
                if rw.get("new_definition"):    fl["definition"]    = rw["new_definition"]
                if rw.get("new_decision_rule"): fl["decision_rule"] = rw["new_decision_rule"]
                if rw.get("new_not_clause"):    fl["not_clause"]    = rw["new_not_clause"]
    for ov in distincts.get("type_b",[]):
        dc, df_name = ov.get("drop_coarse",""), ov.get("drop_fine","")
        kc, kf      = ov.get("keep_coarse",""), ov.get("keep_fine","")
        if dc in full_taxonomy and "fine_label" in df.columns:
            full_taxonomy[dc]["fine_labels"] = [
                fl for fl in full_taxonomy[dc]["fine_labels"] if fl["label"] != df_name]
            mask = (df["coarse_label"]==dc) & (df["fine_label"]==df_name)
            if mask.sum()>0:
                df.loc[mask,"coarse_label"] = kc
                df.loc[mask,"fine_label"]   = kf
                log.info("  Reassigned %d rows → [%s / %s]", mask.sum(), kc, kf)
    for coarse, meta in full_taxonomy.items():
        for fl in meta["fine_labels"]:
            key = f'{coarse}__{fl["label"]}'
            if key in distincts["type_c"] and distincts["type_c"][key]:
                old = fl["label"]; fl["label"] = distincts["type_c"][key]
                if "fine_label" in df.columns:
                    df.loc[(df["coarse_label"]==coarse)&(df["fine_label"]==old),"fine_label"] = fl["label"]
    return full_taxonomy


# PURGE + TABLE
def purge_zero_count(df: pd.DataFrame, full_taxonomy: Dict) -> Dict:
    cleaned = {}
    for coarse, meta in full_taxonomy.items():
        counts  = df[df["coarse_label"]==coarse]["fine_label"].value_counts()
        valid   = [fl for fl in meta["fine_labels"]
                   if fl["label"] in counts and counts[fl["label"]]>0]
        purged  = [fl["label"] for fl in meta["fine_labels"]
                   if fl["label"] not in counts or counts[fl["label"]]==0]
        if purged:
            log.warning("Purging 0-count in '%s': %s", coarse, purged)
            fallback = valid[0]["label"] if valid else coarse
            mask = (df["coarse_label"]==coarse) & (df["fine_label"].isin(purged))
            df.loc[mask,"fine_label"] = fallback
        cleaned[coarse] = {**meta,"fine_labels": valid or meta["fine_labels"][:1]}
    return cleaned

def build_table(df: pd.DataFrame, full_taxonomy: Dict) -> pd.DataFrame:
    total  = len(df)
    def_lk = {(c,fl["label"]): fl for c,meta in full_taxonomy.items()
               for fl in meta["fine_labels"]}
    rows = []
    for coarse, grp in df.groupby("coarse_label"):
        meta = full_taxonomy.get(coarse, {})
        for fine, fg in grp.groupby("fine_label"):
            fl = def_lk.get((coarse,fine), {})
            rows.append({
                "COARSE": coarse, "CONSTRAINT_DIM": meta.get("constraint_dimension",""),
                "COARSE_RATIONALE": meta.get("rationale",""),
                "COARSE_DEFINITION": meta.get("definition",""),
                "FINE": fine,
                "FINE_DEFINITION": fl.get("definition",""),
                "DECISION_RULE": fl.get("decision_rule",""),
                "NOT_CLAUSE": fl.get("not_clause",""),
                "EXAMPLE_SENTENCE": fl.get("example_sentence",""),
                "EXAMPLE_VERIFIED": fl.get("_verified", False),
                "count": len(fg),
                "freq_pct": round(100*len(fg)/total, 2),
            })
    return (pd.DataFrame(rows)
              .sort_values(["COARSE","count"], ascending=[True,False])
              .reset_index(drop=True))


# MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",        required=True)
    ap.add_argument("--col",          default="limitation_clean")
    ap.add_argument("--sample",       type=int, default=None)
    ap.add_argument("--out",          default="taxonomy_output")
    ap.add_argument("--cache",        default="cache")
    ap.add_argument("--over_n",       type=int, default=OVER_CLUSTER_N)
    ap.add_argument("--n_coarse",     type=int, default=0,
                    help="Force exact coarse count (0=auto pick midpoint of range)")
    ap.add_argument("--model",        default=DEFAULT_MODEL)
    ap.add_argument("--strong_model", default=DEFAULT_STRONG_MODEL)
    args = ap.parse_args()

    out   = Path(args.out);   out.mkdir(exist_ok=True)
    cache = Path(args.cache); cache.mkdir(exist_ok=True)

    log.info("=" * 65)
    log.info("MODEL SETUP")
    log.info("  Cheap  (steps 1,8): %s", args.model)
    log.info("  Strong (steps 4,5,7): %s", args.strong_model)
    log.info("=" * 65)

    client = make_client(args.model)
    if args.strong_model != args.model:
        make_client(args.strong_model)

    df    = pd.read_csv(args.input).dropna(subset=[args.col]).reset_index(drop=True)
    if args.sample:
        df = df.sample(min(args.sample, len(df)), random_state=42).reset_index(drop=True)
    n_rows = len(df)

    # Auto-scale over_n — less aggressive for small samples
    if n_rows < 200:
        over_n = max(5, n_rows // 5)   # 100 rows → over_n=20, but see below
        over_n = min(over_n, n_rows // 4)
        over_n = max(over_n, TARGET_MAX + 3)  # must be > TARGET_MAX for merge to work
    else:
        over_n = min(args.over_n, max(TARGET_MAX+3, n_rows//8))
    over_n = min(over_n, n_rows - 1)  # cannot exceed n_rows

    # Target coarse count
    target_n = args.n_coarse if args.n_coarse > 0 else (TARGET_MIN + TARGET_MAX) // 2  # 10

    log.info("Loaded %d limitations. over_n=%d, target_n_coarse=%d.",
             n_rows, over_n, target_n)
    if n_rows < 200:
        log.warning("SMALL SAMPLE (%d rows): taxonomy will be approximate. "
                    "Run with >=500 rows for stable results.", n_rows)

    # STEP 1
    log.info("=" * 65); log.info("STEP 1 — Extract type labels [%s]", args.model); log.info("=" * 65)
    type_labels = extract_type_labels(client, args.model, df, args.col,
                                       cache_path=cache/"type_labels.json")
    df["raw_type_label"] = type_labels
    for lbl, cnt in Counter(type_labels).most_common(20):
        log.info("  [%3d]  %s", cnt, lbl)

    # STEP 2
    log.info("=" * 65); log.info("STEP 2 — Embed type labels"); log.info("=" * 65)
    X_raw = embed(client, type_labels, cache_path=cache/"label_embeddings.npy")
    X     = normalize(X_raw)

    # STEP 3
    log.info("=" * 65); log.info("STEP 3 — Over-cluster (n=%d)", over_n); log.info("=" * 65)
    cluster_ids = ward_cluster(X, over_n)
    df["over_cluster_id"] = cluster_ids
    centroids = np.zeros((over_n, X.shape[1]))
    for cid in range(over_n):
        mask = cluster_ids == cid
        if mask.sum() > 0:
            centroids[cid] = X[mask].mean(0)
    raw_sents = df[args.col].tolist()
    over_info = {}
    for cid in range(over_n):
        mask    = cluster_ids == cid
        members = list(np.where(mask)[0])
        mlabels = [type_labels[i] for i in members]
        exs     = central_sentences(raw_sents, members, X, centroids[cid])
        top_lbl = [l for l,_ in Counter(mlabels).most_common(12)]
        over_info[cid] = {"top_labels":top_lbl,"examples":exs,
                           "count":int(mask.sum()),"members_idx":members}
        log.info("  OC-%2d (n=%d): %s", cid, mask.sum(), ", ".join(top_lbl[:5]))

    # STEP 4 — Semantic consolidation
    log.info("=" * 65)
    log.info("STEP 4 — Semantic consolidation → exactly %d coarse categories [%s]",
             target_n, args.strong_model)
    log.info("=" * 65)
    consolidation = semantic_consolidation(
        client, args.strong_model, over_info, target_n,
        cache_path=cache/"consolidation.json")

    # STRUCTURAL ENFORCEMENT — if GPT produced too many, force merge
    if len(consolidation) > TARGET_MAX:
        log.warning("Got %d coarse categories (max %d) — running forced merge.",
                    len(consolidation), TARGET_MAX)
        consolidation = forced_merge_pass(
            client, args.strong_model, consolidation, target_n,
            cache_path=cache/"forced_merge.json")

    # If still too many, do a simple greedy merge of smallest categories
    while len(consolidation) > TARGET_MAX:
        log.warning("Still %d categories — greedy merging smallest.", len(consolidation))
        sorted_cats = sorted(consolidation.items(), key=lambda x: len(x[1]["cids"]))
        smallest_name, smallest_meta = sorted_cats[0]
        second_name,  second_meta   = sorted_cats[1]
        merged_cids = smallest_meta["cids"] + second_meta["cids"]
        consolidation[second_name]["cids"] = merged_cids
        del consolidation[smallest_name]
        log.info("  Greedy merged '%s' → '%s'", smallest_name, second_name)

    log.info("Final coarse count: %d (target: %d–%d)", len(consolidation), TARGET_MIN, TARGET_MAX)

    # Map rows to coarse labels
    cid_to_coarse = {}
    for coarse, meta in consolidation.items():
        for cid in meta["cids"]:
            cid_to_coarse[int(cid)] = coarse
    df["coarse_label"] = df["over_cluster_id"].map(cid_to_coarse)
    null_count = df["coarse_label"].isna().sum()
    if null_count > 0:
        log.error("%d rows unassigned — forcing to first category.", null_count)
        df["coarse_label"].fillna(next(iter(consolidation)), inplace=True)

    log.info("Coarse distribution:")
    for cat, cnt in df["coarse_label"].value_counts().items():
        log.info("  %5d (%4.1f%%)  %s", cnt, 100*cnt/n_rows, cat)

    # STEP 5 — Fine labels
    log.info("=" * 65)
    log.info("STEP 5 — Generate fine labels [%s]", args.strong_model)
    log.info("=" * 65)
    fine_cache   = cache/"fine_labels_grounded.json"
    full_taxonomy = {}
    for coarse, cmeta in tqdm(consolidation.items(), desc="Fine labels"):
        member_idx     = [i for i in range(n_rows) if df.at[i,"coarse_label"]==coarse]
        n_members      = len(member_idx)
        n_sub_clusters = len(cmeta["cids"])
        member_labels  = [type_labels[i] for i in member_idx]
        coarse_cent    = X[member_idx].mean(0) if member_idx else np.zeros(X.shape[1])
        cent_sents     = central_sentences(raw_sents, member_idx, X, coarse_cent)
        cache_key      = re.sub(r'\W+','_', coarse.lower())

        fine_labels = generate_fine_labels(
            client, args.strong_model, coarse,
            cmeta.get("rationale",""), cmeta.get("definition",""),
            member_labels, cent_sents, n_members, n_sub_clusters,
            cache_key, fine_cache)
        fine_labels = verify_examples(fine_labels, cent_sents)

        min_f = max(2, math.ceil(n_sub_clusters/2)) if n_sub_clusters>=4 else 2
        max_f = min(max(min_f, n_members//3), MAX_FINE_HARD)
        full_taxonomy[coarse] = {
            "rationale":           cmeta.get("rationale",""),
            "definition":          cmeta.get("definition",""),
            "constraint_dimension":cmeta.get("constraint_dimension",""),
            "fine_labels":         fine_labels,
            "n_members":           n_members,
            "n_sub_clusters":      n_sub_clusters,
        }
        log.info("  %-42s → %d fine labels (n=%d, sub=%d, range=%d–%d)",
                 coarse, len(fine_labels), n_members, n_sub_clusters, min_f, max_f)
        for fl in fine_labels:
            log.info("      • %s", fl["label"])
        time.sleep(0.4)

    # STEP 8 — Assign fine labels
    log.info("=" * 65); log.info("STEP 8 — Assign fine labels [%s]", args.model); log.info("=" * 65)
    fine_assign_cache = cache/"fine_assignments.json"
    fine_lookup = {k: v["fine_labels"] for k,v in full_taxonomy.items()}
    saved_fine  = False
    if fine_assign_cache.exists():
        saved = json.loads(fine_assign_cache.read_text())
        if len(saved) == n_rows:
            df["fine_label"] = saved; log.info("Loaded fine assignments from cache.")
            saved_fine = True
    if not saved_fine:
        fine_col_vals = []
        for _, row in tqdm(df.iterrows(), total=n_rows, desc="Assign fine"):
            coarse = row["coarse_label"]
            opts   = fine_lookup.get(coarse, [])
            fine   = assign_fine_label(client, args.model, str(row[args.col]), coarse, opts) if opts else coarse
            fine_col_vals.append(fine)
            time.sleep(0.08)
        df["fine_label"] = fine_col_vals
        fine_assign_cache.write_text(json.dumps(fine_col_vals, indent=2))

    # STEP 7 — Global distinctness
    log.info("=" * 65); log.info("STEP 7 — Global distinctness [%s]", args.strong_model); log.info("=" * 65)
    distincts = global_distinctness_pass(client, args.strong_model, full_taxonomy,
                                          cache_path=cache/"distinctness_pass.json")
    full_taxonomy = apply_distinctness(full_taxonomy, distincts, df)

    # Purge
    log.info("Purging zero-count fine labels ...")
    full_taxonomy = purge_zero_count(df, full_taxonomy)

    # Save
    log.info("=" * 65); log.info("SAVING → %s", out); log.info("=" * 65)
    df.to_csv(out/"limitations_annotated.csv", index=False)
    table = build_table(df, full_taxonomy)
    table.to_csv(out/"taxonomy_table.csv", index=False)

    final_json = {}
    for coarse, meta in full_taxonomy.items():
        coarse_rows = df[df["coarse_label"]==coarse]
        final_json[coarse] = {
            "constraint_dimension": meta.get("constraint_dimension",""),
            "rationale":            meta["rationale"],
            "definition":           meta["definition"],
            "n_sub_clusters":       meta.get("n_sub_clusters",0),
            "count":                int(len(coarse_rows)),
            "pct":                  round(100*len(coarse_rows)/n_rows,1),
            "fine_labels": [
                {
                    "label":            fl["label"],
                    "definition":       fl.get("definition",""),
                    "decision_rule":    fl.get("decision_rule",""),
                    "not_clause":       fl.get("not_clause",""),
                    "example_sentence": fl.get("example_sentence",""),
                    "example_verified": fl.get("_verified",False),
                    "count": int(df[(df["coarse_label"]==coarse) &
                                   (df["fine_label"]==fl["label"])].shape[0]),
                }
                for fl in meta["fine_labels"]
            ]
        }
    (out/"taxonomy_coarse_fine.json").write_text(
        json.dumps(final_json, indent=2, ensure_ascii=False))
    log.info("Saved all outputs.")

    # Print
    total = n_rows
    fine_counts = [len(m["fine_labels"]) for m in final_json.values()]
    print("\n" + "=" * 86)
    print("FINAL TWO-LEVEL TAXONOMY")
    print("=" * 86)
    for coarse, meta in final_json.items():
        n_c = meta["count"]; pct = meta["pct"]
        n_f = len(meta["fine_labels"]); sc = meta.get("n_sub_clusters",0)
        print(f"\n▸ {coarse}  |  {n_c} rows ({pct}%)  |  {n_f} fine  [{sc} sub-clusters]")
        print(f"  Dimension : {meta.get('constraint_dimension','')}")
        print(f"  Rationale : {meta['rationale']}")
        print(f"  Definition: {meta['definition'][:180]}")
        print()
        for fl in sorted(meta["fine_labels"], key=lambda x: x["count"], reverse=True):
            if fl["count"] == 0: continue
            fp  = round(100*fl["count"]/total, 1)
            ver = "✓" if fl.get("example_verified") else "~"
            print(f"  • {fl['label']:<38} n={fl['count']:>4}  ({fp:>4.1f}%)  {ver}")
            print(f"    Def : {fl['definition'][:150]}...")
            print(f"    Rule: {fl.get('decision_rule','')[:140]}")
            print(f"    NOT : {fl.get('not_clause','')[:130]}")
            print(f"    Ex  : {fl.get('example_sentence','')[:110]}")
            print()

    print("─" * 86)
    n_coarse = df["coarse_label"].nunique()
    n_fine   = df["fine_label"].nunique()
    print(f"Total: {total} rows | {n_coarse} coarse | {n_fine} fine (n>0)")
    if fine_counts:
        print(f"Fine distribution: min={min(fine_counts)}, max={max(fine_counts)}, "
              f"mean={sum(fine_counts)/len(fine_counts):.1f}")
    print(f"\nOutputs: {out}/")

if __name__ == "__main__":
    main()

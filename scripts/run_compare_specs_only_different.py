# run:
# python -m nl2spec.scripts.run_compare_specs_only_different

import json
import difflib
import itertools
import re
from pathlib import Path

import pandas as pd


# ---------------- CONFIG PATHS ----------------

ORIGINAL_PATH = Path(
    r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\datasets\baseline_ir"
)
RANDOM_PATH = Path(
    r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\output\llm\openAI\gpt-4o\random\one_k1"
)
STRUCTURAL_PATH = Path(
    r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\output\llm\openAI\gpt-4o\structural\one_k1"
)

RESULT_PATH = Path(
    r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\output\llm\results_only_different_active"
)

INDIVIDUAL_METRICS_PATH = RESULT_PATH / "individual" / "metrics"
INDIVIDUAL_MATRICES_PATH = RESULT_PATH / "individual" / "matrices"
PLOTS_PATH = RESULT_PATH / "plots"

ACTIVE_METRIC_COLS = [
    "levenshtein_similarity",
    "sequence_similarity",
    "jaccard_similarity",
    "tree_similarity",
]


# ---------------- HELPERS ----------------

def ensure_dirs():
    RESULT_PATH.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_METRICS_PATH.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_MATRICES_PATH.mkdir(parents=True, exist_ok=True)
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)


def canonical_json_string(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def levenshtein(a, b):
    n, m = len(a), len(b)

    if n > m:
        a, b = b, a
        n, m = m, n

    current = list(range(n + 1))

    for i in range(1, m + 1):
        previous, current = current, [i] + [0] * n
        for j in range(1, n + 1):
            add = previous[j] + 1
            delete = current[j - 1] + 1
            change = previous[j - 1] + (a[j - 1] != b[i - 1])
            current[j] = min(add, delete, change)

    return current[n]


def levenshtein_similarity(a, b):
    return 1 - levenshtein(a, b) / max(len(a), len(b), 1)


def sequence_similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def tokenize(text):
    return set(re.findall(r"[A-Za-z_]+", text))


def jaccard_similarity(a, b):
    t1, t2 = tokenize(a), tokenize(b)
    union = t1 | t2
    return len(t1 & t2) / len(union) if union else 1.0


def flatten_json(obj, prefix="root"):
    if isinstance(obj, dict):
        paths = []
        for k, v in obj.items():
            paths.extend(flatten_json(v, f"{prefix}.{k}"))
        return paths

    if isinstance(obj, list):
        paths = []
        for i, v in enumerate(obj):
            paths.extend(flatten_json(v, f"{prefix}[{i}]"))
        return paths

    return [prefix]


def tree_similarity(a, b):
    p1, p2 = set(flatten_json(a)), set(flatten_json(b))
    return len(p1 & p2) / max(len(p1), len(p2), 1)


def semantic_ir_match(a, b):
    return canonical_json_string(a) == canonical_json_string(b)


def compute_avg(row):
    vals = [float(row[m]) for m in ACTIVE_METRIC_COLS]
    return round(sum(vals) / len(vals), 2)


def safe_read_json(path):
    try:
        txt = Path(path).read_text(encoding="utf-8")
        obj = json.loads(txt)
        return txt, obj, None
    except Exception as e:
        return None, None, f"{path} -> {e}"


def find_matching_file(root, name):
    matches = list(root.rglob(name))
    return matches[0] if matches else None


def detect_formalism(json_obj):
    if not isinstance(json_obj, dict):
        return "unknown"

    if "formalism" in json_obj and json_obj["formalism"]:
        return str(json_obj["formalism"]).lower()

    ir = json_obj.get("ir", {})
    if isinstance(ir, dict):
        for key in ("event", "ere", "fsm", "ltl"):
            if key in ir:
                return key

    return "unknown"


# ---------------- METRICS ----------------

def pair_metrics(label_a, label_b, ja, jb, ta, tb, name, formalism):
    row = {
        "spec": name,
        "formalism": formalism,
        "comparison": f"{label_a}_vs_{label_b}",
        "semantic_exact_match": int(semantic_ir_match(ja, jb)),
        "levenshtein_similarity": round(levenshtein_similarity(ta, tb) * 100, 2),
        "sequence_similarity": round(sequence_similarity(ta, tb) * 100, 2),
        "jaccard_similarity": round(jaccard_similarity(ta, tb) * 100, 2),
        "tree_similarity": round(tree_similarity(ja, jb) * 100, 2),
    }
    row["average_similarity_score"] = compute_avg(row)
    return row


def compare_files(baseline_file, random_file, structural_file):
    texts = {}
    jsons = {}

    for label, path in [
        ("baseline", baseline_file),
        ("random", random_file),
        ("structural", structural_file),
    ]:
        txt, obj, err = safe_read_json(path)
        if err:
            raise ValueError(err)
        texts[label] = txt
        jsons[label] = obj

    formalism = detect_formalism(jsons["baseline"])
    spec_name = baseline_file.name

    rows = []
    for a, b in itertools.combinations(["baseline", "random", "structural"], 2):
        rows.append(
            pair_metrics(
                a,
                b,
                jsons[a],
                jsons[b],
                texts[a],
                texts[b],
                spec_name,
                formalism,
            )
        )

    return pd.DataFrame(rows)


def build_matrix(df):
    metric_cols = [
        "semantic_exact_match",
        "levenshtein_similarity",
        "sequence_similarity",
        "jaccard_similarity",
        "tree_similarity",
        "average_similarity_score",
    ]

    matrix = df[["comparison"] + metric_cols].copy()
    return matrix


# ---------------- EXTRACTION ----------------

def extract_different(global_df):
    rows = []

    for spec in sorted(global_df["spec"].unique()):
        subset = global_df[global_df["spec"] == spec]

        br = subset[subset["comparison"] == "baseline_vs_random"]
        bs = subset[subset["comparison"] == "baseline_vs_structural"]
        rs = subset[subset["comparison"] == "random_vs_structural"]

        if br.empty or bs.empty:
            continue

        formalism = subset["formalism"].iloc[0]

        br_eq = int(br["semantic_exact_match"].iloc[0])
        bs_eq = int(bs["semantic_exact_match"].iloc[0])

        if br_eq == 1 and bs_eq == 1:
            continue

        br_score = float(br["average_similarity_score"].iloc[0])
        bs_score = float(bs["average_similarity_score"].iloc[0])
        rs_score = float(rs["average_similarity_score"].iloc[0]) if not rs.empty else 0.0

        winner = "tie"
        if bs_score > br_score:
            winner = "structural"
        elif br_score > bs_score:
            winner = "random"

        rows.append(
            {
                "spec": spec,
                "formalism": formalism,
                "winner": winner,
                "difference_%": round(abs(bs_score - br_score), 2),
                "random_score": round(br_score, 2),
                "structural_score": round(bs_score, 2),
                "random_vs_structural_score": round(rs_score, 2),
                "baseline_random_semantic_match": br_eq,
                "baseline_structural_semantic_match": bs_eq,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "spec",
                "formalism",
                "winner",
                "difference_%",
                "random_score",
                "structural_score",
                "random_vs_structural_score",
                "baseline_random_semantic_match",
                "baseline_structural_semantic_match",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        by=["winner", "difference_%", "spec"],
        ascending=[True, False, True]
    ).reset_index(drop=True)


# ---------------- SCOREBOARD ----------------

def compute_overall(global_df):
    rows = []

    for spec in sorted(global_df["spec"].unique()):
        subset = global_df[global_df["spec"] == spec]

        br = subset[subset["comparison"] == "baseline_vs_random"]
        bs = subset[subset["comparison"] == "baseline_vs_structural"]

        if br.empty or bs.empty:
            continue

        formalism = subset["formalism"].iloc[0]
        br_score = float(br["average_similarity_score"].iloc[0])
        bs_score = float(bs["average_similarity_score"].iloc[0])

        winner = "tie"
        diff = 0.0

        if bs_score > br_score:
            winner = "structural"
            diff = bs_score - br_score
        elif br_score > bs_score:
            winner = "random"
            diff = br_score - bs_score

        rows.append(
            {
                "spec": spec,
                "formalism": formalism,
                "winner": winner,
                "difference_%": round(diff, 2),
            }
        )

    if not rows:
        return pd.DataFrame(
            [
                {
                    "structural_closer_to_baseline": 0,
                    "random_closer_to_baseline": 0,
                    "ties": 0,
                    "average_advantage_%": 0.0,
                    "structural_pct_of_total": 0.0,
                    "random_pct_of_total": 0.0,
                    "ties_pct_of_total": 0.0,
                    "total_scenarios": 0,
                }
            ]
        )

    df = pd.DataFrame(rows)
    total = len(df)

    structural_count = int((df["winner"] == "structural").sum())
    random_count = int((df["winner"] == "random").sum())
    tie_count = int((df["winner"] == "tie").sum())

    non_tie_df = df[df["winner"] != "tie"]
    avg_advantage = round(non_tie_df["difference_%"].mean(), 2) if not non_tie_df.empty else 0.0

    return pd.DataFrame(
        [
            {
                "structural_closer_to_baseline": structural_count,
                "random_closer_to_baseline": random_count,
                "ties": tie_count,
                "average_advantage_%": avg_advantage,
                "structural_pct_of_total": round((structural_count / total) * 100, 2),
                "random_pct_of_total": round((random_count / total) * 100, 2),
                "ties_pct_of_total": round((tie_count / total) * 100, 2),
                "total_scenarios": total,
            }
        ]
    )


def compute_overall_only_different(diff_df):
    if diff_df.empty:
        return pd.DataFrame(
            [
                {
                    "structural_closer_to_baseline": 0,
                    "random_closer_to_baseline": 0,
                    "ties": 0,
                    "average_advantage_%": 0.0,
                    "structural_pct_of_total": 0.0,
                    "random_pct_of_total": 0.0,
                    "ties_pct_of_total": 0.0,
                    "total_scenarios": 0,
                }
            ]
        )

    total = len(diff_df)

    structural_count = int((diff_df["winner"] == "structural").sum())
    random_count = int((diff_df["winner"] == "random").sum())
    tie_count = int((diff_df["winner"] == "tie").sum())

    non_tie_df = diff_df[diff_df["winner"] != "tie"]
    avg_advantage = (
        round(non_tie_df["difference_%"].mean(), 2) if not non_tie_df.empty else 0.0
    )

    return pd.DataFrame(
        [
            {
                "structural_closer_to_baseline": structural_count,
                "random_closer_to_baseline": random_count,
                "ties": tie_count,
                "average_advantage_%": avg_advantage,
                "structural_pct_of_total": round((structural_count / total) * 100, 2),
                "random_pct_of_total": round((random_count / total) * 100, 2),
                "ties_pct_of_total": round((tie_count / total) * 100, 2),
                "total_scenarios": total,
            }
        ]
    )


# ---------------- MAIN ----------------

def main():
    ensure_dirs()

    if not ORIGINAL_PATH.exists():
        raise FileNotFoundError(f"Original path not found: {ORIGINAL_PATH}")

    if not RANDOM_PATH.exists():
        raise FileNotFoundError(f"Random path not found: {RANDOM_PATH}")

    if not STRUCTURAL_PATH.exists():
        raise FileNotFoundError(f"Structural path not found: {STRUCTURAL_PATH}")

    random_files = sorted(RANDOM_PATH.rglob("*.json"))

    if not random_files:
        raise FileNotFoundError(f"No JSON files found in RANDOM_PATH: {RANDOM_PATH}")

    global_metrics = []
    skipped = []

    print(f"Found {len(random_files)} random JSON files.")

    for random_file in random_files:
        name = random_file.name

        baseline_file = find_matching_file(ORIGINAL_PATH, name)
        structural_file = find_matching_file(STRUCTURAL_PATH, name)

        if baseline_file is None or structural_file is None:
            skipped.append(
                {
                    "spec": name,
                    "reason": "missing baseline or structural file",
                }
            )
            print(f"[SKIP] {name} -> baseline or structural not found")
            continue

        try:
            df = compare_files(baseline_file, random_file, structural_file)
            global_metrics.append(df)

            df.to_csv(
                INDIVIDUAL_METRICS_PATH / f"{Path(name).stem}_metrics.csv",
                index=False,
                encoding="utf-8",
            )

            matrix_df = build_matrix(df)
            matrix_df.to_csv(
                INDIVIDUAL_MATRICES_PATH / f"{Path(name).stem}_matrix.csv",
                index=False,
                encoding="utf-8",
            )

            print(f"[OK] {name}")

        except Exception as e:
            skipped.append(
                {
                    "spec": name,
                    "reason": str(e),
                }
            )
            print(f"[ERROR] {name} -> {e}")

    skipped_df = pd.DataFrame(skipped)
    skipped_df.to_csv(RESULT_PATH / "skipped_files.csv", index=False, encoding="utf-8")

    if not global_metrics:
        raise RuntimeError("No valid comparisons were generated.")

    global_df = pd.concat(global_metrics, ignore_index=True)
    global_df = global_df.sort_values(
        by=["formalism", "spec", "comparison"]
    ).reset_index(drop=True)
    global_df.to_csv(RESULT_PATH / "global_metrics.csv", index=False, encoding="utf-8")

    diff_df = extract_different(global_df)
    diff_df.to_csv(
        RESULT_PATH / "different_scenarios.csv",
        index=False,
        encoding="utf-8",
    )

    overall_df = compute_overall(global_df)
    overall_df.to_csv(
        RESULT_PATH / "overall_scoreboard.csv",
        index=False,
        encoding="utf-8",
    )

    overall_diff_df = compute_overall_only_different(diff_df)
    overall_diff_df.to_csv(
        RESULT_PATH / "overall_scoreboard_only_different.csv",
        index=False,
        encoding="utf-8",
    )

    print("\nDone.")
    print(f"Results saved in: {RESULT_PATH}")
    print(f"Compared scenarios: {global_df['spec'].nunique()}")
    print(f"Different scenarios: {len(diff_df)}")
    print(f"Skipped files: {len(skipped_df)}")


if __name__ == "__main__":
    main()
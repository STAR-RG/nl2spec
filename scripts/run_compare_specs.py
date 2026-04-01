import json
import difflib
import itertools
import re
import shutil
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon
import matplotlib.pyplot as plt

#run: python -m nl2spec.scripts.run_compare_specs

# ---------------- CONFIG PATHS ----------------

ORIGINAL_PATH = Path(r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\datasets\baseline_ir")
RANDOM_PATH = Path(r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\output\llm\openAI\gpt-4o\random\few_k3")
STRUCTURAL_PATH = Path(r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\output\llm\openAI\gpt-4o\structural\few_k3")
RESULT_PATH = Path(r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\output\llm\results")

INDIVIDUAL_METRICS_PATH = RESULT_PATH / "individual" / "metrics"
INDIVIDUAL_MATRICES_PATH = RESULT_PATH / "individual" / "matrices"

# ---------------- PREPARE RESULT FOLDER ----------------

def prepare_results():
    if RESULT_PATH.exists():
        resp = input("Results folder exists. Delete and recreate? (y/n): ")
        if resp.lower() == "y":
            shutil.rmtree(RESULT_PATH)

    RESULT_PATH.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_METRICS_PATH.mkdir(parents=True, exist_ok=True)
    INDIVIDUAL_MATRICES_PATH.mkdir(parents=True, exist_ok=True)


# ---------------- ALGORITHMS ----------------

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
            change = previous[j - 1]

            if a[j - 1] != b[i - 1]:
                change += 1

            current[j] = min(add, delete, change)

    return current[n]


def lev_sim(a, b):
    return 1 - levenshtein(a, b) / max(len(a), len(b))


def diff_sim(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def flatten_json(obj, prefix=""):
    items = []

    if isinstance(obj, dict):
        for k, v in obj.items():
            items += flatten_json(v, f"{prefix}.{k}")

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            items += flatten_json(v, f"{prefix}[{i}]")

    else:
        items.append(prefix)

    return items


def structural_sim(a, b):
    s1 = set(flatten_json(a))
    s2 = set(flatten_json(b))
    union = len(s1 | s2)
    return len(s1 & s2) / union if union else 1.0


def tokenize(text):
    return set(re.findall(r"[A-Za-z_]+", text))


def jaccard_sim(a, b):
    t1 = tokenize(a)
    t2 = tokenize(b)
    union = len(t1 | t2)
    return len(t1 & t2) / union if union else 1.0


def tree_edit_similarity(a, b):
    p1 = set(flatten_json(a))
    p2 = set(flatten_json(b))
    max_nodes = max(len(p1), len(p2))
    return len(p1 & p2) / max_nodes if max_nodes else 1.0


def semantic_similarity(a, b):
    keys = ["events", "formula", "signature", "violation", "domain", "formalism"]
    scores = []

    for k in keys:
        s1 = json.dumps(a.get(k, ""), sort_keys=True)
        s2 = json.dumps(b.get(k, ""), sort_keys=True)
        scores.append(difflib.SequenceMatcher(None, s1, s2).ratio())

    return sum(scores) / len(scores)


# ---------------- FIND FILE ----------------

def find_file(base, filename):
    for f in base.rglob(filename):
        return f
    return None


# ---------------- COMPARE ----------------

def compare_files(f1, f2, f3):
    texts = {}
    jsons = {}

    for label, path in zip(["original", "random", "structural"], [f1, f2, f3]):
        txt = Path(path).read_text(encoding="utf-8")
        texts[label] = txt
        jsons[label] = json.loads(txt)

    rows = []
    names = list(texts.keys())

    for a, b in itertools.combinations(names, 2):
        lev = lev_sim(texts[a], texts[b])
        diff = diff_sim(texts[a], texts[b])
        struct = structural_sim(jsons[a], jsons[b])
        jacc = jaccard_sim(texts[a], texts[b])
        tree = tree_edit_similarity(jsons[a], jsons[b])
        sem = semantic_similarity(jsons[a], jsons[b])

        final = (lev + diff + struct + jacc + tree + sem) / 6

        rows.append({
            "file_a": a,
            "file_b": b,
            "lev": round(lev * 100, 2),
            "diff": round(diff * 100, 2),
            "struct": round(struct * 100, 2),
            "jaccard": round(jacc * 100, 2),
            "tree": round(tree * 100, 2),
            "semantic": round(sem * 100, 2),
            "final": round(final * 100, 2),
        })

    df = pd.DataFrame(rows)

    matrix = pd.DataFrame(index=names, columns=names)

    for a in names:
        for b in names:
            if a == b:
                matrix.loc[a, b] = 100.0
            else:
                r = df[
                    ((df.file_a == a) & (df.file_b == b)) |
                    ((df.file_a == b) & (df.file_b == a))
                ]
                matrix.loc[a, b] = round(float(r["final"].iloc[0]), 2)

    return df, matrix


# ---------------- SCENARIO WINNERS ----------------

def compute_scenario_winners(global_df):
    scenarios = []
    specs = global_df["spec"].unique()

    for spec in specs:
        subset = global_df[global_df["spec"] == spec]
        formalism = subset["formalism"].iloc[0]

        random_row = subset[
            ((subset.file_a == "original") & (subset.file_b == "random")) |
            ((subset.file_a == "random") & (subset.file_b == "original"))
        ]

        structural_row = subset[
            ((subset.file_a == "original") & (subset.file_b == "structural")) |
            ((subset.file_a == "structural") & (subset.file_b == "original"))
        ]

        if random_row.empty or structural_row.empty:
            continue

        random_score = float(random_row["final"].iloc[0])
        structural_score = float(structural_row["final"].iloc[0])

        diff = round(abs(structural_score - random_score), 2)

        if structural_score > random_score:
            winner = "structural"
        elif random_score > structural_score:
            winner = "random"
        else:
            winner = "tie"

        scenarios.append({
            "scenario": spec,
            "formalism": formalism,
            "winner": winner,
            "difference_%": diff,
            "random_score": random_score,
            "structural_score": structural_score,
        })

    return pd.DataFrame(scenarios)


# ---------------- OVERALL ANALYSIS ----------------

def compute_overall_stats(scenario_df):
    structural_wins = (scenario_df["winner"] == "structural").sum()
    random_wins = (scenario_df["winner"] == "random").sum()
    ties = (scenario_df["winner"] == "tie").sum()

    non_ties = scenario_df[scenario_df["winner"] != "tie"]
    avg_advantage = round(non_ties["difference_%"].mean(), 2) if not non_ties.empty else 0.0

    scoreboard = pd.DataFrame([{
        "structural_wins": structural_wins,
        "random_wins": random_wins,
        "ties": ties,
        "average_advantage_%": avg_advantage
    }])

    scoreboard.to_csv(RESULT_PATH / "overall_scoreboard.csv", index=False)
    return scoreboard


# ---------------- METRIC ANALYSIS ----------------

def metric_analysis(global_df):
    metric_cols = ["lev", "diff", "struct", "jaccard", "tree", "semantic"]

    corr = global_df[metric_cols + ["final"]].corr()
    influence = corr["final"].drop("final").sort_values(ascending=False)

    influence_df = influence.reset_index()
    influence_df.columns = ["metric", "correlation_with_final"]
    influence_df.to_csv(RESULT_PATH / "metric_influence.csv", index=False)

    metric_corr = global_df[metric_cols].corr()
    metric_corr.to_csv(RESULT_PATH / "metric_correlation_matrix.csv")

    plt.figure()
    plt.imshow(metric_corr)
    plt.colorbar()
    plt.xticks(range(len(metric_cols)), metric_cols)
    plt.yticks(range(len(metric_cols)), metric_cols)
    plt.title("Metric Correlation Heatmap")
    plt.savefig(RESULT_PATH / "metric_correlation_heatmap.png", bbox_inches="tight")
    plt.close()


# ---------------- SCENARIO DIFFICULTY ----------------

def scenario_difficulty(scenario_df):
    scenario_df = scenario_df.copy()
    scenario_df["difficulty"] = 100 - scenario_df[["random_score", "structural_score"]].max(axis=1)

    hardest = scenario_df.sort_values("difficulty", ascending=False)
    hardest.to_csv(RESULT_PATH / "hardest_scenarios.csv", index=False)


# ---------------- API DIFFICULTY ----------------

def classify_api(name):
    name = name.lower()

    if "socket" in name or "net" in name or "http" in name or "url" in name or "datagram" in name or "inet" in name:
        return "NET"
    if "file" in name or "stream" in name or "reader" in name or "writer" in name or "console" in name:
        return "IO"
    if "collection" in name or "collections" in name or "list" in name or "map" in name or "set" in name or "queue" in name or "deque" in name or "iterator" in name or "vector" in name or "arraydeque" in name or "treemap" in name or "treeset" in name:
        return "UTIL"
    return "LANG"


def api_difficulty(scenario_df):
    scenario_df = scenario_df.copy()
    scenario_df["api"] = scenario_df["scenario"].apply(classify_api)
    scenario_df["difficulty"] = 100 - scenario_df[["random_score", "structural_score"]].max(axis=1)

    difficulty = scenario_df.groupby("api")[["random_score", "structural_score", "difficulty"]].mean().round(2)
    difficulty.to_csv(RESULT_PATH / "api_difficulty.csv")

    plt.figure()
    ordered_apis = list(difficulty.index)
    data = [scenario_df[scenario_df["api"] == a]["difficulty"] for a in ordered_apis]
    plt.boxplot(data, tick_labels=ordered_apis)
    plt.ylabel("Difficulty")
    plt.title("API Difficulty for LLM")
    plt.savefig(RESULT_PATH / "api_difficulty_boxplot.png", bbox_inches="tight")
    plt.close()

    return scenario_df


# ---------------- FORMALISM ANALYSIS ----------------

def formalism_analysis(scenario_df):
    summary = scenario_df.groupby(["formalism", "winner"]).size().unstack(fill_value=0)
    summary.to_csv(RESULT_PATH / "formalism_wins.csv")

    winrate = summary.div(summary.sum(axis=1), axis=0) * 100
    winrate = winrate.round(2)
    winrate.to_csv(RESULT_PATH / "formalism_winrate.csv")

    advantage = scenario_df.groupby("formalism")["difference_%"].mean().round(2)
    advantage.to_csv(RESULT_PATH / "formalism_average_advantage.csv")

    ax = summary.plot(kind="bar")
    ax.set_title("Wins per Formalism")
    ax.set_ylabel("Number of Scenarios")
    plt.tight_layout()
    plt.savefig(RESULT_PATH / "formalism_wins_plot.png")
    plt.close()


# ---------------- FORMALISM DIFFICULTY ----------------

def formalism_difficulty(scenario_df):
    scenario_df = scenario_df.copy()
    scenario_df["difficulty"] = 100 - scenario_df[["random_score", "structural_score"]].max(axis=1)

    formalisms = list(scenario_df["formalism"].dropna().unique())
    data = [scenario_df[scenario_df["formalism"] == f]["difficulty"] for f in formalisms]

    plt.figure()
    plt.boxplot(data, tick_labels=formalisms)
    plt.ylabel("Difficulty")
    plt.title("Difficulty per Formalism")
    plt.savefig(RESULT_PATH / "formalism_difficulty_boxplot.png", bbox_inches="tight")
    plt.close()


# ---------------- API × FORMALISM HEATMAP ----------------

def api_formalism_heatmap(scenario_df):
    scenario_df = scenario_df.copy()

    if "api" not in scenario_df.columns:
        scenario_df["api"] = scenario_df["scenario"].apply(classify_api)

    pivot = scenario_df.pivot_table(
        index="api",
        columns="formalism",
        values="difference_%",
        aggfunc="mean"
    ).round(2)

    pivot.to_csv(RESULT_PATH / "api_formalism_matrix.csv")

    plt.figure()
    plt.imshow(pivot)
    plt.colorbar()
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.title("API × Formalism Difficulty")
    plt.tight_layout()
    plt.savefig(RESULT_PATH / "api_formalism_heatmap.png")
    plt.close()


# ---------------- STATISTICAL ANALYSIS ----------------

def statistical_analysis(scenario_df):
    random_scores = scenario_df["random_score"].values
    structural_scores = scenario_df["structural_score"].values
    diff = structural_scores - random_scores

    shapiro_stat, shapiro_p = shapiro(diff)

    if shapiro_p > 0.05:
        test = "paired_t_test"
        stat, p = ttest_rel(structural_scores, random_scores)
    else:
        test = "wilcoxon"
        stat, p = wilcoxon(structural_scores, random_scores)

    pd.DataFrame([{
        "test_used": test,
        "statistic": round(float(stat), 6),
        "p_value": round(float(p), 6),
        "shapiro_stat": round(float(shapiro_stat), 6),
        "shapiro_p": round(float(shapiro_p), 6),
    }]).to_csv(RESULT_PATH / "statistical_test.csv", index=False)

    plt.figure()
    plt.boxplot([random_scores, structural_scores], tick_labels=["Random", "Structural"])
    plt.ylabel("Similarity (%)")
    plt.title("Similarity Distribution")
    plt.savefig(RESULT_PATH / "boxplot_similarity.png", bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.hist(diff, bins=15)
    plt.xlabel("Structural - Random (%)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Differences")
    plt.savefig(RESULT_PATH / "difference_histogram.png", bbox_inches="tight")
    plt.close()

    plt.figure()

    better_struct = diff > 0
    better_random = diff < 0
    ties = diff == 0

    plt.scatter(
        random_scores[better_struct],
        structural_scores[better_struct],
        marker="o",
        label="Structural better"
    )

    plt.scatter(
        random_scores[better_random],
        structural_scores[better_random],
        marker="x",
        label="Random better"
    )

    if np.any(ties):
        plt.scatter(
            random_scores[ties],
            structural_scores[ties],
            marker="s",
            label="Tie"
        )

    max_val = max(max(random_scores), max(structural_scores))
    plt.plot([0, max_val], [0, max_val], label="Equal performance")

    plt.xlabel("Random Score")
    plt.ylabel("Structural Score")
    plt.title("Random vs Structural")
    plt.legend()
    plt.savefig(RESULT_PATH / "random_vs_structural.png", bbox_inches="tight")
    plt.close()


# ---------------- MAIN ----------------

def main():
    prepare_results()

    n = int(input("How many specs do you want to compare? "))

    files = sorted(ORIGINAL_PATH.rglob("*.json"))[:n]

    global_metrics = []
    processed = 0

    for file in files:
        name = file.name
        print("Processing", name)

        rand = find_file(RANDOM_PATH, name)
        struct = find_file(STRUCTURAL_PATH, name)

        if not rand or not struct:
            print("Missing:", name)
            continue

        with open(file, "r", encoding="utf-8") as f:
            original_json = json.load(f)

        formalism = original_json.get("formalism", "unknown")

        processed += 1

        df, matrix = compare_files(file, rand, struct)
        df["spec"] = name
        df["formalism"] = formalism

        global_metrics.append(df)

        df.to_csv(INDIVIDUAL_METRICS_PATH / f"{name}_metrics.csv", index=True)
        matrix.to_csv(INDIVIDUAL_MATRICES_PATH / f"{name}_matrix.csv")

    if not global_metrics:
        print("No matching scenarios were processed.")
        return

    global_df = pd.concat(global_metrics, ignore_index=True)
    global_df.to_csv(RESULT_PATH / "global_metrics.csv", index=True)

    global_matrix = global_df.pivot_table(
        index="file_a",
        columns="file_b",
        values="final",
        aggfunc="mean"
    ).round(2)
    global_matrix.to_csv(RESULT_PATH / "global_similarity_matrix.csv")

    scenario_df = compute_scenario_winners(global_df)
    scenario_df.to_csv(RESULT_PATH / "scenario_winners.csv", index=True)

    compute_overall_stats(scenario_df)
    metric_analysis(global_df)
    scenario_difficulty(scenario_df)
    formalism_analysis(scenario_df)
    formalism_difficulty(scenario_df)

    scenario_with_api = api_difficulty(scenario_df)
    api_formalism_heatmap(scenario_with_api)

    statistical_analysis(scenario_df)

    print("\nTotal scenarios analysed:", len(scenario_df))
    print("Requested scenarios:", n)
    print("Files processed:", processed)
    print("\nAnalysis complete.")
    print("Files generated:")
    print(" - global_metrics.csv")
    print(" - global_similarity_matrix.csv")
    print(" - scenario_winners.csv")
    print(" - overall_scoreboard.csv")
    print(" - statistical_test.csv")
    print(" - metric_influence.csv")
    print(" - metric_correlation_matrix.csv")
    print(" - hardest_scenarios.csv")
    print(" - api_difficulty.csv")
    print(" - formalism_wins.csv")
    print(" - formalism_winrate.csv")
    print(" - formalism_average_advantage.csv")
    print(" - individual/metrics/")
    print(" - individual/matrices/")


if __name__ == "__main__":
    main()
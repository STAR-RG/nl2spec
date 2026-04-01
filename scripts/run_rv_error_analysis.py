from __future__ import annotations

from pathlib import Path
import csv
import json
import itertools
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Tuple, Optional


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
    r"C:\UFPE\Siesta\project_LLM_spec\nl2spec\output\llm\results_semantic_rv"
)


# ==========================================================
# VALIDATION
# ==========================================================

def validate_paths() -> None:
    for label, path in {
        "ORIGINAL_PATH": ORIGINAL_PATH,
        "RANDOM_PATH": RANDOM_PATH,
        "STRUCTURAL_PATH": STRUCTURAL_PATH,
    }.items():
        if not path.exists():
            raise FileNotFoundError("{0} not found: {1}".format(label, path))


# ==========================================================
# IO HELPERS
# ==========================================================

def load_json_files(root: Path) -> Dict[str, Dict[str, Any]]:
    files = {}

    for p in root.rglob("*.json"):
        try:
            files[p.name] = json.loads(p.read_text(encoding="utf-8"))
        except Exception as exc:
            raise RuntimeError("Failed reading {0}: {1}".format(p, exc))

    return files


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


# ==========================================================
# NORMALIZATION
# ==========================================================

def normalize_space(text: str) -> str:
    return " ".join((text or "").split())


def normalize_action(action: str) -> str:
    value = (action or "").strip().lower()

    if value in {"create", "creation", "creation event"}:
        return "creation event"

    if value == "":
        return ""

    return "event"


def normalize_log_message(message: str) -> str:
    value = (message or "").strip()
    if len(value) >= 2 and value[0] == value[-1] == '"':
        return value[1:-1]
    return value


def normalize_statement(stmt: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(stmt)
    if out.get("type") == "log":
        out["message"] = normalize_log_message(out.get("message", ""))
    return out


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


# ==========================================================
# MATCH RATE
# ==========================================================

def _coerce_match_rate(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return 1 if value else 0

    if isinstance(value, (int, float)):
        return 1 if float(value) >= 1.0 else 0

    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "1.0", "true", "yes", "match", "equal"}:
            return 1
        if v in {"0", "0.0", "false", "no", "different", "diff"}:
            return 0

    return None


def extract_match_rate(candidate_spec: Dict[str, Any], baseline_spec: Dict[str, Any]) -> int:
    """
    Primary correctness signal.

    It first tries to read a stored field from the candidate JSON.
    If absent, it falls back to exact canonical equality between baseline and candidate.
    """
    candidate_paths = [
        ("extract_match_rate",),
        ("match_rate",),
        ("metrics", "extract_match_rate"),
        ("metrics", "match_rate"),
        ("comparison", "extract_match_rate"),
        ("comparison", "match_rate"),
        ("analysis", "extract_match_rate"),
        ("analysis", "match_rate"),
        ("evaluation", "extract_match_rate"),
        ("evaluation", "match_rate"),
    ]

    for path in candidate_paths:
        cur = candidate_spec
        ok = True

        for key in path:
            if isinstance(cur, dict) and key in cur:
                cur = cur[key]
            else:
                ok = False
                break

        if ok:
            coerced = _coerce_match_rate(cur)
            if coerced is not None:
                return coerced

    return 1 if canonical_json(candidate_spec) == canonical_json(baseline_spec) else 0


# ==========================================================
# IR ACCESS
# ==========================================================

def get_methods(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    ir = spec.get("ir", {})
    events = ir.get("events", [])

    if not isinstance(events, list) or not events:
        return []

    methods = []
    for item in events:
        if not isinstance(item, dict):
            continue
        body = item.get("body", {})
        if not isinstance(body, dict):
            continue
        inner = body.get("methods", [])
        if isinstance(inner, list):
            methods.extend([m for m in inner if isinstance(m, dict)])

    return methods


def method_map(spec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    result = {}

    for m in get_methods(spec):
        name = m.get("name")
        if isinstance(name, str) and name:
            result[name] = m

    return result


def method_params(method: Dict[str, Any]) -> Tuple[Tuple[str, str], ...]:
    params = []

    for p in method.get("parameters", []) or []:
        if isinstance(p, dict):
            params.append((str(p.get("type", "")), str(p.get("name", ""))))

    return tuple(params)


def method_returning(method: Dict[str, Any]) -> Optional[Tuple[Tuple[str, str], ...]]:
    ret = method.get("returning")
    if isinstance(ret, dict):
        return tuple(sorted((str(k), str(v)) for k, v in ret.items()))
    return None


def pointcut_atoms(method: Dict[str, Any]) -> List[Tuple[str, Tuple[str, ...], bool]]:
    atoms = []

    for fn in method.get("function", []) or []:
        if not isinstance(fn, dict):
            continue

        name = str(fn.get("name", ""))
        args = []

        for arg in fn.get("arguments", []) or []:
            if isinstance(arg, dict):
                args.append(str(arg.get("value", "")))

        negated = bool(fn.get("negated", False))
        atoms.append((name, tuple(args), negated))

    return atoms


def pointcut_ops(method: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(str(op) for op in (method.get("operation", []) or []))


def violation_tag(spec: Dict[str, Any]) -> str:
    return str(spec.get("ir", {}).get("violation", {}).get("tag", "")).strip().lower()


def violation_has_reset(spec: Dict[str, Any]) -> bool:
    return bool(
        spec.get("ir", {})
        .get("violation", {})
        .get("body", {})
        .get("has_reset", False)
    )


def violation_statements(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    stmts = (
        spec.get("ir", {})
        .get("violation", {})
        .get("body", {})
        .get("statements", [])
    ) or []

    out = []
    for stmt in stmts:
        if isinstance(stmt, dict):
            out.append(normalize_statement(stmt))

    return out


def ere_expression(spec: Dict[str, Any]) -> str:
    return normalize_space(spec.get("ir", {}).get("ere", {}).get("expression", ""))


# ==========================================================
# ERE ANALYSIS
# ==========================================================

def tokenize_ere(expr: str) -> List[Tuple[str, str]]:
    token_spec = [
        ("LPAREN", r"\("),
        ("RPAREN", r"\)"),
        ("OR", r"\|"),
        ("STAR", r"\*"),
        ("PLUS", r"\+"),
        ("QMARK", r"\?"),
        ("NOT", r"[!~]"),
        ("EPSILON", r"\bepsilon\b"),
        ("EMPTY", r"\bempty\b"),
        ("IDENT", r"[A-Za-z_]\w*"),
        ("WS", r"\s+"),
    ]

    regex = "|".join("(?P<{0}>{1})".format(name, pattern) for name, pattern in token_spec)
    pos = 0
    tokens = []

    while pos < len(expr):
        m = re.match(regex, expr[pos:])
        if not m:
            break

        kind = m.lastgroup
        value = m.group(kind)

        if kind != "WS":
            tokens.append((kind, value))

        pos += len(m.group(0))

    return tokens


def ere_operator_sequence(expr: str) -> List[str]:
    return [kind for kind, _ in tokenize_ere(expr) if kind in {"OR", "STAR", "PLUS", "QMARK", "NOT"}]


def ere_identifier_sequence(expr: str) -> List[str]:
    return [value for kind, value in tokenize_ere(expr) if kind == "IDENT"]


def expression_referenced_events(expr: str) -> List[str]:
    reserved = {"epsilon", "empty"}
    return [v for k, v in tokenize_ere(expr) if k == "IDENT" and v not in reserved]


# ==========================================================
# CATEGORIZATION
# ==========================================================

def compare_specs_semantic(
    baseline_spec: Dict[str, Any],
    candidate_spec: Dict[str, Any],
) -> List[str]:
    categories = []

    # ---------------- ERE FORMULA ----------------
    b_expr = ere_expression(baseline_spec)
    c_expr = ere_expression(candidate_spec)

    if b_expr != c_expr:
        categories.append("ere_expression_error")

        if ere_operator_sequence(b_expr) != ere_operator_sequence(c_expr):
            categories.append("ere_operator_error")

        if ere_identifier_sequence(b_expr) != ere_identifier_sequence(c_expr):
            categories.append("ere_order_error")

    candidate_declared_events = set(method_map(candidate_spec).keys())
    candidate_referenced_events = set(expression_referenced_events(c_expr))
    if not candidate_referenced_events.issubset(candidate_declared_events):
        categories.append("ere_event_reference_error")

    # ---------------- EVENT SET ----------------
    baseline_methods = method_map(baseline_spec)
    candidate_methods = method_map(candidate_spec)

    baseline_names = set(baseline_methods.keys())
    candidate_names = set(candidate_methods.keys())

    if baseline_names - candidate_names:
        categories.append("missing_event")

    if candidate_names - baseline_names:
        categories.append("extra_event")

    # ---------------- EVENT SIGNATURE / POINTCUT ----------------
    for name in sorted(baseline_names & candidate_names):
        b = baseline_methods[name]
        c = candidate_methods[name]

        if name != c.get("name", name):
            categories.append("event_name_error")

        if normalize_action(b.get("action", "")) != normalize_action(c.get("action", "")):
            categories.append("action_label_error")

        if str(b.get("timing", "")).strip().lower() != str(c.get("timing", "")).strip().lower():
            categories.append("timing_error")

        if method_params(b) != method_params(c):
            categories.append("method_parameter_error")

        if method_returning(b) != method_returning(c):
            categories.append("returning_error")

        baseline_atoms = pointcut_atoms(b)
        candidate_atoms = pointcut_atoms(c)

        if baseline_atoms != candidate_atoms:
            baseline_atom_set = set(baseline_atoms)
            candidate_atom_set = set(candidate_atoms)

            if baseline_atom_set - candidate_atom_set:
                categories.append("pointcut_atom_missing")

            if candidate_atom_set - baseline_atom_set:
                categories.append("pointcut_atom_extra")

            for ba, ca in itertools.zip_longest(baseline_atoms, candidate_atoms, fillvalue=None):
                if ba is None or ca is None:
                    categories.append("pointcut_order_error")
                    continue

                if ba[0] != ca[0]:
                    categories.append("pointcut_order_error")

                    # specialized call kind / target checks
                    if ba[0] in {"call", "execution"} and ca[0] in {"call", "execution"}:
                        categories.append("pointcut_call_kind_error")

                    if "target" in {ba[0], ca[0]}:
                        categories.append("pointcut_target_error")

                else:
                    if ba[1] != ca[1]:
                        categories.append("pointcut_argument_error")
                    if ba[2] != ca[2]:
                        categories.append("pointcut_negation_error")

                    if ba[0] == "target":
                        categories.append("pointcut_target_error")

        if pointcut_ops(b) != pointcut_ops(c):
            categories.append("pointcut_operator_error")

    # ---------------- VIOLATION ----------------
    if violation_tag(baseline_spec) != violation_tag(candidate_spec):
        categories.append("violation_tag_error")

    if violation_has_reset(baseline_spec) != violation_has_reset(candidate_spec):
        categories.append("violation_reset_error")

    b_statements = violation_statements(baseline_spec)
    c_statements = violation_statements(candidate_spec)

    if b_statements != c_statements:
        categories.append("violation_statements_error")

        b_logs = [s.get("message", "") for s in b_statements if s.get("type") == "log"]
        c_logs = [s.get("message", "") for s in c_statements if s.get("type") == "log"]

        if b_logs != c_logs:
            categories.append("violation_log_message_error")

    # ---------------- DEDUP ----------------
    seen = set()
    deduped = []

    for cat in categories:
        if cat not in seen:
            deduped.append(cat)
            seen.add(cat)

    return deduped


# ==========================================================
# SCORING / SEVERITY / WINNER
# ==========================================================

CATEGORY_WEIGHTS = {
    "match": 0,
    "ere_expression_error": 6,
    "ere_operator_error": 5,
    "ere_order_error": 5,
    "ere_event_reference_error": 5,
    "missing_event": 5,
    "extra_event": 4,
    "event_name_error": 3,
    "action_label_error": 2,
    "timing_error": 3,
    "method_parameter_error": 3,
    "returning_error": 3,
    "pointcut_atom_missing": 4,
    "pointcut_atom_extra": 3,
    "pointcut_order_error": 4,
    "pointcut_argument_error": 3,
    "pointcut_negation_error": 4,
    "pointcut_operator_error": 4,
    "pointcut_target_error": 3,
    "pointcut_call_kind_error": 4,
    "violation_tag_error": 5,
    "violation_reset_error": 4,
    "violation_statements_error": 2,
    "violation_log_message_error": 2,
    "near_miss_minor": 0,
    "near_miss_moderate": 0,
    "near_miss_major": 0,
}


def score_categories(categories: List[str]) -> int:
    return sum(CATEGORY_WEIGHTS.get(cat, 1) for cat in categories)


def add_near_miss_severity(categories: List[str]) -> List[str]:
    score = score_categories(categories)
    out = list(categories)

    if score <= 4:
        out.append("near_miss_minor")
    elif score <= 9:
        out.append("near_miss_moderate")
    else:
        out.append("near_miss_major")

    return out


def decide_winner(
    random_match_rate: int,
    structural_match_rate: int,
    random_categories: List[str],
    structural_categories: List[str],
) -> str:
    if random_match_rate == 1 and structural_match_rate == 1:
        return "tie_correct"

    if random_match_rate == 1 and structural_match_rate == 0:
        return "random"

    if structural_match_rate == 1 and random_match_rate == 0:
        return "structural"

    random_score = score_categories(random_categories)
    structural_score = score_categories(structural_categories)

    if random_score < structural_score:
        return "random"

    if structural_score < random_score:
        return "structural"

    return "tie_wrong"


# ==========================================================
# MAIN ANALYSIS
# ==========================================================

def analyze(
    baseline_root: Path,
    random_root: Path,
    structural_root: Path,
) -> Tuple[List[Dict[str, Any]], Counter, Dict[str, Counter]]:
    baseline_files = load_json_files(baseline_root)
    random_files = load_json_files(random_root)
    structural_files = load_json_files(structural_root)

    scenario_names = sorted(set(random_files.keys()) & set(structural_files.keys()))

    rows = []
    winner_counter = Counter()
    category_summary = {
        "random": Counter(),
        "structural": Counter(),
    }

    for scenario in scenario_names:
        if scenario not in baseline_files:
            continue

        baseline_spec = baseline_files[scenario]
        random_spec = random_files[scenario]
        structural_spec = structural_files[scenario]

        random_match_rate = extract_match_rate(random_spec, baseline_spec)
        structural_match_rate = extract_match_rate(structural_spec, baseline_spec)

        if random_match_rate == 1:
            random_categories = ["match"]
        else:
            random_categories = add_near_miss_severity(
                compare_specs_semantic(baseline_spec, random_spec)
            )

        if structural_match_rate == 1:
            structural_categories = ["match"]
        else:
            structural_categories = add_near_miss_severity(
                compare_specs_semantic(baseline_spec, structural_spec)
            )

        winner = decide_winner(
            random_match_rate=random_match_rate,
            structural_match_rate=structural_match_rate,
            random_categories=random_categories,
            structural_categories=structural_categories,
        )
        winner_counter[winner] += 1

        if random_match_rate == 0:
            category_summary["random"].update(random_categories)

        if structural_match_rate == 0:
            category_summary["structural"].update(structural_categories)

        rows.append(
            {
                "scenario": scenario,
                "random_extract_match_rate": random_match_rate,
                "structural_extract_match_rate": structural_match_rate,
                "random_categories": ";".join(random_categories),
                "random_score": score_categories(random_categories),
                "structural_categories": ";".join(structural_categories),
                "structural_score": score_categories(structural_categories),
                "winner": winner,
                "baseline_ere": ere_expression(baseline_spec),
                "random_ere": ere_expression(random_spec),
                "structural_ere": ere_expression(structural_spec),
                "baseline_events": ";".join(sorted(method_map(baseline_spec).keys())),
                "random_events": ";".join(sorted(method_map(random_spec).keys())),
                "structural_events": ";".join(sorted(method_map(structural_spec).keys())),
            }
        )

    return rows, winner_counter, category_summary


# ==========================================================
# EXPORT
# ==========================================================

def export_results(
    out_dir: Path,
    rows: List[Dict[str, Any]],
    winner_counter: Counter,
    category_summary: Dict[str, Counter],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        out_dir / "winner_by_case.csv",
        [
            "scenario",
            "random_extract_match_rate",
            "structural_extract_match_rate",
            "random_categories",
            "random_score",
            "structural_categories",
            "structural_score",
            "winner",
            "baseline_ere",
            "random_ere",
            "structural_ere",
            "baseline_events",
            "random_events",
            "structural_events",
        ],
        rows,
    )

    summary_rows = []
    for strategy in ["random", "structural"]:
        for category, count in sorted(category_summary[strategy].items()):
            summary_rows.append(
                {
                    "strategy": strategy,
                    "category": category,
                    "count": count,
                }
            )

    write_csv(
        out_dir / "category_summary.csv",
        ["strategy", "category", "count"],
        summary_rows,
    )

    winner_rows = [
        {"winner": winner, "count": count}
        for winner, count in sorted(winner_counter.items())
    ]
    write_csv(
        out_dir / "winner_summary.csv",
        ["winner", "count"],
        winner_rows,
    )

    match_rate_rows = []
    for row in rows:
        match_rate_rows.append(
            {
                "scenario": row["scenario"],
                "random_extract_match_rate": row["random_extract_match_rate"],
                "structural_extract_match_rate": row["structural_extract_match_rate"],
                "winner": row["winner"],
            }
        )

    write_csv(
        out_dir / "match_rate_summary.csv",
        [
            "scenario",
            "random_extract_match_rate",
            "structural_extract_match_rate",
            "winner",
        ],
        match_rate_rows,
    )

    flat_rows = []
    for row in rows:
        random_categories = row["random_categories"].split(";") if row["random_categories"] else []
        structural_categories = row["structural_categories"].split(";") if row["structural_categories"] else []

        if not random_categories:
            random_categories = [""]

        if not structural_categories:
            structural_categories = [""]

        max_len = max(len(random_categories), len(structural_categories))

        while len(random_categories) < max_len:
            random_categories.append("")

        while len(structural_categories) < max_len:
            structural_categories.append("")

        for idx in range(max_len):
            flat_rows.append(
                {
                    "scenario": row["scenario"],
                    "winner": row["winner"],
                    "random_extract_match_rate": row["random_extract_match_rate"],
                    "structural_extract_match_rate": row["structural_extract_match_rate"],
                    "random_score": row["random_score"],
                    "structural_score": row["structural_score"],
                    "random_category": random_categories[idx],
                    "structural_category": structural_categories[idx],
                }
            )

    write_csv(
        out_dir / "per_case_analysis.csv",
        [
            "scenario",
            "winner",
            "random_extract_match_rate",
            "structural_extract_match_rate",
            "random_score",
            "structural_score",
            "random_category",
            "structural_category",
        ],
        flat_rows,
    )

    random_failure_rows = []
    structural_failure_rows = []

    for row in rows:
        if row["random_extract_match_rate"] == 0:
            random_failure_rows.append(
                {
                    "scenario": row["scenario"],
                    "extract_match_rate": row["random_extract_match_rate"],
                    "score": row["random_score"],
                    "categories": row["random_categories"],
                    "baseline_ere": row["baseline_ere"],
                    "candidate_ere": row["random_ere"],
                    "baseline_events": row["baseline_events"],
                    "candidate_events": row["random_events"],
                }
            )

        if row["structural_extract_match_rate"] == 0:
            structural_failure_rows.append(
                {
                    "scenario": row["scenario"],
                    "extract_match_rate": row["structural_extract_match_rate"],
                    "score": row["structural_score"],
                    "categories": row["structural_categories"],
                    "baseline_ere": row["baseline_ere"],
                    "candidate_ere": row["structural_ere"],
                    "baseline_events": row["baseline_events"],
                    "candidate_events": row["structural_events"],
                }
            )

    write_csv(
        out_dir / "random_failures.csv",
        [
            "scenario",
            "extract_match_rate",
            "score",
            "categories",
            "baseline_ere",
            "candidate_ere",
            "baseline_events",
            "candidate_events",
        ],
        random_failure_rows,
    )

    write_csv(
        out_dir / "structural_failures.csv",
        [
            "scenario",
            "extract_match_rate",
            "score",
            "categories",
            "baseline_ere",
            "candidate_ere",
            "baseline_events",
            "candidate_events",
        ],
        structural_failure_rows,
    )


# ==========================================================
# ENTRYPOINT
# ==========================================================

def main() -> None:
    validate_paths()
    RESULT_PATH.mkdir(parents=True, exist_ok=True)

    rows, winner_counter, category_summary = analyze(
        baseline_root=ORIGINAL_PATH,
        random_root=RANDOM_PATH,
        structural_root=STRUCTURAL_PATH,
    )

    export_results(
        out_dir=RESULT_PATH,
        rows=rows,
        winner_counter=winner_counter,
        category_summary=category_summary,
    )

    print("=" * 72)
    print("[OK] RV error analysis finished")
    print("Cases analyzed:", len(rows))
    print("Winner summary:", dict(winner_counter))
    print("Output:", RESULT_PATH)
    print("=" * 72)


if __name__ == "__main__":
    main()
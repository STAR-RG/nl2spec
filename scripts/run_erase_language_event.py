from pathlib import Path
import re
import shutil
import csv

#run python -m nl2spec.scripts.run_erase_language_event

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parent.parent

INPUT_ROOT = PROJECT_ROOT / "datasets" / "raw_mop"
OUTPUT_ROOT = PROJECT_ROOT / "datasets" / "dataset_mop"

DOMAINS = {"io", "net", "lang", "util"}

# =========================================================
# FORMALISM DETECTION
# event = default if not ere/ltl/fsm
# =========================================================

ERE_RE = re.compile(r'^\s*ere\b', re.IGNORECASE | re.MULTILINE)
LTL_RE = re.compile(r'^\s*ltl\b', re.IGNORECASE | re.MULTILINE)
FSM_RE = re.compile(r'^\s*fsm\b', re.IGNORECASE | re.MULTILINE)

# =========================================================
# BASIC PATTERNS
# =========================================================

EVENT_START_RE = re.compile(r'^\s*event\b')
LOG_LINE_RE = re.compile(r'^\s*RVMLogging\.out\.println\s*\(.*\)\s*;\s*$')

# method signature without {
METHOD_SIGNATURE_RE = re.compile(
    r'^\s*'
    r'(?:(?:public|private|protected)\s+)?'
    r'(?:(?:static|final|synchronized|native|abstract)\s+)*'
    r'([\w\<\>\[\],\.\?]+)\s+'
    r'(\w+)\s*'
    r'\([^;{}]*\)\s*$'
)

# method signature with { on same line
METHOD_INLINE_RE = re.compile(
    r'^\s*'
    r'(?:(?:public|private|protected)\s+)?'
    r'(?:(?:static|final|synchronized|native|abstract)\s+)*'
    r'([\w\<\>\[\],\.\?]+)\s+'
    r'(\w+)\s*'
    r'\([^;{}]*\)\s*'
    r'\{\s*$'
)

# method call in one line:
#   foo(...);
#   this.foo(...);
#   obj.foo(...);
METHOD_CALL_RE = re.compile(r'^\s*(?:[\w]+\.)?(\w+)\s*\([^;]*\)\s*;\s*$')


# =========================================================
# DOMAIN DETECTION
# =========================================================

def detect_domain_from_path(input_file):
    rel_parts = input_file.relative_to(INPUT_ROOT).parts
    for part in rel_parts:
        if part in DOMAINS:
            return part
    return None


def infer_domain_from_content(text, filename):
    text_lower = text.lower()
    filename_lower = filename.lower()

    io_markers = [
        "java.io", "inputstream", "outputstream", "reader", "writer",
        "file", "closeable", "serializable", "objectinput", "objectoutput"
    ]
    net_markers = [
        "java.net", "socket", "serversocket", "url", "uri",
        "httpurlconnection", "datagramsocket", "inetaddress",
        "datagrampacket"
    ]
    lang_markers = [
        "java.lang", "string", "system", "thread", "runtime",
        "classloader", "character", "math", "boolean", "integer",
        "long", "double"
    ]
    util_markers = [
        "java.util", "collection", "collections", "map", "set", "list",
        "queue", "deque", "arraydeque", "arrays", "iterator", "hashmap",
        "hashset", "comparator", "treemap"
    ]

    def score(markers):
        return sum(1 for marker in markers if marker in text_lower or marker in filename_lower)

    scores = {
        "io": score(io_markers),
        "net": score(net_markers),
        "lang": score(lang_markers),
        "util": score(util_markers),
    }

    best_domain = max(scores, key=scores.get)
    if scores[best_domain] == 0:
        raise ValueError("Could not infer domain for file: {}".format(filename))

    return best_domain


def detect_domain(input_file, text):
    domain = detect_domain_from_path(input_file)
    if domain:
        return domain
    return infer_domain_from_content(text, input_file.name)


# =========================================================
# FORMALISM DETECTION
# =========================================================

def detect_formalism(text):
    if ERE_RE.search(text):
        return "ere"
    if LTL_RE.search(text):
        return "ltl"
    if FSM_RE.search(text):
        return "fsm"
    return "event"


# =========================================================
# GENERIC BLOCK EXTRACTION
# =========================================================

def extract_balanced_block(lines, start_idx):
    block_lines = []
    balance = 0
    opened = False
    i = start_idx

    while i < len(lines):
        line = lines[i]
        block_lines.append(line)

        if "{" in line:
            balance += line.count("{")
            opened = True
        if "}" in line:
            balance -= line.count("}")

        if opened and balance == 0:
            return i, block_lines

        i += 1

    return len(lines) - 1, block_lines


# =========================================================
# METHOD DETECTION
# =========================================================

def is_method_start(lines, idx):
    """
    Supports:
      private void foo(...) {
      private void foo(...)
      {
      void foo(...) {
      void foo(...)
      {
    Avoids matching spec header like:
      SomeSpec() {
    because there is no return type.
    """
    line = lines[idx]

    inline_match = METHOD_INLINE_RE.match(line)
    if inline_match:
        method_name = inline_match.group(2)
        return method_name, idx

    sig_match = METHOD_SIGNATURE_RE.match(line)
    if not sig_match:
        return None

    method_name = sig_match.group(2)

    j = idx + 1
    while j < len(lines) and lines[j].strip() == "":
        j += 1

    if j < len(lines) and lines[j].strip() == "{":
        return method_name, j

    return None


def extract_method_block(lines, sig_idx, brace_idx):
    block_lines = []
    balance = 0
    opened = False
    i = sig_idx

    while i < len(lines):
        line = lines[i]
        block_lines.append(line)

        if i >= brace_idx:
            if "{" in line:
                balance += line.count("{")
                opened = True
            if "}" in line:
                balance -= line.count("}")

            if opened and balance == 0:
                return i, block_lines

        i += 1

    return len(lines) - 1, block_lines


# =========================================================
# HELPER METHOD ANALYSIS
# =========================================================

def extract_called_method_names(lines):
    called = []
    for line in lines:
        stripped = line.strip()
        match = METHOD_CALL_RE.match(stripped)
        if match:
            method_name = match.group(1)
            if method_name != "println":
                called.append(method_name)
    return called


def extract_helper_methods(mop_text):
    """
    Extract helper methods outside event blocks.

    Returns:
      method_map = {
        method_name: {
          "direct_logs": [...],
          "calls": [...]
        }
      }
    """
    lines = mop_text.splitlines()
    method_map = {}

    i = 0
    while i < len(lines):
        if EVENT_START_RE.match(lines[i]):
            end_idx, _ = extract_balanced_block(lines, i)
            i = end_idx + 1
            continue

        method_info = is_method_start(lines, i)
        if method_info:
            method_name, brace_idx = method_info
            end_idx, block_lines = extract_method_block(lines, i, brace_idx)

            direct_logs = []
            for block_line in block_lines:
                if LOG_LINE_RE.match(block_line):
                    direct_logs.append(block_line.strip())

            calls = extract_called_method_names(block_lines)

            method_map[method_name] = {
                "direct_logs": direct_logs,
                "calls": calls,
            }

            i = end_idx + 1
            continue

        i += 1

    return method_map


def resolve_method_logs(method_name, method_map, visiting=None, memo=None):
    """
    Recursively resolve logs from:
      - direct logs in the method
      - logs from helper methods called by this method
    """
    if memo is None:
        memo = {}
    if visiting is None:
        visiting = set()

    if method_name in memo:
        return memo[method_name]

    if method_name in visiting:
        return []

    if method_name not in method_map:
        return []

    visiting.add(method_name)

    logs = []
    method_info = method_map[method_name]

    logs.extend(method_info["direct_logs"])

    for called_name in method_info["calls"]:
        if called_name in method_map:
            logs.extend(resolve_method_logs(called_name, method_map, visiting, memo))

    visiting.remove(method_name)

    deduped = []
    seen = set()
    for log_line in logs:
        if log_line not in seen:
            seen.add(log_line)
            deduped.append(log_line)

    memo[method_name] = deduped
    return deduped


def build_resolved_helper_logs(method_map):
    helper_logs = {}
    memo = {}

    for method_name in method_map:
        helper_logs[method_name] = resolve_method_logs(method_name, method_map, set(), memo)

    return helper_logs


def remove_helper_methods(mop_text):
    """
    Remove helper methods outside event blocks.
    """
    lines = mop_text.splitlines()
    result = []

    i = 0
    while i < len(lines):
        if EVENT_START_RE.match(lines[i]):
            end_idx, block_lines = extract_balanced_block(lines, i)
            result.extend(block_lines)
            i = end_idx + 1
            continue

        method_info = is_method_start(lines, i)
        if method_info:
            _, brace_idx = method_info
            end_idx, _ = extract_method_block(lines, i, brace_idx)
            i = end_idx + 1
            continue

        result.append(lines[i])
        i += 1

    return "\n".join(result) + "\n"


# =========================================================
# EVENT CLEANING
# =========================================================

def clean_event_block(block_lines, helper_logs):
    """
    Keeps:
      1. direct RVMLogging lines already inside the event
      2. RVMLogging lines from helper methods called by the event
         including transitive helper-to-helper calls
    """
    open_idx = None
    for i, line in enumerate(block_lines):
        if "{" in line:
            open_idx = i
            break

    if open_idx is None:
        return block_lines

    header_lines = block_lines[:open_idx + 1]
    body_lines = block_lines[open_idx + 1:-1]

    event_indent_match = re.match(r'^(\s*)', header_lines[0])
    event_indent = event_indent_match.group(1) if event_indent_match else ""
    body_indent = event_indent + "\t"

    kept_logs = []

    # direct logs inside event
    for line in body_lines:
        if LOG_LINE_RE.match(line):
            kept_logs.append(line.strip())

    # logs coming from helper methods called by the event
    called_methods = extract_called_method_names(body_lines)
    for method_name in called_methods:
        if method_name in helper_logs:
            kept_logs.extend(helper_logs[method_name])

    # deduplicate preserving order
    deduped_logs = []
    seen = set()
    for log_line in kept_logs:
        if log_line not in seen:
            seen.add(log_line)
            deduped_logs.append(log_line)

    cleaned = []
    cleaned.extend(header_lines)

    for log_line in deduped_logs:
        cleaned.append("{}{}".format(body_indent, log_line))

    cleaned.append("{}}}".format(event_indent))
    return cleaned


def strip_events_keep_logs(mop_text, helper_logs):
    lines = mop_text.splitlines()
    result = []

    i = 0
    while i < len(lines):
        if EVENT_START_RE.match(lines[i]):
            end_idx, block_lines = extract_balanced_block(lines, i)
            cleaned_block = clean_event_block(block_lines, helper_logs)
            result.extend(cleaned_block)
            i = end_idx + 1
        else:
            result.append(lines[i])
            i += 1

    return "\n".join(result) + "\n"


def normalize_event_file(text):
    method_map = extract_helper_methods(text)
    helper_logs = build_resolved_helper_logs(method_map)
    text = remove_helper_methods(text)
    text = strip_events_keep_logs(text, helper_logs)
    return text


# =========================================================
# FILE PROCESSING
# =========================================================

def build_output_path(domain, input_file):
    return OUTPUT_ROOT / domain / input_file.name


def process_file(input_file):
    text = input_file.read_text(encoding="utf-8")
    domain = detect_domain(input_file, text)
    formalism = detect_formalism(text)

    output_file = build_output_path(domain, input_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if formalism == "event":
        cleaned = normalize_event_file(text)
        output_file.write_text(cleaned, encoding="utf-8")
        action = "treated"
    else:
        shutil.copy2(str(input_file), str(output_file))
        action = "copied"

    return {
        "file": input_file.name,
        "domain": domain,
        "formalism": formalism,
        "action": action,
        "input_path": str(input_file),
        "output_path": str(output_file),
    }


# =========================================================
# SUMMARY
# =========================================================

def save_summary_csv(rows, csv_path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file",
        "domain",
        "formalism",
        "action",
        "input_path",
        "output_path",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# =========================================================
# MAIN
# =========================================================

def process_all_files():
    if not INPUT_ROOT.exists():
        raise FileNotFoundError("Input folder not found: {}".format(INPUT_ROOT))

    for domain in sorted(DOMAINS):
        (OUTPUT_ROOT / domain).mkdir(parents=True, exist_ok=True)

    mop_files = sorted(INPUT_ROOT.rglob("*.mop"))

    if not mop_files:
        print("[WARN] No .mop files found in: {}".format(INPUT_ROOT))
        return

    rows = []
    total = 0
    treated = 0
    copied = 0
    errors = 0

    for input_file in mop_files:
        try:
            info = process_file(input_file)
            rows.append(info)

            total += 1
            if info["action"] == "treated":
                treated += 1
            else:
                copied += 1

            print(
                "[OK] {} | domain={} | formalism={} | action={}".format(
                    info["file"],
                    info["domain"],
                    info["formalism"],
                    info["action"],
                )
            )

        except Exception as e:
            errors += 1
            print("[ERROR] {}: {}".format(input_file, e))

    summary_csv = OUTPUT_ROOT / "processing_summary.csv"
    save_summary_csv(rows, summary_csv)

    print("\n========== SUMMARY ==========")
    print("Input root : {}".format(INPUT_ROOT))
    print("Output root: {}".format(OUTPUT_ROOT))
    print("Total OK   : {}".format(total))
    print("Treated    : {}".format(treated))
    print("Copied     : {}".format(copied))
    print("Errors     : {}".format(errors))
    print("CSV        : {}".format(summary_csv))


if __name__ == "__main__":
    process_all_files()
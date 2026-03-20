import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# ==========================================================
# JSON -> MOP
# ==========================================================

def format_signature(signature: Dict) -> str:
    name = signature.get("name", "")
    params = signature.get("parameters", [])
    params_str = ", ".join(f'{p["type"]} {p["name"]}' for p in params)
    return f"{name}({params_str})"


def format_event_method(method: Dict) -> str:
    action = method.get("action", "event")
    name = method["name"]
    timing = method["timing"]
    params = method.get("parameters", [])
    params_str = ", ".join(f'{p["type"]} {p["name"]}' for p in params)

    returning = method.get("returning")
    returning_str = ""
    if returning:
        returning_str = f' returning({returning["type"]} {returning["name"]})'

    funcs = method.get("function", [])
    ops = method.get("operation", [])

    pieces = []
    for i, fn in enumerate(funcs):
        fn_name = fn["name"]
        fn_params = fn.get("parameters", [])

        inner_parts = []
        for p in fn_params:
            ret = p.get("return", "")
            pname = p.get("name", "")
            if ret:
                inner_parts.append(f"{ret} {pname}".strip())
            else:
                inner_parts.append(pname)

        inner = ", ".join(inner_parts)

        if fn_name == "unknown":
            pieces.append(inner)
        else:
            pieces.append(f"{fn_name}({inner})")

        if i < len(ops):
            pieces.append(ops[i])

    pointcut = " ".join(pieces).strip()

    raw_body = method.get("raw_body", [])
    if raw_body:
        body_str = "\n\t\t" + "\n\t\t".join(raw_body) + "\n\t"
    else:
        body_str = ""

    return f'{action} {name} {timing}({params_str}){returning_str} : {pointcut} {{{body_str}}}'


def format_events(events_block: Dict) -> str:
    methods = events_block.get("body", {}).get("methods", [])
    return "\n\t".join(format_event_method(m) for m in methods)


def format_fsm(fsm: Dict) -> str:
    lines = ["fsm :"]
    for state in fsm.get("states", []):
        lines.append(f'\t\t{state["name"]} [')
        for tr in state.get("transitions", []):
            lines.append(f'\t\t\t{tr["event"]} -> {tr["target"]}')
        lines.append("\t\t]")
    return "\n\t".join(lines)


def format_violation(violation: Dict) -> str:
    tag = violation.get("tag", "fail")
    body = violation.get("body", {})
    statements = body.get("statements", [])

    lines = [f"@{tag} {{"]

    for stmt in statements:
        stype = stmt.get("type")
        if stype == "log":
            level = stmt.get("level", "CRITICAL")
            message = stmt.get("message", "__DEFAULT_MESSAGE")
            lines.append(f'\t\tRVMLogging.out.println(Level.{level}, {message});')
        elif stype == "command":
            lines.append(f'\t\t{stmt.get("name", "__RESET")};')
        elif stype == "raw":
            lines.append(f'\t\t{stmt.get("value", "")}')

    lines.append("\t}")
    return "\n".join(lines)


def json_to_mop(data: Dict) -> str:
    signature = format_signature(data["signature"])
    events = format_events(data["ir"]["events"])
    fsm = format_fsm(data["ir"]["fsm"])
    violation = format_violation(data["ir"]["violation"])

    return (
        f"{signature} {{\n"
        f"\t{events}\n\n"
        f"\t{fsm}\n\n"
        f"\t{violation}\n"
        f"}}"
    )


# ==========================================================
# MOP -> JSON
# ==========================================================

_PROP_HEADER_RE = re.compile(r"(?m)^\s*([A-Za-z_]\w*)\s*\(")

EVENT_FULL_RE = re.compile(
    r"(?is)"
    r"\b(creation\s+)?event\s+(\w+)\s+"
    r"(before|after|around)\((.*?)\)\s*"
    r"(?:returning\((.*?)\))?\s*:\s*"
    r"(.*?)\s*\{(.*?)\}"
)

_FSM_LINE_RE = re.compile(r"(?im)^\s*fsm\s*:\s*$")
_STATE_OPEN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\[\s*$")
_TRANS_LINE_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*->\s*([A-Za-z_]\w*)\s*$")

VIOLATION_BLOCK_RE = re.compile(
    r"@(?P<tag>fail|unsafe|err|violation|match)\s*\{(?P<body>.*?)\}",
    flags=re.DOTALL | re.IGNORECASE,
)

LOG_STMT_RE = re.compile(
    r"""RVMLogging\.out\.println\(\s*
        Level\.(?P<level>[A-Z_]+)\s*,\s*
        (?P<message>.*?)
        \s*\)\s*;?\s*$
    """,
    flags=re.VERBOSE,
)


def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def _extract_balanced_parens(s: str, open_pos: int) -> Tuple[Optional[str], Optional[int]]:
    depth = 0
    i = open_pos
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return s[open_pos + 1 : i], i
        i += 1
    return None, None


def _split_commas_balanced(s: str) -> List[str]:
    out: List[str] = []
    cur: List[str] = []

    angle = 0
    paren = 0
    bracket = 0

    for ch in s:
        if ch == "<":
            angle += 1
        elif ch == ">":
            angle = max(0, angle - 1)
        elif ch == "(":
            paren += 1
        elif ch == ")":
            paren = max(0, paren - 1)
        elif ch == "[":
            bracket += 1
        elif ch == "]":
            bracket = max(0, bracket - 1)

        if ch == "," and angle == 0 and paren == 0 and bracket == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)

    if cur:
        out.append("".join(cur).strip())

    return out


def _split_type_name(param: str) -> Tuple[str, str]:
    param = re.sub(r"@\w+(\([^)]*\))?\s*", "", param).strip()
    param = re.sub(r"^\s*final\s+", "", param).strip()

    tokens = param.split()
    if len(tokens) < 2:
        return "", ""

    name = tokens[-1]
    ptype = " ".join(tokens[:-1]).strip()
    return ptype, name


def extract_signature(text: str) -> Dict:
    m = _PROP_HEADER_RE.search(text)
    if not m:
        return {"parameters": []}

    name = m.group(1)
    open_paren = text.find("(", m.start())
    if open_paren == -1:
        return {"parameters": []}

    params_raw, _ = _extract_balanced_parens(text, open_paren)
    if params_raw is None:
        return {"parameters": []}

    params_raw = params_raw.strip()
    if not params_raw:
        return {"name": name, "parameters": []}

    parts = _split_commas_balanced(params_raw)
    params = []

    for part in parts:
        ptype, pname = _split_type_name(part.strip())
        if ptype and pname:
            params.append({"type": ptype, "name": pname})

    return {"name": name, "parameters": params}


def parse_parameters(param_text: str) -> List[Dict]:
    if not param_text.strip():
        return []

    parts = _split_commas_balanced(param_text)
    params = []

    for p in parts:
        ptype, pname = _split_type_name(p.strip())
        if ptype and pname:
            params.append({"type": ptype, "name": pname})

    return params


def split_logical_operators_balanced(s: str) -> Tuple[List[str], List[str]]:
    clauses: List[str] = []
    operators: List[str] = []

    cur: List[str] = []
    depth = 0
    i = 0
    n = len(s)

    while i < n:
        ch = s[i]

        if ch == "(":
            depth += 1
            cur.append(ch)
            i += 1
            continue

        if ch == ")":
            depth = max(0, depth - 1)
            cur.append(ch)
            i += 1
            continue

        if depth == 0 and i + 1 < n:
            two = s[i:i+2]
            if two in ("&&", "||"):
                clause = "".join(cur).strip()
                if clause:
                    clauses.append(clause)
                operators.append(two)
                cur = []
                i += 2
                continue

        cur.append(ch)
        i += 1

    tail = "".join(cur).strip()
    if tail:
        clauses.append(tail)

    return clauses, operators


def parse_single_pointcut_function(part: str) -> Dict:
    part = normalize(part)

    m = re.match(r"^([A-Za-z_]\w*)\((.*)\)$", part, flags=re.DOTALL)
    if not m:
        return {
            "name": "unknown",
            "parameters": [{"return": "", "name": part}]
        }

    fname = m.group(1).strip()
    inner = m.group(2).strip()

    param_obj = {"return": "", "name": inner}

    if fname == "call":
        star_match = re.match(r"^(\*+)\s+(.*)$", inner, flags=re.DOTALL)
        if star_match:
            param_obj["return"] = star_match.group(1).strip()
            param_obj["name"] = star_match.group(2).strip()

    return {
        "name": fname,
        "parameters": [param_obj]
    }


def parse_pointcut_functions(pointcut: str) -> Dict:
    raw = normalize(pointcut.strip())
    clauses, operations = split_logical_operators_balanced(raw)
    functions = [parse_single_pointcut_function(clause) for clause in clauses]

    return {
        "procediments": ":",
        "function": functions,
        "operation": operations
    }


def extract_events(text: str) -> Dict:
    methods: List[Dict] = []

    for creation_kw, name, timing, params, returning, pointcut, body in EVENT_FULL_RE.findall(text):
        action = "creation event" if (creation_kw and creation_kw.strip()) else "event"

        method: Dict = {
            "action": action,
            "name": name.strip(),
            "timing": timing.strip(),
            "parameters": parse_parameters(params),
        }

        if returning and returning.strip():
            rtype, rname = _split_type_name(returning.strip())
            if rtype and rname:
                method["returning"] = {
                    "type": rtype,
                    "name": rname
                }

        pointcut_info = parse_pointcut_functions(pointcut)
        method["procediments"] = pointcut_info["procediments"]
        method["function"] = pointcut_info["function"]
        method["operation"] = pointcut_info["operation"]

        body_lines = [ln.rstrip() for ln in body.splitlines() if ln.strip()]
        if body_lines:
            method["raw_body"] = body_lines

        methods.append(method)

    return {"body": {"methods": methods}}


def _find_next_directive_or_property_end(text: str, start: int) -> int:
    m_dir = re.search(r"(?im)^\s*@(fail|unsafe|err|violation|match)\b", text[start:])
    if m_dir:
        return start + m_dir.start()

    m_end = re.search(r"(?m)^\s*\}\s*$", text[start:])
    if m_end:
        return start + m_end.start()

    return len(text)


def _build_fsm_ast(lines: List[str]) -> Dict:
    ast_states: List[Dict] = []
    initial_state: Optional[str] = None

    current_state: Optional[Dict] = None
    inside = False

    for ln in lines:
        line = ln.rstrip()

        m_open = _STATE_OPEN_RE.match(line)
        if m_open:
            state_name = m_open.group(1)
            current_state = {"name": state_name, "transitions": []}
            ast_states.append(current_state)

            if initial_state is None:
                initial_state = state_name

            inside = True
            continue

        if inside and line.strip() == "]":
            inside = False
            current_state = None
            continue

        if inside and current_state is not None:
            m_tr = _TRANS_LINE_RE.match(line)
            if m_tr:
                ev, dst = m_tr.group(1), m_tr.group(2)
                current_state["transitions"].append({
                    "event": ev,
                    "target": dst
                })

    return {
        "type": "fsm",
        "initial_state": initial_state,
        "states": ast_states,
    }


def extract_fsm_block(text: str) -> Dict:
    m = _FSM_LINE_RE.search(text)
    if not m:
        return {
            "type": "fsm",
            "initial_state": None,
            "states": []
        }

    start = m.end()
    end = _find_next_directive_or_property_end(text, start)
    block_text = text[start:end].rstrip()
    raw_lines = [ln.rstrip("\n") for ln in block_text.splitlines() if ln.strip() != ""]

    return _build_fsm_ast(raw_lines)


def extract_violation_block(text: str) -> Dict:
    m = VIOLATION_BLOCK_RE.search(text)

    if not m:
        return {
            "tag": None,
            "body": {
                "statements": [],
                "has_reset": False
            }
        }

    tag = m.group("tag").lower()
    block = m.group("body").strip()

    raw_lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
    statements: List[Dict] = []
    has_reset = False

    for line in raw_lines:
        clean = line.rstrip().rstrip(",")

        if clean == "__RESET;" or clean == "__RESET":
            statements.append({
                "type": "command",
                "name": "__RESET"
            })
            has_reset = True
            continue

        log_match = LOG_STMT_RE.match(clean)
        if log_match:
            level = log_match.group("level").strip()
            message = log_match.group("message").strip()

            statements.append({
                "type": "log",
                "level": level,
                "message": message
            })
            continue

        statements.append({
            "type": "raw",
            "value": clean
        })

    return {
        "tag": tag,
        "body": {
            "statements": statements,
            "has_reset": has_reset
        }
    }


def mop_to_json(text: str, spec_id: str, domain: str) -> Dict:
    return {
        "id": spec_id,
        "formalism": "fsm",
        "domain": domain,
        "signature": extract_signature(text),
        "ir": {
            "events": extract_events(text),
            "fsm": extract_fsm_block(text),
            "violation": extract_violation_block(text),
        }
    }


# ==========================================================
# FILE HELPERS
# ==========================================================

def convert_json_file_to_mop(json_path: str, mop_path: Optional[str] = None) -> str:
    src = Path(json_path)
    data = json.loads(src.read_text(encoding="utf-8"))
    mop_text = json_to_mop(data)

    if mop_path is None:
        mop_path = str(src.with_suffix(".mop"))

    Path(mop_path).write_text(mop_text, encoding="utf-8")
    return mop_path


def convert_mop_file_to_json(mop_path: str, json_path: Optional[str] = None,
                             spec_id: Optional[str] = None, domain: str = "") -> str:
    src = Path(mop_path)
    text = src.read_text(encoding="utf-8")

    if spec_id is None:
        spec_id = src.stem

    data = mop_to_json(text, spec_id=spec_id, domain=domain)

    if json_path is None:
        json_path = str(src.with_suffix(".json"))

    Path(json_path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return json_path


# ==========================================================
# EXAMPLE USAGE
# ==========================================================

if __name__ == "__main__":
    # JSON -> MOP
    # out_mop = convert_json_file_to_mop("fsm_01.json")

    # MOP -> JSON
    # out_json = convert_mop_file_to_json("fsm_01.mop", domain="io")

    pass
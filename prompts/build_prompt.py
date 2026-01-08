import json
from pathlib import Path
from typing import List

# Diretórios base
PROMPTS_DIR = Path(__file__).resolve().parent
NL2SPEC_DIR = PROMPTS_DIR.parent
DATASETS_DIR = NL2SPEC_DIR / "datasets"

SUPPORTED_IR_TYPES = {"fsm", "ere", "event", "ltl"}



# ---------- loaders ----------

def load_text_from_prompts(relative_path: str) -> str:
    path = PROMPTS_DIR / relative_path
    return path.read_text(encoding="utf-8")


def load_json_from_dataset(relative_path: str):
    path = DATASETS_DIR / relative_path
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------- formatting ----------

def format_fewshot_examples(examples: List[dict]) -> str:
    blocks = []
    for i, ex in enumerate(examples, start=1):
        blocks.append(
            f"Example {i}:\n{json.dumps(ex, indent=2)}"
        )
    return "\n\n".join(blocks)


# ---------- builder ----------

def build_prompt(
    ir_type: str,
    scenario_text: str,
    fewshot_files: List[str]
) -> str:
    """
    Build a prompt for a given IR type (fsm, ere, event).
    """

    ir_type = ir_type.lower()
    if ir_type not in SUPPORTED_IR_TYPES:
        raise ValueError(
            f"Unsupported ir_type '{ir_type}'. "
            f"Supported types: {SUPPORTED_IR_TYPES}"
        )

    # fixed files
    header = load_text_from_prompts("base/header.txt")
    template = load_text_from_prompts(f"{ir_type}/template.txt")

    # few-shot examples
    fewshot_examples = [
        load_json_from_dataset(p) for p in fewshot_files
    ]
    fewshot_block = format_fewshot_examples(fewshot_examples)

    # inject examples
    template = template.replace(
        "{{FEW_SHOT_EXAMPLES}}",
        fewshot_block
    )

    task = f"""
Task:
Generate a {ir_type.upper()} IR in JSON for the following natural language rule.

Natural language description:
\"\"\"
{scenario_text}
\"\"\"
"""

    return "\n\n".join([header, template, task])

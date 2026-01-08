from typing import Dict, Any


def build_experiment_log(
    scenario: Dict[str, Any],
    config: Dict[str, Any],
    ir_type: str,
    prompt: str,
    ir_result: Dict[str, Any],
    generation_time_ms: int,
    valid_ir: bool
) -> Dict[str, Any]:

    prompting = config["prompting"]

    if prompting["shot_mode"] == "zero":
        k = 0
    elif prompting["shot_mode"] == "one":
        k = 1
    else:
        k = prompting.get("k", 0)

    return {
        # scenario
        "scenario_id": scenario.get("id"),
        "category": ir_type,

        # prompting
        "shot_mode": prompting["shot_mode"],
        "k": k,

        # llm
        "llm_provider": config["llm"]["provider"],

        # efficiency
        "prompt_length": len(prompt),
        "generation_time_ms": generation_time_ms,

        # quality
        "valid_ir": valid_ir,
        "ir_size": len(str(ir_result)) if ir_result else 0,
    }

import json
from pathlib import Path

from nl2spec.config import load_config
from nl2spec.pipeline.generate import generate_one
from nl2spec.pipeline.prompting import resolve_fewshot_files
from nl2spec.pipeline.logging import build_experiment_log
from nl2spec.pipeline.infer_ir_type import infer_ir_type
from nl2spec.core.handlers.fewshot_loader import FewShotLoader
from nl2spec.core.llms.factory import load_llm
from nl2spec.pipeline.export import export_logs_to_csv


def run_batch():
    config = load_config("nl2spec/config.yaml")

    # Load scenarios
    with open(config["paths"]["dataset_nl"], "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    # Init components
    loader = FewShotLoader(
        fewshot_dir=config["prompting"]["fewshot"]["dataset_dir"],
        seed=42
    )

    llm = load_llm(config)
    schema_path = config["paths"]["schema_ir"]

    logs = []

    for i, scenario in enumerate(scenarios, start=1):
        print(f"[{i}/{len(scenarios)}] Scenario {scenario.get('id')}")

        try:
            ir_type = infer_ir_type(scenario)

            fewshot_files = resolve_fewshot_files(
                config=config,
                loader=loader,
                ir_type=ir_type
            )

            result = generate_one(
                scenario=scenario,
                ir_type=ir_type,
                fewshot_files=fewshot_files,
                llm=llm,
                schema_path=schema_path
            )

            log = build_experiment_log(
                scenario=scenario,
                config=config,
                ir_type=ir_type,
                prompt=result["prompt"],
                ir_result=result["ir"],
                generation_time_ms=result["generation_time_ms"],
                valid_ir=True
            )

        except Exception as e:
            log = {
                "scenario_id": scenario.get("id"),
                "category": scenario.get("category"),
                "shot_mode": config["prompting"]["shot_mode"],
                "k": config["prompting"].get("k", 0),
                "llm_provider": config["llm"]["provider"],
                "valid_ir": False,
                "error": str(e)
            }

        logs.append(log)

    export_logs_to_csv(
        logs,
        output_dir=config["paths"]["output_dir"]
    )


if __name__ == "__main__":
    run_batch()

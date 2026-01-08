from nl2spec.config import load_config
from nl2spec.core.handlers.fewshot_loader import FewShotLoader
from nl2spec.pipeline.prompting import resolve_fewshot_files
from nl2spec.pipeline.infer_ir_type import infer_ir_type
from nl2spec.prompts.build_prompt import build_prompt


def test_config_fewshot_loader_build_prompt_integration():
    config = load_config("nl2spec/config.yaml")

    loader = FewShotLoader(
        fewshot_dir=config["prompting"]["fewshot"]["dataset_dir"],
        seed=42
    )

    scenario = {
        "id": "t1",
        "category": "FSM",
        "natural_language": "A socket must be configured before sending data."
    }

    ir_type = infer_ir_type(scenario)

    fewshot_files = resolve_fewshot_files(config, loader, ir_type)

    prompt = build_prompt(
        ir_type=ir_type,
        scenario_text=scenario["natural_language"],
        fewshot_files=fewshot_files
    )

    assert "Natural language description" in prompt

    if config["prompting"]["shot_mode"] != "zero":
        assert "Example 1" in prompt
    print(prompt)


if __name__ == "__main__":
    test_config_fewshot_loader_build_prompt_integration()
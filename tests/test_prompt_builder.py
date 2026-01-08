from nl2spec.prompts.build_prompt import build_prompt


def test_prompt_builder_fsm():
    scenario_text = "A socket must be configured before sending data."

    prompt = build_prompt(
        ir_type="fsm",
        scenario_text=scenario_text,
        fewshot_files=[
            "fewshot/fsm/fsm_02.json"
        ]
    )

    print("\n===== GENERATED PROMPT =====\n")
    print(prompt)
    print("\n===== END PROMPT =====\n")

    assert "Example 1" in prompt
    assert "finite state machine" in prompt.lower()
    assert "socket must be configured" in prompt.lower()

def test_prompt_builder_ere():
    scenario_text = (
        "A file must be opened before it is read and "
        "must not be read after it is closed."
    )

    prompt = build_prompt(
        ir_type="ere",
        scenario_text=scenario_text,
        fewshot_files=[
            "fewshot/ere/ere_01.json"
        ]
    )

    print("\n===== GENERATED ERE PROMPT =====\n")
    print(prompt)
    print("\n===== END ERE PROMPT =====\n")

    # sanity checks
    assert "Example 1" in prompt
    assert "extended regular expression" in prompt.lower()
    assert "file must be opened" in prompt.lower()


if __name__ == "__main__":
    test_prompt_builder_ere()
    test_prompt_builder_fsm()
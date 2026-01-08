def infer_ir_type(scenario: dict) -> str:
    category = scenario.get("category")

    if not category:
        raise ValueError("Scenario does not define 'category'")

    return category.lower()

def resolve_fewshot_files(config, loader, ir_type: str):
    prompting = config["prompting"]

    if not prompting["fewshot"]["enabled"]:
        return []

    mode = prompting["shot_mode"]

    if mode == "zero":
        return []

    if mode == "one":
        return loader.sample(ir_type, k=1)

    if mode == "few":
        k = prompting.get("k", 1)
        return loader.sample(ir_type, k=k)

    raise ValueError(f"Unknown shot_mode: {mode}")

import pandas as pd

df = pd.read_csv("outputs/experiment_runs.csv")

table = (
    df.groupby(["shot_mode", "k"])
      .agg(
          valid_ir_rate=("valid_ir", "mean"),
          avg_time_ms=("generation_time_ms", "mean"),
          avg_prompt_len=("prompt_length", "mean"),
          avg_ir_size=("ir_size", "mean")
      )
      .reset_index()
)

table["valid_ir_rate"] = table["valid_ir_rate"] * 100

print(table)

# Export to LaTeX
latex = table.to_latex(
    index=False,
    float_format="%.2f",
    caption="Ablation study on prompting strategies.",
    label="tab:ablation_prompting"
)

with open("outputs/ablation_table.tex", "w") as f:
    f.write(latex)

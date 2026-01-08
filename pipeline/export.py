import pandas as pd
from pathlib import Path


def export_logs_to_csv(logs, output_dir: str, filename="experiment_runs.csv"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(logs)
    path = Path(output_dir) / filename
    df.to_csv(path, index=False)

    return path

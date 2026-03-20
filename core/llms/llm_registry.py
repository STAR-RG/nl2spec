import csv
from pathlib import Path


class LLMRegistry:

    def __init__(self, csv_path: Path):
        self.registry = {}
        self._load(csv_path)

    def _load(self, csv_path: Path):

        if not csv_path.exists():
            raise FileNotFoundError(f"LLM registry file not found: {csv_path}")

        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:

                provider = row["provider"].strip().lower()
                model = row["model"].strip().lower()

                key = (provider, model)

                self.registry[key] = {
                    "provider": provider,
                    "model": model,
                    "api_key": row["api_key"],
                    "temperature": float(row.get("temperature", 0)),
                    "max_tokens": int(row.get("max_tokens", 2000))
                }

    def get(self, provider: str, model: str):

        key = (provider.lower(), model.lower())

        if key not in self.registry:
            raise ValueError(
                f"Provider '{provider}' with model '{model}' not found in LLM registry"
            )

        return self.registry[key]
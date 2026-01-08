import random
from pathlib import Path
from typing import List, Optional


class FewShotLoader:
    """
    Loader for few-shot examples (paths only).

    Responsibilities:
    - Discover few-shot JSON files
    - Organize by category (folder name)
    - Deterministically sample file paths

    It does NOT parse or validate IRs.
    """

    def __init__(self, fewshot_dir: str, seed: int = 42):
        self.fewshot_dir = Path(fewshot_dir)
        self.seed = seed

        if not self.fewshot_dir.exists():
            raise FileNotFoundError(
                f"Few-shot directory not found: {self.fewshot_dir}"
            )

        self._index = self._build_index()

    def _build_index(self):
        index = {}

        for category_dir in self.fewshot_dir.iterdir():
            if not category_dir.is_dir():
                continue

            files = sorted(category_dir.glob("*.json"))
            if files:
                index[category_dir.name.lower()] = files

        return index

    def list_all(self, category: Optional[str] = None) -> List[str]:
        if category:
            return [
                str(p.relative_to(self.fewshot_dir.parent))
                for p in self._index.get(category.lower(), [])
            ]

        all_files = []
        for files in self._index.values():
            all_files.extend(files)

        return [
            str(p.relative_to(self.fewshot_dir.parent))
            for p in all_files
        ]

    def sample(self, category: str, k: int) -> List[str]:
        pool = self._index.get(category.lower(), [])

        if not pool or k <= 0:
            return []

        rng = random.Random(self.seed)

        if k >= len(pool):
            selected = pool
        else:
            selected = rng.sample(pool, k)

        return [
            str(p.relative_to(self.fewshot_dir.parent))
            for p in selected
        ]

"""Data loader for donoharm benchmark.

Loads clinical case vignettes (base + perturbed) from dataset/items.jsonl
and prepends the system prompt from prompts/{name}.md.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass
class BenchmarkItem:
    id: str
    prompt: str
    metadata: Dict[str, Any]
    ground_truth: Optional[str] = None


class DonoHarmLoader:
    def __init__(self, config: Dict[str, Any], prompt_name: str = "default"):
        self.config = config
        self.name = config.get("benchmark", {}).get("name", "donoharm")

        self.script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = self.script_dir / "dataset"
        self.prompts_dir = self.script_dir / "prompts"
        self.items_path = self.dataset_dir / "items.jsonl"

        prompt_path = self.prompts_dir / f"{prompt_name}.md"
        if not prompt_path.resolve().is_relative_to(self.prompts_dir.resolve()):
            raise ValueError(f"Invalid prompt name: {prompt_name}")
        if not prompt_path.exists():
            raise FileNotFoundError(
                f"Prompt '{prompt_name}' not found at {prompt_path}"
            )

        self.prompt_template = prompt_path.read_text().strip()
        self._items_cache: Optional[List[BenchmarkItem]] = None

    def _load_all_items(self) -> List[BenchmarkItem]:
        if self._items_cache is not None:
            return self._items_cache

        items = []
        if self.items_path.exists():
            with open(self.items_path, "r") as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            prompt = data.get("prompt", "")
                            if self.prompt_template:
                                prompt = f"{self.prompt_template}\n\n{prompt}"
                            items.append(BenchmarkItem(
                                id=str(data.get("id", "")),
                                prompt=prompt,
                                metadata=data.get("metadata", {}),
                                ground_truth=data.get("ground_truth"),
                            ))
                        except json.JSONDecodeError:
                            continue

        self._items_cache = items
        return items

    def load_items(self, limit: Optional[int] = None) -> Iterator[BenchmarkItem]:
        all_items = self._load_all_items()
        if limit is not None:
            all_items = all_items[:limit]
        for item in all_items:
            yield item

    def get_item(self, id: str) -> Optional[BenchmarkItem]:
        for item in self._load_all_items():
            if item.id == id:
                return item
        return None


def create_loader(config: Dict[str, Any], prompt_name: str = "default") -> DonoHarmLoader:
    return DonoHarmLoader(config, prompt_name=prompt_name)

"""
SCT (Script Concordance Test) data loader.

Loads SCT questions from JSONL and builds prompts for inference.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Iterator, Optional, List


# Subtest short names -> full source names in the CSV
SUBTESTS = {
    "adelaide": "Adelaide SCT",
    "iu_national": "IU National SCT",
    "iu_em": "IU SCT-EM",
    "iu_student": "IU Student SCT",
    "infant_lp": "Infant LP SCT",
    "neurology": "Neurology SCT",
    "open_medical": "Open Medical SCT",
    "physio": "Physiotherapy SCT",
    "singapore_im": "singapore_im",
    "singapore_neuro": "singapore_neuro",
}

# Reverse mapping
SOURCE_TO_SHORT = {v: k for k, v in SUBTESTS.items()}


@dataclass
class SCTItem:
    """Represents a single SCT benchmark item."""
    id: str  # Unified field name (was item_id)
    prompt: str
    metadata: Dict[str, Any]

    # SCT-specific fields
    scenario: str
    hypothesis: str
    new_information: str
    source: str  # Which subtest this came from
    expert_distribution: List[float]  # [dist for -2, -1, 0, +1, +2]


class SCTLoader:
    """
    Loads SCT benchmark items from JSONL files.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the loader with benchmark configuration.

        Args:
            config: Merged config dict (model + benchmark configs)
        """
        self.config = config
        self.name = config.get("benchmark", {}).get("name", "sct")

        # SCT-specific settings
        sct_config = config.get("sct", {})
        self.few_shot = sct_config.get("few_shot", False)
        self.reason = sct_config.get("reason", False)
        self.test_mode = sct_config.get("test_mode", False)

        # Locate dataset directory
        self.script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
        self.dataset_dir = self.script_dir / "dataset"
        self.templates_dir = self.script_dir / "templates"

        # Load templates
        self.templates = self._load_templates()

        # Cache for items and rubrics
        self._items_cache: Optional[List[dict]] = None
        self._rubrics_cache: Optional[Dict[str, dict]] = None

    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates from files."""
        templates = {}
        template_names = [
            "guideline", "testcase",
            "example_-2", "example_-1", "example_0", "example_+1", "example_+2"
        ]
        for name in template_names:
            path = self.templates_dir / f"{name}.md"
            if path.exists():
                templates[name] = path.read_text()
        return templates

    def _build_prompt(self, item: dict) -> str:
        """Build the prompt for an SCT item."""
        # Start with guideline
        guideline = self.templates.get("guideline", "")

        # Remove explanation requirement if not reasoning
        if not self.reason:
            guideline = guideline.replace(
                ' and a brief explanation for your choice', ''
            )

        prompt = guideline

        # Add few-shot examples if requested
        if self.few_shot:
            prompt += "\n## Examples with Response Labels\n"
            for rating in ['-2', '-1', '0', '+1', '+2']:
                if f"example_{rating}" in self.templates:
                    prompt += self.templates[f"example_{rating}"]

        # Add the test case
        testcase = self.templates.get("testcase", "")
        testcase = testcase.replace('{{ scenario }}', item['scenario'])
        testcase = testcase.replace('{{ hypothesis }}', item['hypothesis'])
        testcase = testcase.replace('{{ additional information }}', item['new_information'])
        prompt += testcase

        return prompt

    def _load_items_from_jsonl(self) -> List[dict]:
        """Load raw items from items.jsonl."""
        if self._items_cache is not None:
            return self._items_cache

        items = []
        items_path = self.dataset_dir / "items.jsonl"
        if items_path.exists():
            with open(items_path, 'r') as f:
                for line in f:
                    if line.strip():
                        items.append(json.loads(line))

        self._items_cache = items
        return items

    def _load_rubrics(self) -> Dict[str, dict]:
        """Load rubrics (expert distributions) from rubrics.jsonl."""
        if self._rubrics_cache is not None:
            return self._rubrics_cache

        rubrics = {}
        rubrics_path = self.dataset_dir / "rubrics.jsonl"
        if rubrics_path.exists():
            with open(rubrics_path, 'r') as f:
                for line in f:
                    if line.strip():
                        obj = json.loads(line)
                        rubrics[obj["id"]] = obj

        self._rubrics_cache = rubrics
        return rubrics

    def load_items(
        self,
        subset: str = "all",
        limit: Optional[int] = None
    ) -> Iterator[SCTItem]:
        """
        Load benchmark items.

        Args:
            subset: Subset name - "all" or a specific subtest short name
            limit: Maximum number of examples to load (None = all)

        Yields:
            SCTItem objects with prompts ready for inference
        """
        raw_items = self._load_items_from_jsonl()
        rubrics = self._load_rubrics()

        # Determine which sources to include
        if subset == "all":
            selected_sources = set(SUBTESTS.values())
        elif subset in SUBTESTS:
            selected_sources = {SUBTESTS[subset]}
        else:
            # Try treating it as a full source name
            selected_sources = {subset}

        # Filter by source
        filtered_items = [
            item for item in raw_items
            if item.get("source") in selected_sources
        ]

        # Apply test mode (first 5 per subtest)
        if self.test_mode:
            by_source = {}
            for item in filtered_items:
                src = item.get("source", "unknown")
                if src not in by_source:
                    by_source[src] = []
                if len(by_source[src]) < 5:
                    by_source[src].append(item)
            filtered_items = [item for items in by_source.values() for item in items]

        # Apply limit
        if limit is not None:
            filtered_items = filtered_items[:limit]

        # Yield items
        for item in filtered_items:
            item_id = str(item["id"])
            rubric = rubrics.get(item_id, {})

            yield SCTItem(
                id=item_id,
                prompt=self._build_prompt(item),
                metadata={
                    "source": item.get("source"),
                    "source_short": SOURCE_TO_SHORT.get(item.get("source"), item.get("source")),
                },
                scenario=item["scenario"],
                hypothesis=item["hypothesis"],
                new_information=item["new_information"],
                source=item.get("source", "unknown"),
                expert_distribution=rubric.get("expert_distribution", [0, 0, 0, 0, 0]),
            )

    def get_item(self, item_id: str) -> Optional[SCTItem]:
        """Get a specific item by ID."""
        for item in self.load_items():
            if item.item_id == item_id:
                return item
        return None

    def get_ground_truth(self, item_id: str) -> Optional[List[float]]:
        """
        Get expert distribution for a specific item.

        Returns:
            List of 5 floats [dist for -2, -1, 0, +1, +2]
        """
        rubrics = self._load_rubrics()
        rubric = rubrics.get(item_id)
        if rubric:
            return rubric.get("expert_distribution")
        return None


# Convenience function for creating loader
def create_loader(config: Dict[str, Any]) -> SCTLoader:
    """Create an SCT loader with the given config."""
    return SCTLoader(config)

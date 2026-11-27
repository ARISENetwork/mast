"""Run with python -m unittest discover -s <SCT directory>."""
import csv
import json
import shutil
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from score import parse_response, read_trial


class RatingTests(unittest.TestCase):
    def test_plain_and_structured_ratings(self) -> None:
        for text, rating in [
            ("**Rating:** 2, revised Rating: −1", -1),
            ("Rating: – 2", -2),
            ('{"Rating": "−2"}', -2),
            (json.dumps({"rating": "−1"}), -1),
            ('{"Rating": +1}', 1),
            ("Rating: 12", None),
        ]:
            with self.subTest(text=text):
                self.assertEqual(parse_response(text), rating)


class TrialTests(unittest.TestCase):
    def read(self, rows: list[dict]) -> dict[str, dict]:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "responses.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
            return read_trial(path, 1, {"1"}, {"1", "2"})

    def test_integer_id_and_selected_scope(self) -> None:
        row = {"id": 1, "response": "Rating: +1"}
        self.assertEqual(self.read([row, {"id": "2"}, {"id": "1", "trial": 2}]), {"1": row})

    def test_unusable_trial_is_rejected(self) -> None:
        row = {"id": "1", "response": "Rating: +1"}
        for rows, reason in [
            ([dict(row, trial=2)], "missing"),
            ([row, dict(row, id=1)], "duplicate"),
            ([row, dict(row, id=3)], "unknown"),
            ([dict(row, response=" ")], "empty or failed"),
            ([dict(row, error="timed out")], "empty or failed"),
        ]:
            with self.subTest(reason=reason), self.assertRaisesRegex(SystemExit, reason):
                self.read(rows)


class ScoreCommandTests(unittest.TestCase):
    def test_minimal_input_uses_dataset_subtest_and_requires_rubric(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            kit = root / "benchmarks" / "sct"
            dataset = kit / "dataset"
            dataset.mkdir(parents=True)
            for name in ("score.py", "data_loader.py"):
                shutil.copy(Path(__file__).with_name(name), kit / name)
            item = dict(id="7", source="Adelaide SCT", scenario="", hypothesis="", new_information="")
            (dataset / "items.jsonl").write_text(json.dumps(item) + "\n")
            rubric = dict(id="7", expert_distribution=[1, 0, 0, 0, 0])
            (dataset / "rubrics.jsonl").write_text(json.dumps(rubric) + "\n")
            raw = root / "results" / "raw" / "sct"
            raw.mkdir(parents=True)
            (raw / "test.jsonl").write_text(json.dumps({"id": 7, "trial": 1, "response": "Rating: −2"}) + "\n")
            (root / "model.yaml").write_text("model: {name: test}\n")
            (root / "bench.yaml").write_text("benchmark: {name: sct}\n")
            command = [sys.executable, "-B", str(kit / "score.py"), "--model-config", "model.yaml",
                       "--benchmark-config", "bench.yaml"]
            success = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertEqual(success.returncode, 0, success.stderr)
            csv_path = root / "results" / "scores" / "sct" / "test.csv"
            with csv_path.open() as stream:
                scores = {row["category"]: float(row["mean"]) for row in csv.DictReader(stream)
                          if row["metric"] == "sct_score"}
            self.assertEqual(scores, {"Overall": 1.0, "adelaide": 1.0})

            (dataset / "rubrics.jsonl").write_text("")
            failure = subprocess.run(command, cwd=root, capture_output=True, text=True)
            self.assertNotEqual(failure.returncode, 0)
            self.assertIn("expert distribution for item 7", failure.stderr)

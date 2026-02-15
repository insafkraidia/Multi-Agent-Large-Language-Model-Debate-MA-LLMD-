# filename: dataset_processing.py
from __future__ import annotations

from dataclasses import dataclass
from glob import glob
from typing import Any, Dict, Optional, Type

import json
import pandas as pd
import datasets


# ============================================================
# Paper terminology: Dataset Processing / Unified Representation
# ============================================================

@dataclass
class DatasetConfig:
    dataset_name: str
    n_samples: int = 50
    data_dir: str = "data"
    seed: int = 0
    context: bool = False  # optional context augmentation


class BaseBenchmarkDataset:
    """
    Paper Figure 2:
    - load raw samples
    - optional dataset-dependent filtering
    - sampling into selected subset
    - __getitem__ returns raw record (dict)
    """

    def __init__(self, cfg: DatasetConfig):
        self.cfg = cfg
        self.data: pd.DataFrame = pd.DataFrame()
        self.selected_data: pd.DataFrame = pd.DataFrame()
        self.load_data()
        self.filter_data()
        self.sample_data()

    # --- to override ---
    def load_data(self) -> None:
        raise NotImplementedError

    def filter_data(self) -> None:
        # optional override
        return

    # --- shared ---
    def sample_data(self) -> None:
        if len(self.data) == 0:
            raise ValueError(f"{self.cfg.dataset_name}: loaded empty dataset.")
        n = min(self.cfg.n_samples, len(self.data))
        self.selected_data = self.data.sample(n=n, random_state=self.cfg.seed).reset_index(drop=True)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.selected_data.iloc[idx].to_dict()

    def __len__(self) -> int:
        return len(self.selected_data)


# ============================================================
# Dataset loaders
# ============================================================

class MMLUDataset(BaseBenchmarkDataset):
    def load_data(self) -> None:
        pattern = f"{self.cfg.data_dir}/{self.cfg.dataset_name}/test/*.csv"
        files = glob(pattern)
        if not files:
            raise FileNotFoundError(f"MMLU: no csv files found at {pattern}")
        dfs = [pd.read_csv(f, header=None) for f in files]
        self.data = pd.concat(dfs, ignore_index=True)


class MathDataset(BaseBenchmarkDataset):
    def __init__(self, cfg: DatasetConfig, level_geq: int = 1):
        self.level_geq = level_geq
        super().__init__(cfg)

    def load_data(self) -> None:
        base = f"{self.cfg.data_dir}/{self.cfg.dataset_name}/test"
        categories = glob(base + "/*")
        rows = []
        for cat in categories:
            for fp in glob(cat + "/*.json"):
                with open(fp, "r") as f:
                    rows.append(json.load(f))
        self.data = pd.DataFrame(rows)
        # normalize "level" like "Level 3"
        self.data["level_int"] = self.data["level"].apply(lambda x: int(str(x).split(" ")[-1]))

    def filter_data(self) -> None:
        self.data = self.data[self.data["level_int"] >= self.level_geq].reset_index(drop=True)


class ChessDataset(BaseBenchmarkDataset):
    def load_data(self) -> None:
        pattern = f"{self.cfg.data_dir}/{self.cfg.dataset_name}/synthetic_short/*.json"
        files = glob(pattern)
        if not files:
            raise FileNotFoundError(f"Chess: no json files found at {pattern}")
        payload = json.load(open(files[0], "r"))
        self.data = pd.DataFrame(payload["examples"])


class MQuakeDataset(BaseBenchmarkDataset):
    def load_data(self) -> None:
        fp = f"{self.cfg.data_dir}/{self.cfg.dataset_name}/MQuAKE-CF-3k.json"
        payload = json.load(open(fp, "r"))
        self.data = pd.DataFrame(payload)


class MusiqueDataset(BaseBenchmarkDataset):
    def load_data(self) -> None:
        fp = f"{self.cfg.data_dir}/{self.cfg.dataset_name}/musique_ans_v1.0_dev_new_question_decomposition.jsonl"
        self.data = pd.read_json(fp, lines=True)


class TruthfulQADataset(BaseBenchmarkDataset):
    """
    Supports optional context augmentation (used in your Figure 10 "extra knowledge injection").
    """
    def load_data(self) -> None:
        fp = f"{self.cfg.data_dir}/{self.cfg.dataset_name}/mc_task.json"
        self.data = pd.read_json(fp)

        if self.cfg.context:
            # "Extra knowledge injection" (dataset-provided context).
            ctx = datasets.load_dataset("portkey/truthful_qa_context", split="train").to_pandas()
            ctx = ctx[["question", "context", "source"]]

            # clean obvious errors and constrain length
            ctx = ctx[~ctx["context"].str.lower().str.contains("error", na=False)]
            ctx = ctx[ctx["context"].str.len() < 2000]

            merged = pd.merge(self.data, ctx, on="question", how="inner")
            self.data = merged.reset_index(drop=True)


class MedMCQADataset(BaseBenchmarkDataset):
    def load_data(self) -> None:
        fp = f"{self.cfg.data_dir}/{self.cfg.dataset_name}/dev.json"
        self.data = pd.read_json(fp, lines=True)

    def filter_data(self) -> None:
        # only single-choice questions
        self.data = self.data[self.data["choice_type"] == "single"].reset_index(drop=True)


class ScalrDataset(BaseBenchmarkDataset):
    def load_data(self) -> None:
        fp = f"{self.cfg.data_dir}/{self.cfg.dataset_name}/test.jsonl"
        self.data = pd.read_json(fp, lines=True)


# ============================================================
# Registry + factory (clean extensibility)
# ============================================================

DATASET_REGISTRY: Dict[str, Type[BaseBenchmarkDataset]] = {
    "mmlu": MMLUDataset,
    "math": MathDataset,
    "chess": ChessDataset,
    "mquake": MQuakeDataset,
    "musique": MusiqueDataset,
    "truthfulqa": TruthfulQADataset,
    "medmcqa": MedMCQADataset,
    "scalr": ScalrDataset,
}


def get_dataset(dataset_name: str = "mmlu", n_samples: int = 50, data_dir: str = "data", context: bool = False, seed: int = 0, **kwargs) -> BaseBenchmarkDataset:
    """
    Backward-compatible entry point.
    kwargs allows dataset-specific params, e.g.:
      get_dataset("math", level_geq=3)
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(f"Dataset {dataset_name} not supported")

    cfg = DatasetConfig(dataset_name=dataset_name, n_samples=n_samples, data_dir=data_dir, seed=seed, context=context)
    cls = DATASET_REGISTRY[dataset_name]

    # allow special ctor args (e.g., MathDataset level_geq)
    return cls(cfg, **kwargs) if kwargs else cls(cfg)

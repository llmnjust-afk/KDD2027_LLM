"""Baseline interface — all baselines and our method implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from kg_scaffold.generation.hypothesis_gen import Hypothesis


@dataclass
class RunConfig:
    num_hypotheses: int = 10
    use_kg: bool = True
    use_completion: bool = True
    use_faithfulness: bool = True
    method_name: str = ""


class BaseMethod(ABC):
    """Every method (baseline or ours) takes a query and returns hypotheses."""

    name: str = "base"

    @abstractmethod
    def run(self, question: str, seed_entity: str,
            cfg: RunConfig | None = None) -> list[Hypothesis]:
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name}>"

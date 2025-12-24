from abc import ABC, abstractmethod
from typing import Self


class SimpleTokenizer(ABC):
    @abstractmethod
    def encode(self: Self, text: str) -> list[int]:
        """Takes in text and returns tokens."""
        raise NotImplementedError

    @abstractmethod
    def decode(self: Self, ids: list[int]) -> str:
        """Takes in tokens and returns text. decode(encode(s)) should equal s."""
        raise NotImplementedError

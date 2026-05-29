from __future__ import annotations

from collections.abc import Callable
from typing import Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> class registry used for models and benchmarks."""

    def __init__(self, kind: str) -> None:
        self._kind = kind
        self._items: dict[str, type[T]] = {}

    def register(self, name: str) -> Callable[[type[T]], type[T]]:
        def deco(cls: type[T]) -> type[T]:
            if name in self._items:
                raise ValueError(f"{self._kind} '{name}' already registered")
            self._items[name] = cls
            return cls

        return deco

    def get(self, name: str) -> type[T]:
        if name not in self._items:
            raise KeyError(f"Unknown {self._kind} '{name}'. Registered: {self.names()}")
        return self._items[name]

    def names(self) -> list[str]:
        return sorted(self._items)

from typing import Callable, Dict


class ModelRegistry:
    def __init__(self):
        self._models: Dict[str, Callable] = {}

    def register(self, name: str, handler: Callable):
        self._models[name] = handler

    def list_models(self):
        return list(self._models.keys())

    def get(self, name: str):
        return self._models.get(name)


registry = ModelRegistry()
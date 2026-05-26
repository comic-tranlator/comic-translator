import importlib
import importlib.util
from functools import partial
from typing import Any, Callable, Mapping

from .data import REGISTRY as DATA_REGISTRY
from .engine import REGISTRY as ENGINE_REGISTRY
from .modeling import REGISTRY as MODELING_REGISTRY
from .tasks import REGISTRY as TASKS_REGISTRY
from .util import deep_merge


class Factory:
    def __init__(self, registry: Mapping[str, Callable]) -> None:
        self._registry = registry

    def build(self, value: Any, **overrides) -> Any:
        if self._is_build_config(value):
            key, params = next(iter(value.items()))
            return self.construct(self._resolve(key), params, **overrides)

        if isinstance(value, str) and self._is_obj_name(value):
            return self.construct(self._resolve(value), {}, **overrides)

        if isinstance(value, list):
            return [self.build(v) for v in value]

        if isinstance(value, dict):
            return {k: self.build(v) for k, v in value.items()}

        return value

    def construct(self, target: Callable, params: dict, **overrides) -> Any:
        if overrides:
            params = deep_merge(params, overrides)

        as_partial = params.pop("partial", False)
        params = {k: self.build(v) for k, v in params.items()}

        return partial(target, **params) if as_partial else target(**params)

    def _resolve(self, name: str) -> Any:
        if name in self._registry:
            return self._registry[name]
        if "." in name:
            module_path, attr = name.rsplit(".", 1)
            if importlib.util.find_spec(module_path):
                return getattr(importlib.import_module(module_path), attr)
        raise KeyError(f"Unknown component: {name!r}")

    def _is_build_config(self, value: Any) -> bool:
        if not (isinstance(value, dict) and len(value) == 1):
            return False
        name = next(iter(value))

        return self._is_obj_name(name)

    def _is_obj_name(self, name: str) -> bool:
        if name in self._registry:
            return True

        if "." in name:
            module_name = name.rsplit(".", 1)[0]
            try:
                return importlib.util.find_spec(module_name) is not None
            except (ModuleNotFoundError, ValueError, AttributeError):
                return False

        return False


factory = Factory(DATA_REGISTRY | MODELING_REGISTRY | ENGINE_REGISTRY | TASKS_REGISTRY)

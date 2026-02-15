import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional


def read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_path(base_dir: str, p: str) -> str:
    if not p:
        return p
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


@dataclass
class AIConfig:
    path: str
    data: Dict[str, Any]

    @staticmethod
    def load(path: str) -> "AIConfig":
        return AIConfig(path=path, data=read_json(path))

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


@dataclass
class GUIConfig:
    path: str
    data: Dict[str, Any]

    @staticmethod
    def load(path: str) -> "GUIConfig":
        return GUIConfig(path=path, data=read_json(path))

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)


@dataclass
class ThemeConfig:
    path: str
    data: Dict[str, Any]

    @staticmethod
    def load(path: str) -> "ThemeConfig":
        return ThemeConfig(path=path, data=read_json(path))

    def qss_text(self) -> str:
        # supports { "QSS": "..." } or { "QSS_LINES": ["..", ".."] }
        if "QSS" in self.data and isinstance(self.data["QSS"], str):
            return self.data["QSS"]
        lines = self.data.get("QSS_LINES", [])
        if isinstance(lines, list):
            return "\n".join([str(x) for x in lines])
        return ""

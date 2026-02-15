import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from AI_config import AIConfig, GUIConfig, ThemeConfig, resolve_path


@dataclass
class AISettings:
    base_dir: str
    ai: AIConfig
    gui: GUIConfig
    theme: ThemeConfig

    @classmethod
    def load(
        cls,
        ai_path: str = "Config/AI_Config.json",
        gui_path: str = "Config/GUI_Config.json",
        theme_path: str = "Config/Theme_QSS.json",
    ) -> "AISettings":
        base_dir = os.path.dirname(os.path.abspath(__file__))

        ai_abs = resolve_path(base_dir, ai_path)
        gui_abs = resolve_path(base_dir, gui_path)
        theme_abs = resolve_path(base_dir, theme_path)

        ai = AIConfig.load(ai_abs)
        gui = GUIConfig.load(gui_abs)
        theme = ThemeConfig.load(theme_abs)

        return cls(base_dir=base_dir, ai=ai, gui=gui, theme=theme)

    # -------- AI getters --------
    def get_ai(self, key: str, default: Any = None) -> Any:
        return self.ai.get(key, default)

    def get_gui(self, key: str, default: Any = None) -> Any:
        return self.gui.get(key, default)

    # -------- Derived paths --------
    @property
    def storage_dir(self) -> str:
        return resolve_path(self.base_dir, self.get_ai("STORAGE_DIR", "./History"))

    @property
    def chats_dir(self) -> str:
        return os.path.join(self.storage_dir, self.get_ai("CHATS_SUBDIR", "Chats"))

    @property
    def files_dir(self) -> str:
        return os.path.join(self.storage_dir, self.get_ai("FILES_SUBDIR", "Files"))

    def ensure_storage(self) -> None:
        os.makedirs(self.storage_dir, exist_ok=True)
        os.makedirs(self.chats_dir, exist_ok=True)
        os.makedirs(self.files_dir, exist_ok=True)

    # -------- Prompts --------
    @property
    def system_prompt_default(self) -> str:
        return str(self.get_ai("SYSTEM_PROMPT_DEFAULT", "You are a helpful assistant."))

    @property
    def welcome_message(self) -> str:
        return str(self.get_gui("WELCOME_MESSAGE", "")).strip()

    # -------- Model / generation --------
    @property
    def model_id(self) -> str:
        return str(self.get_ai("MODEL_ID", ""))

    @property
    def local_dir(self) -> str:
        return resolve_path(self.base_dir, str(self.get_ai("LOCAL_DIR", "")))

    @property
    def max_tokens(self) -> int:
        return int(self.get_ai("MAX_TOKENS", 4096))

    @property
    def temperature(self) -> float:
        return float(self.get_ai("TEMPERATURE", 0.3))

    @property
    def max_history_messages(self) -> int:
        return int(self.get_ai("MAX_HISTORY_MESSAGES", 10))

    # -------- Theme --------
    def qss(self) -> str:
        return self.theme.qss_text()

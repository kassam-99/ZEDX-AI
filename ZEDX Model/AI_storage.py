import json
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


def _parse_time_any(v: Any) -> float:
    """
    Supports:
      - float/int unix timestamps
      - "1700000000.0" strings
      - ISO strings "2026-02-15T14:24:07"
    Returns unix timestamp float.
    """
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        s = v.strip()
        if not s:
            return 0.0
        # numeric string?
        try:
            return float(s)
        except Exception:
            pass
        # ISO?
        try:
            dt = datetime.fromisoformat(s)
            return dt.timestamp()
        except Exception:
            return 0.0
    return 0.0


def _read_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, data: Dict[str, Any]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


class ChatManager:
    def __init__(self, settings):
        self.cfg = settings
        self.cfg.ensure_storage()

        self.chats_dir = self.cfg.chats_dir
        self.files_dir = self.cfg.files_dir

    def _chat_path(self, chat_id: str) -> str:
        return os.path.join(self.chats_dir, f"{chat_id}.json")

    def _chat_files_dir(self, chat_id: str) -> str:
        p = os.path.join(self.files_dir, chat_id)
        os.makedirs(p, exist_ok=True)
        return p

    def _load_chat(self, chat_id: str) -> Dict[str, Any]:
        path = self._chat_path(chat_id)
        if not os.path.exists(path):
            # create minimal if missing
            data = {
                "id": chat_id,
                "title": "New Chat",
                "pinned": False,
                "persona": self.cfg.system_prompt_default,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
                "messages": []
            }
            _write_json(path, data)
            return data
        return _read_json(path)

    def _save_chat(self, chat_id: str, data: Dict[str, Any]) -> None:
        data["updated_at"] = _now_iso()
        _write_json(self._chat_path(chat_id), data)

    # ---------- Public API used by GUI ----------
    def list_chats(self) -> List[Dict[str, Any]]:
        out = []
        for fn in os.listdir(self.chats_dir):
            if not fn.endswith(".json"):
                continue
            path = os.path.join(self.chats_dir, fn)
            try:
                data = _read_json(path)
            except Exception:
                continue

            chat_id = data.get("id") or fn.replace(".json", "")
            title = data.get("title", "New Chat")
            pinned = bool(data.get("pinned", False))

            updated_raw = data.get("updated_at", data.get("updated", 0.0))
            updated_ts = _parse_time_any(updated_raw)

            # preview = last non-system msg
            preview = ""
            msgs = data.get("messages", []) or []
            for m in reversed(msgs):
                if m.get("role") in ("user", "assistant"):
                    preview = str(m.get("content", "")).strip()
                    break
            preview = preview.replace("\n", " ")
            if len(preview) > 80:
                preview = preview[:80] + "…"

            # Rich search text: title + persona + all user/assistant messages.
            searchable_parts = [str(title), str(data.get("persona", ""))]
            for m in msgs:
                if m.get("role") in ("user", "assistant"):
                    searchable_parts.append(str(m.get("content", "")))
            search_text = " ".join(searchable_parts).lower()

            out.append({
                "id": chat_id,
                "title": title,
                "preview": preview,
                "search_text": search_text,
                "pinned": pinned,
                "updated_at": updated_ts,
            })

        # Pinned chats always stay at the top, then sort by most recently updated.
        out.sort(key=lambda x: (0 if x.get("pinned", False) else 1, -x.get("updated_at", 0.0)))
        return out

    def create_chat(self, title: str = "New Chat") -> str:
        chat_id = str(uuid.uuid4())
        data = {
            "id": chat_id,
            "title": title or "New Chat",
            "pinned": False,
            "persona": self.cfg.system_prompt_default,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
            "messages": []
        }
        _write_json(self._chat_path(chat_id), data)

        os.makedirs(os.path.join(self.files_dir, chat_id), exist_ok=True)
        return chat_id

    def rename_chat(self, chat_id: str, new_title: str) -> None:
        data = self._load_chat(chat_id)
        data["title"] = new_title.strip() or data.get("title", "New Chat")
        self._save_chat(chat_id, data)

    def delete_chat(self, chat_id: str) -> None:
        p = self._chat_path(chat_id)
        if os.path.exists(p):
            os.remove(p)
        folder = os.path.join(self.files_dir, chat_id)
        if os.path.isdir(folder):
            shutil.rmtree(folder, ignore_errors=True)

    def add_files(self, chat_id: str, source_paths: List[str]) -> List[Dict[str, Any]]:
        target_dir = self._chat_files_dir(chat_id)
        out: List[Dict[str, Any]] = []
        for src in source_paths:
            if not src or not os.path.isfile(src):
                continue
            base = os.path.basename(src)
            name, ext = os.path.splitext(base)
            candidate = base
            i = 1
            while os.path.exists(os.path.join(target_dir, candidate)):
                candidate = f"{name}_{i}{ext}"
                i += 1
            dst = os.path.join(target_dir, candidate)
            shutil.copy2(src, dst)
            out.append({"name": candidate, "path": dst})
        return out

    def list_files(self, chat_id: str) -> List[Dict[str, Any]]:
        target_dir = self._chat_files_dir(chat_id)
        out: List[Dict[str, Any]] = []
        image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
        for fn in os.listdir(target_dir):
            p = os.path.join(target_dir, fn)
            if not os.path.isfile(p):
                continue
            ext = os.path.splitext(fn)[1].lower()
            try:
                mtime = os.path.getmtime(p)
            except Exception:
                mtime = 0.0
            out.append({
                "name": fn,
                "path": p,
                "ext": ext,
                "is_image": ext in image_exts,
                "mtime": mtime,
            })
        out.sort(key=lambda x: x.get("mtime", 0.0), reverse=True)
        return out

    def build_file_context(self, chat_id: str, max_chars: int, preferred_name: str = "") -> str:
        files = self.list_files(chat_id)
        if not files or max_chars <= 0:
            return ""

        preferred = str(preferred_name or "").strip().lower()
        if preferred:
            files.sort(key=lambda x: (0 if str(x.get("name", "")).lower() == preferred else 1, -x.get("mtime", 0.0)))

        allowed = self.cfg.get_ai("ALLOWED_TEXT_EXTENSIONS", [])
        allowed_exts = {str(x).lower() for x in allowed} if isinstance(allowed, list) else set()
        remaining = int(max_chars)
        parts: List[str] = []

        for f in files:
            if remaining <= 0:
                break

            name = str(f.get("name", "file"))
            ext = str(f.get("ext", "")).lower()
            path = str(f.get("path", ""))

            if f.get("is_image"):
                line = f"[IMAGE FILE] {name}"
                if len(line) + 1 <= remaining:
                    parts.append(line)
                    remaining -= len(line) + 1
                continue

            if ext not in allowed_exts or not path:
                continue

            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as fp:
                    txt = fp.read(max(0, remaining - 64))
            except Exception:
                continue

            if not txt.strip():
                continue

            block = f"[FILE] {name}\n{txt.strip()}"
            if len(block) > remaining:
                block = block[:remaining]
            parts.append(block)
            remaining -= len(block) + 1

        return "\n\n".join(parts).strip()

    def is_pinned(self, chat_id: str) -> bool:
        data = self._load_chat(chat_id)
        return bool(data.get("pinned", False))

    def set_pinned(self, chat_id: str, pinned: bool) -> None:
        data = self._load_chat(chat_id)
        data["pinned"] = bool(pinned)
        self._save_chat(chat_id, data)

    def clear_chat(self, chat_id: str) -> None:
        data = self._load_chat(chat_id)
        data["messages"] = []
        self._save_chat(chat_id, data)

    def get_messages(self, chat_id: str) -> List[Dict[str, str]]:
        data = self._load_chat(chat_id)
        msgs = data.get("messages", []) or []
        # normalize
        out = []
        for m in msgs:
            role = str(m.get("role", "")).strip()
            content = str(m.get("content", ""))
            if role in ("user", "assistant"):
                out.append({"role": role, "content": content})
        return out

    def add_message(self, chat_id: str, role: str, content: str) -> None:
        data = self._load_chat(chat_id)
        msgs = data.get("messages", []) or []
        msgs.append({
            "role": role,
            "content": content,
            "ts": _now_iso()
        })
        data["messages"] = msgs
        self._save_chat(chat_id, data)

    def get_persona(self, chat_id: str) -> str:
        data = self._load_chat(chat_id)
        # support older keys
        p = data.get("persona", data.get("persona_prompt", ""))
        p = str(p or "").strip()
        return p if p else self.cfg.system_prompt_default

    def set_persona(self, chat_id: str, persona: str) -> None:
        data = self._load_chat(chat_id)
        data["persona"] = str(persona or "").strip()
        self._save_chat(chat_id, data)

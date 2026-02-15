import os
import re
import sys
from PySide6 import QtCore, QtGui, QtWidgets

from AI_Settings import AISettings
from AI_storage import ChatManager
from AI_model import generate_code, get_model_name, load_ai_model, get_model_runtime_stats
from hardware_monitor import get_system_stats


# ---------- Helper widgets ----------
class CodeSyntaxHighlighter(QtGui.QSyntaxHighlighter):
    def __init__(self, document, language: str):
        super().__init__(document)
        self.language = self._normalize_language(language)
        self.rules = []
        self.comment_patterns = []
        self.comment_start = None
        self.comment_end = None
        self.fmt_comment = self._fmt("#8B949E", italic=True)
        self._build_rules()

    @staticmethod
    def _normalize_language(language: str) -> str:
        lang = (language or "").strip().lower()
        aliases = {
            "py": "python",
            "python3": "python",
            "c++": "cpp",
            "cc": "cpp",
            "js": "javascript",
            "ts": "typescript",
            "sh": "bash",
            "shell": "bash",
        }
        return aliases.get(lang, lang)

    @staticmethod
    def _fmt(color: str, bold: bool = False, italic: bool = False) -> QtGui.QTextCharFormat:
        f = QtGui.QTextCharFormat()
        f.setForeground(QtGui.QColor(color))
        if bold:
            f.setFontWeight(QtGui.QFont.Bold)
        if italic:
            f.setFontItalic(True)
        return f

    def _add_word_rules(self, words, fmt: QtGui.QTextCharFormat):
        for w in words:
            pattern = QtCore.QRegularExpression(rf"\b{re.escape(w)}\b")
            self.rules.append((pattern, fmt))

    def _build_rules(self):
        fmt_keyword = self._fmt("#FF7B72", bold=True)
        fmt_string = self._fmt("#A5D6FF")
        fmt_number = self._fmt("#79C0FF")
        fmt_function = self._fmt("#D2A8FF")
        fmt_type = self._fmt("#FFA657")
        fmt_preproc = self._fmt("#79C0FF")
        fmt_decorator = self._fmt("#D2A8FF")

        # Strings and numbers
        self.rules.append((QtCore.QRegularExpression(r'"([^"\\]|\\.)*"'), fmt_string))
        self.rules.append((QtCore.QRegularExpression(r"'([^'\\]|\\.)*'"), fmt_string))
        self.rules.append((QtCore.QRegularExpression(r"`([^`\\]|\\.)*`"), fmt_string))
        self.rules.append((QtCore.QRegularExpression(r"\b\d+(\.\d+)?\b"), fmt_number))

        # Function calls and decorators/annotations
        self.rules.append((QtCore.QRegularExpression(r"\b[A-Za-z_][A-Za-z0-9_]*(?=\()"), fmt_function))
        self.rules.append((QtCore.QRegularExpression(r"@[A-Za-z_][A-Za-z0-9_]*"), fmt_decorator))

        if self.language == "python":
            keywords = [
                "and", "as", "assert", "async", "await", "break", "class", "continue",
                "def", "del", "elif", "else", "except", "False", "finally", "for",
                "from", "global", "if", "import", "in", "is", "lambda", "None",
                "nonlocal", "not", "or", "pass", "raise", "return", "True", "try",
                "while", "with", "yield",
            ]
            types_and_builtins = [
                "dict", "list", "set", "tuple", "str", "int", "float", "bool",
                "bytes", "object", "print", "len", "range", "enumerate",
            ]
            self._add_word_rules(keywords, fmt_keyword)
            self._add_word_rules(types_and_builtins, fmt_type)
            self.comment_patterns.append(QtCore.QRegularExpression(r"#.*$"))
        else:
            c_like_keywords = [
                "break", "case", "catch", "class", "const", "continue", "default",
                "delete", "do", "else", "enum", "extern", "false", "final",
                "finally", "for", "if", "import", "inline", "interface", "new",
                "nullptr", "private", "protected", "public", "return", "static",
                "struct", "super", "switch", "this", "throw", "true", "try",
                "typedef", "typename", "using", "var", "virtual", "void", "while",
            ]
            c_like_types = [
                "int", "long", "short", "float", "double", "char", "bool", "byte",
                "string", "String", "auto", "size_t", "usize", "i32", "u32", "i64",
                "u64", "Vec", "Result", "Option",
            ]
            self._add_word_rules(c_like_keywords, fmt_keyword)
            self._add_word_rules(c_like_types, fmt_type)
            self.rules.append((QtCore.QRegularExpression(r"^\s*#\w+.*$"), fmt_preproc))
            self.comment_patterns.append(QtCore.QRegularExpression(r"//.*$"))
            self.comment_start = QtCore.QRegularExpression(r"/\*")
            self.comment_end = QtCore.QRegularExpression(r"\*/")

        # Common constants
        self.rules.append((QtCore.QRegularExpression(r"\b[A-Z_]{2,}\b"), fmt_preproc))

    def highlightBlock(self, text: str):
        for pattern, fmt in self.rules:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), fmt)

        for pattern in self.comment_patterns:
            it = pattern.globalMatch(text)
            while it.hasNext():
                m = it.next()
                self.setFormat(m.capturedStart(), m.capturedLength(), self.fmt_comment)

        if self.comment_start is None or self.comment_end is None:
            return

        self.setCurrentBlockState(0)
        start = -1
        if self.previousBlockState() != 1:
            sm = self.comment_start.match(text)
            if sm.hasMatch():
                start = sm.capturedStart()
        else:
            start = 0

        while start >= 0:
            em = self.comment_end.match(text, start + 2)
            if em.hasMatch():
                end = em.capturedEnd()
                self.setFormat(start, end - start, self.fmt_comment)
                sm = self.comment_start.match(text, end)
                start = sm.capturedStart() if sm.hasMatch() else -1
            else:
                self.setCurrentBlockState(1)
                self.setFormat(start, len(text) - start, self.fmt_comment)
                break


class CodeBlock(QtWidgets.QFrame):
    def __init__(self, language: str, code: str, parent=None):
        super().__init__(parent)
        self.setObjectName("CodePanel")
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(language.strip() or "code")
        lbl.setStyleSheet("color: #8B949E; font-weight: 800;")
        top.addWidget(lbl)
        top.addStretch(1)

        btn = QtWidgets.QPushButton("Copy")
        btn.setObjectName("BtnCopy")
        btn.clicked.connect(lambda: QtWidgets.QApplication.clipboard().setText(code))
        top.addWidget(btn)

        v.addLayout(top)

        edit = QtWidgets.QPlainTextEdit()
        edit.setReadOnly(True)
        edit.setPlainText(code)
        edit.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        edit.setStyleSheet(
            "QPlainTextEdit { color: #C9D1D9; background: transparent; border: none; }"
        )
        font = QtGui.QFont("JetBrains Mono")
        font.setStyleHint(QtGui.QFont.Monospace)
        font.setPointSize(11)
        edit.setFont(font)
        edit.setTabStopDistance(QtGui.QFontMetricsF(font).horizontalAdvance(" ") * 4)
        edit.setMinimumHeight(60)
        self._highlighter = CodeSyntaxHighlighter(edit.document(), language)

        v.addWidget(edit)


class Bubble(QtWidgets.QFrame):
    def __init__(self, role: str, text: str, max_width: int):
        super().__init__()
        self.role = role
        self.setObjectName("BubbleUser" if role == "user" else "BubbleAssistant")

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        content = QtWidgets.QVBoxLayout()
        content.setContentsMargins(12, 10, 12, 10)
        content.setSpacing(10)

        # render assistant with code blocks
        if role == "assistant":
            parts = self._split_code_blocks(text)
            for typ, payload in parts:
                if typ == "text":
                    lbl = QtWidgets.QLabel(payload)
                    lbl.setStyleSheet("color: #C9D1D9;")
                    lbl.setWordWrap(True)
                    lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
                    lbl.setMaximumWidth(max_width)
                    content.addWidget(lbl)
                else:
                    lang, code = payload
                    block = CodeBlock(lang, code)
                    block.setMaximumWidth(max_width)
                    content.addWidget(block)
        else:
            lbl = QtWidgets.QLabel(text)
            lbl.setStyleSheet("color: #F0F6FC;")
            lbl.setWordWrap(True)
            lbl.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
            lbl.setMaximumWidth(max_width)
            content.addWidget(lbl)

        wrap = QtWidgets.QWidget()
        wrap.setLayout(content)
        outer.addWidget(wrap)

    def _split_code_blocks(self, text: str):
        """
        Returns list of ("text", str) or ("code", (lang, code))
        """
        res = []
        s = text
        while "```" in s:
            before, rest = s.split("```", 1)
            if before.strip():
                res.append(("text", before.strip()))
            if "```" not in rest:
                res.append(("text", rest.strip()))
                return res
            fence, after = rest.split("```", 1)
            lines = fence.splitlines()
            lang = lines[0].strip() if lines else ""
            code = "\n".join(lines[1:]) if len(lines) > 1 else ""
            res.append(("code", (lang, code)))
            s = after
        if s.strip():
            res.append(("text", s.strip()))
        return res


# ---------- Worker ----------
class GenerateWorker(QtCore.QThread):
    done = QtCore.Signal(str)
    fail = QtCore.Signal(str)

    def __init__(self, history, system_prompt, settings):
        super().__init__()
        self.history = history
        self.system_prompt = system_prompt
        self.settings = settings

    def run(self):
        try:
            reply = generate_code(self.history, self.system_prompt, self.settings)
            self.done.emit(reply)
        except Exception as e:
            self.fail.emit(str(e))


class ModelWarmupWorker(QtCore.QThread):
    ready = QtCore.Signal()
    fail = QtCore.Signal(str)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def run(self):
        try:
            load_ai_model(self.settings)
            self.ready.emit()
        except Exception as e:
            self.fail.emit(str(e))


class MonitorDialog(QtWidgets.QDialog):
    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings
        self.setObjectName("Modal")
        self.setWindowTitle("Hardware + AI Model Monitor")
        self.resize(920, 660)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(18, 18, 18, 18)
        root.setSpacing(12)

        title = QtWidgets.QLabel("System Monitor")
        title.setObjectName("ModalTitle")
        sub = QtWidgets.QLabel(
            "Live CPU, RAM, storage, VRAM, process usage, and current model runtime details."
        )
        sub.setObjectName("ModalSub")

        root.addWidget(title)
        root.addWidget(sub)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(10)

        self.cpu_val, self.cpu_bar = self._make_meter("CPU Usage")
        self.ram_val, self.ram_bar = self._make_meter("RAM Usage")
        self.disk_val, self.disk_bar = self._make_meter("Storage Usage")
        self.vram_val, self.vram_bar = self._make_meter("VRAM Usage")

        grid.addWidget(self._meter_card("CPU Usage", self.cpu_val, self.cpu_bar), 0, 0)
        grid.addWidget(self._meter_card("RAM Usage", self.ram_val, self.ram_bar), 0, 1)
        grid.addWidget(self._meter_card("Storage Usage", self.disk_val, self.disk_bar), 1, 0)
        grid.addWidget(self._meter_card("VRAM Usage", self.vram_val, self.vram_bar), 1, 1)
        root.addLayout(grid)

        self.model_box = QtWidgets.QFrame()
        self.model_box.setObjectName("MonitorCard")
        mb = QtWidgets.QVBoxLayout(self.model_box)
        mb.setContentsMargins(12, 12, 12, 12)
        mb.setSpacing(8)

        model_title = QtWidgets.QLabel("Current AI Model Usage")
        model_title.setObjectName("MonitorTitle")
        self.model_text = QtWidgets.QPlainTextEdit()
        self.model_text.setObjectName("MonitorText")
        self.model_text.setReadOnly(True)
        self.model_text.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        mono = QtGui.QFont("JetBrains Mono")
        mono.setStyleHint(QtGui.QFont.Monospace)
        mono.setPointSize(10)
        self.model_text.setFont(mono)
        self.model_text.setMinimumHeight(260)

        mb.addWidget(model_title)
        mb.addWidget(self.model_text, 1)
        root.addWidget(self.model_box, 1)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)
        self.btn_refresh = QtWidgets.QPushButton("Refresh")
        self.btn_refresh.setObjectName("BtnMonitor")
        self.btn_close = QtWidgets.QPushButton("Close")
        self.btn_close.setObjectName("BtnModalDefault")
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_close.clicked.connect(self.accept)
        row.addWidget(self.btn_refresh)
        row.addWidget(self.btn_close)
        root.addLayout(row)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()
        self.refresh()

    def _make_meter(self, _title: str):
        value = QtWidgets.QLabel("0.0%")
        value.setObjectName("MonitorValue")
        bar = QtWidgets.QProgressBar()
        bar.setObjectName("MonitorBar")
        bar.setRange(0, 100)
        bar.setValue(0)
        bar.setTextVisible(False)
        return value, bar

    def _meter_card(self, title: str, value: QtWidgets.QLabel, bar: QtWidgets.QProgressBar):
        card = QtWidgets.QFrame()
        card.setObjectName("MonitorCard")
        v = QtWidgets.QVBoxLayout(card)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(8)

        lbl = QtWidgets.QLabel(title)
        lbl.setObjectName("MonitorTitle")
        v.addWidget(lbl)
        v.addWidget(value)
        v.addWidget(bar)
        return card

    @staticmethod
    def _fmt_used_total(used: float, total: float, suffix: str = "GB") -> str:
        return f"{used:.2f}/{total:.2f} {suffix}" if total > 0 else f"{used:.2f} {suffix}"

    def refresh(self):
        stats = get_system_stats(self.settings.storage_dir)
        cpu = stats.get("cpu", {})
        ram = stats.get("ram", {})
        disk = stats.get("storage", {})
        vram = stats.get("vram", {})
        proc = stats.get("process", {})

        cpu_pct = float(cpu.get("percent", 0.0))
        ram_pct = float(ram.get("percent", 0.0))
        disk_pct = float(disk.get("percent", 0.0))
        vram_pct = float(vram.get("percent", 0.0))

        self.cpu_val.setText(
            f"{cpu_pct:.1f}%  ({int(cpu.get('cores_logical', 0))} threads)"
        )
        self.ram_val.setText(
            f"{ram_pct:.1f}%  ({self._fmt_used_total(float(ram.get('used_gb', 0.0)), float(ram.get('total_gb', 0.0)))})"
        )
        self.disk_val.setText(
            f"{disk_pct:.1f}%  ({self._fmt_used_total(float(disk.get('used_gb', 0.0)), float(disk.get('total_gb', 0.0)))})"
        )
        self.vram_val.setText(
            f"{vram_pct:.1f}%  ({self._fmt_used_total(float(vram.get('used_gb', 0.0)), float(vram.get('total_gb', 0.0)))})"
        )

        self.cpu_bar.setValue(int(max(0.0, min(100.0, cpu_pct))))
        self.ram_bar.setValue(int(max(0.0, min(100.0, ram_pct))))
        self.disk_bar.setValue(int(max(0.0, min(100.0, disk_pct))))
        self.vram_bar.setValue(int(max(0.0, min(100.0, vram_pct))))

        m = get_model_runtime_stats(self.settings)
        loaded = "Yes" if m.get("loaded") else "No"
        model_path = str(m.get("model_path", "") or get_model_name(self.settings))
        lines = [
            f"Loaded: {loaded}",
            f"Model Path: {model_path}",
            f"CUDA Available: {m.get('cuda_available')}",
            f"GPU: {m.get('gpu_name', '') or 'N/A'}",
            f"GPU Allocated: {float(m.get('gpu_allocated_gb', 0.0)):.2f} GB",
            f"GPU Reserved: {float(m.get('gpu_reserved_gb', 0.0)):.2f} GB",
            f"GPU Total: {float(m.get('gpu_total_gb', 0.0)):.2f} GB",
            "",
            f"Device: {m.get('device', 'N/A')}",
            f"DType: {m.get('dtype', 'N/A')}",
            f"Parameters: {int(m.get('parameters', 0)):,}",
            f"Trainable Parameters: {int(m.get('trainable_parameters', 0)):,}",
            f"Tokenizer Vocab: {int(m.get('tokenizer_vocab', 0)):,}",
            f"Parameter Memory: {float(m.get('param_mem_gb', 0.0)):.2f} GB",
            f"Buffer Memory: {float(m.get('buffer_mem_gb', 0.0)):.2f} GB",
            "",
            f"Process RSS: {float(proc.get('rss_gb', 0.0)):.2f} GB",
            f"Process Threads: {int(proc.get('threads', 0))}",
            f"PID: {int(proc.get('pid', 0))}",
            "",
            f"Generation MAX_TOKENS: {int(self.settings.max_tokens)}",
            f"Generation TEMPERATURE: {float(self.settings.temperature):.2f}",
            f"History Window: {int(self.settings.max_history_messages)} messages",
        ]
        self.model_text.setPlainText("\n".join(lines))

    def closeEvent(self, event):
        if self.timer.isActive():
            self.timer.stop()
        super().closeEvent(event)


# ---------- Main Window ----------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setObjectName("MainWindow")

        self.settings = AISettings.load()
        self.settings.ensure_storage()
        self.chat_mgr = ChatManager(self.settings)

        self.current_chat_id = None
        self.worker = None
        self.model_worker = None
        self.model_ready = False
        self._suppress_item_changed = False

        self.setWindowTitle(self.settings.get_gui("APP_TITLE", "NeuralIDE"))
        self.resize(
            int(self.settings.get_gui("WINDOW_WIDTH", 1300)),
            int(self.settings.get_gui("WINDOW_HEIGHT", 820)),
        )

        self._build_ui()
        self._apply_theme()
        self._refresh_chat_list()
        self._ensure_selected_chat()
        self._start_model_warmup()

    def _apply_theme(self):
        qss = self.settings.qss()
        if not isinstance(qss, str):
            qss = str(qss)
        QtWidgets.QApplication.instance().setStyleSheet(qss)

    def _build_ui(self):
        cw = QtWidgets.QWidget()
        root = QtWidgets.QHBoxLayout(cw)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(14)

        # Sidebar
        self.sidebar = QtWidgets.QFrame()
        self.sidebar.setObjectName("Sidebar")
        side = QtWidgets.QVBoxLayout(self.sidebar)
        side.setContentsMargins(14, 14, 14, 14)
        side.setSpacing(10)

        self.btn_new = QtWidgets.QPushButton("+  New Chat")
        self.btn_new.setObjectName("BtnNewChat")
        self.btn_new.clicked.connect(self.on_new_chat)

        self.search = QtWidgets.QLineEdit()
        self.search.setObjectName("SearchBox")
        self.search.setPlaceholderText("Search chats and messages...")
        self.search.textChanged.connect(self._refresh_chat_list)

        lbl = QtWidgets.QLabel("Chats")
        lbl.setStyleSheet("font-weight: 800; color: #8B949E; letter-spacing: 0.4px;")

        self.chat_list = QtWidgets.QListWidget()
        self.chat_list.setObjectName("ChatList")
        self.chat_list.currentItemChanged.connect(self.on_select_chat)
        self.chat_list.itemChanged.connect(self.on_chat_item_edited)
        self.chat_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.chat_list.customContextMenuRequested.connect(self.on_chat_menu)

        files_lbl = QtWidgets.QLabel("Files")
        files_lbl.setStyleSheet("font-weight: 800; color: #8B949E; letter-spacing: 0.4px;")
        self.file_list = QtWidgets.QListWidget()
        self.file_list.setObjectName("FileList")
        self.file_list.setMaximumHeight(170)
        self.file_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.file_list.customContextMenuRequested.connect(self.on_file_menu)

        side.addWidget(self.btn_new)
        side.addWidget(self.search)
        side.addWidget(lbl)
        side.addWidget(self.chat_list, 1)
        side.addWidget(files_lbl)
        side.addWidget(self.file_list)

        # Main Panel
        self.main = QtWidgets.QFrame()
        self.main.setObjectName("MainPanel")
        main = QtWidgets.QVBoxLayout(self.main)
        main.setContentsMargins(18, 18, 18, 18)
        main.setSpacing(12)

        # Top Bar
        top = QtWidgets.QHBoxLayout()
        title_box = QtWidgets.QVBoxLayout()
        self.lbl_title = QtWidgets.QLabel(self.settings.get_gui("APP_TITLE", "NeuralIDE"))
        self.lbl_title.setStyleSheet("font-size: 20px; font-weight: 900; color: #F0F6FC;")
        self.lbl_status = QtWidgets.QLabel(f"Ready · {get_model_name(self.settings)}")
        self.lbl_status.setObjectName("StatusBar")
        title_box.addWidget(self.lbl_title)
        title_box.addWidget(self.lbl_status)
        top.addLayout(title_box)
        top.addStretch(1)

        self.btn_persona = QtWidgets.QPushButton("Persona")
        self.btn_persona.setObjectName("BtnPersona")
        self.btn_persona.clicked.connect(self.on_persona)

        self.btn_monitor = QtWidgets.QPushButton("Monitor")
        self.btn_monitor.setObjectName("BtnMonitor")
        self.btn_monitor.clicked.connect(self.on_monitor)

        self.btn_clear = QtWidgets.QPushButton("Clear")
        self.btn_clear.setObjectName("BtnClear")
        self.btn_clear.clicked.connect(self.on_clear)

        top.addWidget(self.btn_monitor)
        top.addWidget(self.btn_persona)
        top.addWidget(self.btn_clear)
        main.addLayout(top)

        # Messages scroll area
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setObjectName("ScrollArea")
        self.scroll.setWidgetResizable(True)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.msg_viewport = QtWidgets.QWidget()
        self.msg_viewport.setObjectName("MessagesViewport")
        self.msg_layout = QtWidgets.QVBoxLayout(self.msg_viewport)
        self.msg_layout.setContentsMargins(16, 16, 16, 16)
        self.msg_layout.setSpacing(12)
        self.msg_layout.addStretch(1)

        self.scroll.setWidget(self.msg_viewport)
        main.addWidget(self.scroll, 1)

        # Input row
        row = QtWidgets.QHBoxLayout()
        row.setSpacing(10)

        self.input = QtWidgets.QLineEdit()
        self.input.setObjectName("InputBox")
        self.input.setPlaceholderText("Ask anything...")
        self.input.returnPressed.connect(self.on_send)

        self.btn_upload = QtWidgets.QPushButton("Upload")
        self.btn_upload.setObjectName("BtnUpload")
        self.btn_upload.clicked.connect(self.on_upload_files)

        self.btn_send = QtWidgets.QPushButton("➤")
        self.btn_send.setObjectName("BtnSend")
        self.btn_send.clicked.connect(self.on_send)

        row.addWidget(self.btn_upload)
        row.addWidget(self.input, 1)
        row.addWidget(self.btn_send)

        main.addLayout(row)

        # Bottom status
        self.status = QtWidgets.QLabel("Ready")
        self.status.setObjectName("StatusBar")
        main.addWidget(self.status)

        root.addWidget(self.sidebar)
        root.addWidget(self.main, 1)

        self.setCentralWidget(cw)

    def _start_model_warmup(self):
        self.model_ready = False
        self.input.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.btn_upload.setEnabled(False)
        self.status.setText("Loading model...")
        self.lbl_status.setText(f"Loading model · {get_model_name(self.settings)}")

        self.model_worker = ModelWarmupWorker(self.settings)
        self.model_worker.ready.connect(self._on_model_ready)
        self.model_worker.fail.connect(self._on_model_error)
        self.model_worker.start()

    def _on_model_ready(self):
        self.model_ready = True
        self.input.setEnabled(True)
        self.btn_send.setEnabled(True)
        self.btn_upload.setEnabled(True)
        self.status.setText("Ready")
        self.lbl_status.setText(f"Ready · {get_model_name(self.settings)}")

    def _on_model_error(self, msg: str):
        self.model_ready = False
        self.input.setEnabled(False)
        self.btn_send.setEnabled(False)
        self.btn_upload.setEnabled(False)
        self.status.setText("Model load failed")
        self.lbl_status.setText("Model load failed")
        QtWidgets.QMessageBox.critical(self, "Model Load Error", msg)

    # ---------- scrolling (smart) ----------
    def _is_near_bottom(self) -> bool:
        mode = str(self.settings.get_gui("AUTO_SCROLL_MODE", "smart")).lower()
        if mode == "always":
            return True
        sb = self.scroll.verticalScrollBar()
        near = int(self.settings.get_gui("SCROLL_NEAR_BOTTOM_PX", 60))
        return (sb.maximum() - sb.value()) <= near

    def _scroll_to_bottom(self):
        sb = self.scroll.verticalScrollBar()
        sb.setValue(sb.maximum())
        QtCore.QTimer.singleShot(0, lambda: sb.setValue(sb.maximum()))
        QtCore.QTimer.singleShot(30, lambda: sb.setValue(sb.maximum()))

    # ---------- Chat list ----------
    def _refresh_chat_list(self):
        q = self.search.text().strip().lower()
        chats = self.chat_mgr.list_chats()

        self.chat_list.blockSignals(True)
        self.chat_list.clear()

        for c in chats:
            title = c["title"]
            search_text = c.get("search_text", "")
            if q and q not in search_text:
                continue

            label = f"{title} 📌" if c.get("pinned") else title
            it = QtWidgets.QListWidgetItem(label)
            it.setData(QtCore.Qt.UserRole, c["id"])
            it.setData(QtCore.Qt.UserRole + 1, title)
            it.setData(QtCore.Qt.UserRole + 2, bool(c.get("pinned")))
            it.setFlags(it.flags() | QtCore.Qt.ItemIsEditable)
            it.setForeground(QtGui.QBrush(QtGui.QColor("#C9D1D9")))
            self.chat_list.addItem(it)

        self.chat_list.blockSignals(False)

    def _ensure_selected_chat(self):
        if self.chat_list.count() > 0 and not self.chat_list.currentItem():
            self.chat_list.setCurrentRow(0)

    def _select_chat_item(self, chat_id: str):
        for i in range(self.chat_list.count()):
            it = self.chat_list.item(i)
            if it.data(QtCore.Qt.UserRole) == chat_id:
                self.chat_list.setCurrentItem(it)
                return

    def _refresh_file_list(self, chat_id: str = None):
        self.file_list.clear()
        cid = chat_id or self.current_chat_id
        if not cid:
            return
        files = self.chat_mgr.list_files(cid)
        for f in files:
            name = str(f.get("name", "file"))
            label = f"🖼 {name}" if f.get("is_image") else f"📄 {name}"
            it = QtWidgets.QListWidgetItem(label)
            it.setData(QtCore.Qt.UserRole, str(f.get("path", "")))
            it.setData(QtCore.Qt.UserRole + 1, bool(f.get("is_image")))
            it.setData(QtCore.Qt.UserRole + 2, name)
            it.setForeground(QtGui.QBrush(QtGui.QColor("#C9D1D9")))
            self.file_list.addItem(it)

    def _selected_file_name(self) -> str:
        it = self.file_list.currentItem()
        if not it:
            return ""
        return str(it.data(QtCore.Qt.UserRole + 2) or "").strip()

    def on_chat_menu(self, pos):
        it = self.chat_list.itemAt(pos)
        if not it:
            return
        chat_id = it.data(QtCore.Qt.UserRole)
        is_pinned = self.chat_mgr.is_pinned(chat_id)

        menu = QtWidgets.QMenu(self)
        act_rename = menu.addAction("Rename")
        act_pin = menu.addAction("Unpin Chat" if is_pinned else "Pin Chat")
        act_clear_chat = menu.addAction("Clear Messages")
        menu.addSeparator()
        act_delete = menu.addAction("Delete")

        action = menu.exec(self.chat_list.mapToGlobal(pos))
        if action == act_rename:
            self._suppress_item_changed = True
            it.setText(str(it.data(QtCore.Qt.UserRole + 1) or it.text()))
            self._suppress_item_changed = False
            self.chat_list.editItem(it)
        elif action == act_pin:
            self.chat_mgr.set_pinned(chat_id, not is_pinned)
            self._refresh_chat_list()
            self._select_chat_item(chat_id)
        elif action == act_clear_chat:
            self.chat_mgr.clear_chat(chat_id)
            self._refresh_chat_list()
            self._select_chat_item(chat_id)
            if self.current_chat_id == chat_id:
                self._load_chat_messages(chat_id)
        elif action == act_delete:
            self.chat_mgr.delete_chat(chat_id)
            if self.current_chat_id == chat_id:
                self.current_chat_id = None
                self._clear_messages_view()
                self.file_list.clear()
            self._refresh_chat_list()
            self._ensure_selected_chat()

    def on_chat_item_edited(self, item: QtWidgets.QListWidgetItem):
        if self._suppress_item_changed:
            return

        chat_id = item.data(QtCore.Qt.UserRole)
        if not chat_id:
            return

        old_title = str(item.data(QtCore.Qt.UserRole + 1) or "").strip()
        new_title = item.text().strip() or old_title or "New Chat"
        if new_title == old_title:
            # Restore pin marker formatting when text is unchanged.
            self._suppress_item_changed = True
            if bool(item.data(QtCore.Qt.UserRole + 2)):
                item.setText(f"{old_title} 📌")
            else:
                item.setText(old_title)
            self._suppress_item_changed = False
            return

        self.chat_mgr.rename_chat(chat_id, new_title)
        self.status.setText("Chat renamed")
        self._refresh_chat_list()
        self._select_chat_item(chat_id)

    # ---------- Messages ----------
    def _clear_messages_view(self):
        while self.msg_layout.count():
            item = self.msg_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self.msg_layout.addStretch(1)

    def _bubble_width(self) -> int:
        base = int(self.settings.get_gui("MAX_BUBBLE_WIDTH", 760))
        viewport = self.scroll.viewport().width()
        if viewport > 0:
            base = min(base, int(viewport * 0.78))
        return max(360, base)

    def _load_chat_messages(self, chat_id: str):
        self._clear_messages_view()
        msgs = self.chat_mgr.get_messages(chat_id)
        max_w = self._bubble_width()

        stick = self._is_near_bottom()
        for m in msgs:
            b = Bubble(m["role"], m["content"], max_w)
            self.msg_layout.insertWidget(self.msg_layout.count() - 1, b)

        if stick:
            self._scroll_to_bottom()

    # ---------- Actions ----------
    def on_new_chat(self):
        chat_id = self.chat_mgr.create_chat("New Chat")
        self.current_chat_id = chat_id

        if self.settings.welcome_message:
            self.chat_mgr.add_message(chat_id, "assistant", self.settings.welcome_message)

        self._refresh_chat_list()

        # select new
        self._select_chat_item(chat_id)

        self._load_chat_messages(chat_id)
        self._refresh_file_list(chat_id)

    def on_select_chat(self, current, previous):
        if not current:
            return
        chat_id = current.data(QtCore.Qt.UserRole)
        self.current_chat_id = chat_id
        self._load_chat_messages(chat_id)
        self._refresh_file_list(chat_id)

    def on_clear(self):
        if not self.current_chat_id:
            return
        self.chat_mgr.clear_chat(self.current_chat_id)
        self._refresh_chat_list()
        self._load_chat_messages(self.current_chat_id)

    def on_persona(self):
        if not self.current_chat_id:
            return

        current = self.chat_mgr.get_persona(self.current_chat_id)

        dlg = QtWidgets.QDialog(self)
        dlg.setObjectName("Modal")
        dlg.setWindowTitle("Persona (per chat)")
        dlg.resize(760, 420)

        v = QtWidgets.QVBoxLayout(dlg)
        v.setContentsMargins(18, 18, 18, 18)
        v.setSpacing(12)

        title = QtWidgets.QLabel("Persona (per chat)")
        title.setObjectName("ModalTitle")

        sub = QtWidgets.QLabel("Overrides SYSTEM_PROMPT_DEFAULT only for this chat.")
        sub.setObjectName("ModalSub")

        txt = QtWidgets.QTextEdit()
        txt.setObjectName("ModalText")
        txt.setPlainText(current)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1)

        btn_default = QtWidgets.QPushButton("Use Default")
        btn_default.setObjectName("BtnModalDefault")
        btn_save = QtWidgets.QPushButton("Save")
        btn_save.setObjectName("BtnModalSave")

        def use_default():
            self.chat_mgr.set_persona(self.current_chat_id, self.settings.system_prompt_default)
            self.status.setText("Persona updated (default)")
            self.lbl_status.setText(f"Ready · {get_model_name(self.settings)} · Persona updated")
            dlg.accept()

        def save():
            self.chat_mgr.set_persona(self.current_chat_id, txt.toPlainText())
            self.status.setText("Persona updated")
            self.lbl_status.setText(f"Ready · {get_model_name(self.settings)} · Persona updated")
            dlg.accept()

        btn_default.clicked.connect(use_default)
        btn_save.clicked.connect(save)

        row.addWidget(btn_default)
        row.addWidget(btn_save)

        v.addWidget(title)
        v.addWidget(sub)
        v.addWidget(txt, 1)
        v.addLayout(row)

        dlg.exec()

    def on_monitor(self):
        dlg = MonitorDialog(self.settings, self)
        dlg.exec()

    def on_send(self):
        if not self.model_ready:
            self.status.setText("Loading model...")
            self.lbl_status.setText(f"Loading model · {get_model_name(self.settings)}")
            return

        if self.worker and self.worker.isRunning():
            return

        text = self.input.text().strip()
        if not text:
            return

        if not self.current_chat_id:
            self.on_new_chat()

        chat_id = self.current_chat_id

        stick = self._is_near_bottom()

        # save & show user
        self.chat_mgr.add_message(chat_id, "user", text)
        b = Bubble("user", text, self._bubble_width())
        self.msg_layout.insertWidget(self.msg_layout.count() - 1, b)

        if stick:
            self._scroll_to_bottom()

        self.input.clear()
        self.btn_send.setEnabled(False)
        self.status.setText("Generating...")
        self.lbl_status.setText(f"Sending · {get_model_name(self.settings)}")

        persona = self.chat_mgr.get_persona(chat_id)
        system_prompt = persona if persona else self.settings.system_prompt_default
        files_meta = self.chat_mgr.list_files(chat_id)
        selected_name = self._selected_file_name()
        file_ctx = self.chat_mgr.build_file_context(
            chat_id,
            int(self.settings.get_ai("MAX_FILE_CHARS", 14000)),
            preferred_name=selected_name
        )
        if files_meta:
            file_names = ", ".join([str(f.get("name", "")) for f in files_meta[:12] if f.get("name")])
            selected_line = selected_name if selected_name else "(none selected; use most recent upload)"
            system_prompt = (
                f"{system_prompt}\n\n"
                "Uploaded files are already available below. "
                "Do not say you cannot access uploaded files.\n"
                "When the user says 'this file', treat it as the selected file in the file list; "
                "if none is selected, treat it as the most recently uploaded file.\n"
                f"Uploaded file names: {file_names}\n"
                f"Selected file: {selected_line}"
            )

        history = self.chat_mgr.get_messages(chat_id)[-self.settings.max_history_messages :]
        effective_history = list(history)
        if file_ctx and effective_history and effective_history[-1].get("role") == "user":
            selected_line = selected_name if selected_name else "(most recent upload)"
            original_question = str(effective_history[-1].get("content", "")).strip()
            effective_history[-1] = {
                "role": "user",
                "content": (
                    "Use the uploaded file content below to answer the question. "
                    "Do not say you cannot access uploaded files.\n"
                    f"Target file: {selected_line}\n\n"
                    f"{file_ctx}\n\n"
                    f"Question: {original_question}"
                ),
            }

        self.worker = GenerateWorker(history=effective_history, system_prompt=system_prompt, settings=self.settings)
        self.worker.done.connect(lambda reply: self._on_reply(chat_id, reply))
        self.worker.fail.connect(self._on_error)
        self.worker.start()

    def on_upload_files(self):
        if not self.current_chat_id:
            self.on_new_chat()

        chat_id = self.current_chat_id
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Upload Files",
            "",
            "All Files (*.*)"
        )
        if not paths:
            return

        added = self.chat_mgr.add_files(chat_id, paths)
        self._refresh_file_list(chat_id)
        n = len(added)
        self.status.setText(f"Uploaded {n} file(s)")
        self.lbl_status.setText(f"Ready · {get_model_name(self.settings)} · {n} file(s) uploaded")

    def on_file_menu(self, pos):
        it = self.file_list.itemAt(pos)
        if not it:
            return

        path = str(it.data(QtCore.Qt.UserRole) or "")
        is_image = bool(it.data(QtCore.Qt.UserRole + 1))
        if not path:
            return

        menu = QtWidgets.QMenu(self)
        act_copy_path = menu.addAction("Copy Path")
        act_copy_image = menu.addAction("Copy Image") if is_image else None
        action = menu.exec(self.file_list.mapToGlobal(pos))

        if action == act_copy_path:
            QtWidgets.QApplication.clipboard().setText(path)
            self.status.setText("File path copied")
        elif is_image and action == act_copy_image:
            img = QtGui.QImage(path)
            if img.isNull():
                QtWidgets.QMessageBox.warning(self, "Copy Image", "Unable to read image.")
                return
            QtWidgets.QApplication.clipboard().setImage(img)
            self.status.setText("Image copied")

    def _on_reply(self, chat_id: str, reply: str):
        stick = self._is_near_bottom()

        self.chat_mgr.add_message(chat_id, "assistant", reply)
        b = Bubble("assistant", reply, self._bubble_width())
        self.msg_layout.insertWidget(self.msg_layout.count() - 1, b)

        self._refresh_chat_list()
        if stick:
            self._scroll_to_bottom()

        self.btn_send.setEnabled(True)
        self.status.setText("Ready")
        self.lbl_status.setText(f"Ready · {get_model_name(self.settings)}")

    def _on_error(self, msg: str):
        self.btn_send.setEnabled(True)
        self.status.setText("Error")
        self.lbl_status.setText("Error")
        QtWidgets.QMessageBox.critical(self, "Error", msg)

    def closeEvent(self, event):
        # Ensure background workers do not keep the process alive after window close.
        workers = [self.worker, self.model_worker]
        for w in workers:
            if w and w.isRunning():
                w.requestInterruption()
                w.quit()
                if not w.wait(1500):
                    w.terminate()
                    w.wait(500)
        QtWidgets.QApplication.quit()
        event.accept()


def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

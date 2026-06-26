"""Always-on-top caption overlay.

A frameless, translucent window pinned above other windows: committed text in
white, the still-unstable hypothesis tail dimmed in indigo, with small live
chips for language and display latency. Drag anywhere to move it.
"""

from __future__ import annotations

import html
import queue

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath
from PySide6.QtWidgets import QApplication, QHBoxLayout, QLabel, QVBoxLayout, QWidget

ACCENT = "#4F46E5"
ACCENT_SOFT = "#A5B4FC"
TEXT = "#F8FAFC"
MUTED = "#94A3B8"
BG = QColor(15, 15, 24, 235)
MAX_CHARS = 220  # roughly two caption lines


class CaptionWorker(QThread):
    committed = Signal(str)
    pending = Signal(str)
    status = Signal(str)

    def __init__(self, model: str, device: str, language: str | None, route: str | None = None):
        super().__init__()
        self._model = model
        self._device = device
        self._language = language
        self._route = route
        self._running = True
        self.transcriber = None  # the engine, set once loaded (read by the dashboard)
        self.requests: "queue.Queue[tuple[str, object]]" = queue.Queue()
        self.transcript: list[dict] = []
        self._text_acc = ""

    def request(self, kind: str, value=None) -> None:
        """Thread-safe control from the UI: ('language', code|None) or ('clear', None)."""
        self.requests.put((kind, value))

    def run(self) -> None:
        from .__main__ import build_engine
        from .asr import StreamingTranscriber
        from .audio import LoopbackCapture

        self.status.emit(f"loading {self._model}…")
        try:
            transcriber = StreamingTranscriber(
                model_size=self._model, device=self._device, language=self._language
            )
            transcriber = build_engine(transcriber, self._route)
            self.transcriber = transcriber
            capture = LoopbackCapture()
            capture.start()
        except Exception as e:  # noqa: BLE001 - surface instead of a dead overlay
            self.status.emit(f"error: {e}")
            return
        self.status.emit(f"{transcriber.device} · listening")

        try:
            while self._running:
                while not self.requests.empty():
                    kind, value = self.requests.get_nowait()
                    if kind == "language":
                        transcriber.set_language(value)
                        self._text_acc = ""
                        self.committed.emit("")
                        self.pending.emit("")
                        pin = value or "auto"
                        self.status.emit(f"language: {pin} · {transcriber.device}")
                    elif kind == "clear":
                        transcriber.set_language(transcriber.language)
                        self._text_acc = ""
                        self.committed.emit("")
                        self.pending.emit("")
                try:
                    block = capture.blocks.get(timeout=0.5)
                except queue.Empty:
                    continue
                for commit in transcriber.feed(block):
                    acc = (self._text_acc + " " + commit.text).strip()
                    if len(acc) > MAX_CHARS:
                        acc = acc[-MAX_CHARS:]
                        cut = acc.find(" ")  # don't open on half a word
                        if 0 <= cut < 40:
                            acc = acc[cut + 1 :]
                    self._text_acc = acc
                    self.transcript.append(
                        {"start": commit.start, "end": commit.end, "text": commit.text}
                    )
                    self.committed.emit(self._text_acc)
                    lang = commit.language or "…"
                    d50 = transcriber.stats._pct(transcriber.stats.display_latencies, 50)
                    latency = f"{d50:.2f}s" if d50 is not None else "…"
                    self.status.emit(f"{lang} · {latency} · {transcriber.device}")
                self.pending.emit(transcriber.pending_text)
        except Exception as e:  # noqa: BLE001 - died mid-session: say so on the chip
            self.status.emit(f"error: {e}")
        finally:
            capture.stop()
            if hasattr(transcriber, "drain"):
                for commit in transcriber.drain():
                    self.transcript.append(
                        {"start": commit.start, "end": commit.end, "text": commit.text}
                    )
            elif hasattr(transcriber, "close"):
                transcriber.close()
            from .export import save_session

            save_session(self.transcript)

    def stop(self) -> None:
        """Signal the loop to exit and wait for drain + session export."""
        self._running = False
        if self.isRunning():
            self.wait(10_000)


class Overlay(QWidget):
    def __init__(self, model: str, device: str, language: str | None, route: str | None = None):
        super().__init__(
            None,
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool,
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self._drag_offset = None

        font = QFont("Segoe UI", 15)
        chip_font = QFont("Cascadia Code", 9)
        if not chip_font.exactMatch():
            chip_font = QFont("Consolas", 9)

        self._text = QLabel("")
        self._text.setFont(font)
        self._text.setWordWrap(True)
        self._text.setTextFormat(Qt.RichText)
        self._text.setStyleSheet(f"color: {TEXT}; background: transparent;")

        self._chip = QLabel("starting…")
        self._chip.setFont(chip_font)
        self._chip.setStyleSheet(f"color: {MUTED}; background: transparent;")

        brand = QLabel("streamcap")
        brand.setFont(chip_font)
        brand.setStyleSheet(f"color: {ACCENT_SOFT}; background: transparent;")

        top = QHBoxLayout()
        top.addWidget(brand)
        top.addStretch(1)
        top.addWidget(self._chip)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 12, 22, 16)
        layout.setSpacing(6)
        layout.addLayout(top)
        layout.addWidget(self._text)

        screen = QApplication.primaryScreen().availableGeometry()
        width = int(screen.width() * 0.62)
        self.resize(width, 130)
        self.move((screen.width() - width) // 2, screen.height() - 190)

        self._committed = ""
        self._pending = ""

        self._worker = CaptionWorker(model, device, language, route=route)
        self._worker.committed.connect(self._on_committed)
        self._worker.pending.connect(self._on_pending)
        self._worker.status.connect(self._chip.setText)
        self._worker.start()

    # -- painting ---------------------------------------------------------

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(self.rect().adjusted(0, 0, -1, -1), 18, 18)
        p.fillPath(path, BG)
        p.setPen(QColor(79, 70, 229, 70))
        p.drawPath(path)

    # -- captions ---------------------------------------------------------

    def _render(self) -> None:
        # captions are untrusted text going into a RichText label — escape it
        committed = html.escape(self._committed)
        pending = (
            f' <span style="color: {ACCENT_SOFT};">{html.escape(self._pending)}</span>'
            if self._pending
            else ""
        )
        self._text.setText(f"{committed}{pending}")

    def _on_committed(self, text: str) -> None:
        self._committed = text
        self._render()

    def _on_pending(self, text: str) -> None:
        self._pending = text
        self._render()

    # -- dragging ---------------------------------------------------------

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.LeftButton:
            self._drag_offset = e.globalPosition().toPoint() - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e) -> None:
        if self._drag_offset is not None:
            self.move(e.globalPosition().toPoint() - self._drag_offset)

    def mouseReleaseEvent(self, e) -> None:
        self._drag_offset = None

    def keyPressEvent(self, e) -> None:
        if e.key() == Qt.Key_C and e.modifiers() & Qt.ControlModifier:
            full = "\n".join(entry["text"] for entry in self._worker.transcript)
            QApplication.clipboard().setText(full or self._committed)
        elif e.key() == Qt.Key_Escape:
            self.close()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e) -> None:
        # hide first so quitting feels instant, then let the worker drain the
        # cloud stream and write the session export before the process ends
        self.hide()
        self._worker.stop()
        e.accept()
        QApplication.quit()


def run_overlay(
    model: str,
    device: str,
    language: str | None,
    duration: float | None = None,
    dashboard: bool = False,
    route: str | None = None,
) -> int:
    import sys

    app = QApplication(sys.argv)
    overlay = Overlay(model, device, language, route=route)
    overlay.show()

    from .hotkeys import MOD_CONTROL, VK_F7, VK_F8, VK_F9, VK_F10, GlobalHotkeys

    hotkeys = GlobalHotkeys()
    app.installNativeEventFilter(hotkeys)

    languages = [None, "en", "fr", "ar"]
    lang_index = {"i": languages.index(language) if language in languages else 0}

    def cycle_language() -> None:
        lang_index["i"] = (lang_index["i"] + 1) % len(languages)
        overlay._worker.request("language", languages[lang_index["i"]])

    def toggle_click_through() -> None:
        on = not bool(overlay.windowFlags() & Qt.WindowTransparentForInput)
        overlay.setWindowFlag(Qt.WindowTransparentForInput, on)
        overlay.show()

    bindings = [
        (VK_F7, 0, "F7", cycle_language),
        (VK_F8, 0, "F8", lambda: overlay._worker.request("clear")),
        (VK_F9, 0, "F9", lambda: overlay.setVisible(not overlay.isVisible())),
        (VK_F10, 0, "F10", toggle_click_through),
        # quit must work even when the overlay is hidden or click-through
        (VK_F9, MOD_CONTROL, "Ctrl+F9", overlay.close),
    ]
    for vk, mods, name, action in bindings:
        if not hotkeys.register(vk, action, modifiers=mods):
            print(f"warning: global hotkey {name} unavailable (in use by another app)")
    app.aboutToQuit.connect(hotkeys.unregister_all)

    if dashboard:
        from .dashboard import start_dashboard
        from .metrics import SessionStats

        def provider() -> dict:
            t = overlay._worker.transcriber
            snap = (t.stats if t else SessionStats()).snapshot()
            snap.update(
                model=model,
                device=t.device if t else "loading",
                language=(t.detected_language if t else None) or "auto",
            )
            return snap

        print(f"dashboard: {start_dashboard(provider)}", flush=True)
    if duration:
        from PySide6.QtCore import QTimer

        # close (not quit): the worker drains and exports before the app exits
        QTimer.singleShot(int(duration * 1000), overlay.close)
    code = app.exec()
    overlay._worker.stop()  # no-op if closeEvent already ran
    return code

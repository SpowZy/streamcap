"""Global hotkeys via Win32 RegisterHotKey.

Single function keys, registered system-wide with no modifier: they work no
matter which window has focus, and unlike letter keys they never steal
keystrokes from whatever the user is typing in.

    F7      — cycle pinned language (auto → en → fr → ar)
    F8      — clear captions and reset the stream
    F9      — show / hide the caption overlay
    F10     — toggle click-through (mouse passes through the overlay)
    Ctrl+F9 — quit (drains, exports the session, exits cleanly)
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
from typing import Callable

from PySide6.QtCore import QAbstractNativeEventFilter

WM_HOTKEY = 0x0312
MOD_CONTROL = 0x0002
MOD_NOREPEAT = 0x4000
VK_F7 = 0x76
VK_F8 = 0x77
VK_F9 = 0x78
VK_F10 = 0x79


class GlobalHotkeys(QAbstractNativeEventFilter):
    def __init__(self):
        super().__init__()
        self._actions: dict[int, Callable[[], None]] = {}
        self._next_id = 1

    def register(self, vk: int, callback: Callable[[], None], modifiers: int = 0) -> bool:
        hotkey_id = self._next_id
        ok = ctypes.windll.user32.RegisterHotKey(None, hotkey_id, modifiers | MOD_NOREPEAT, vk)
        if ok:
            self._actions[hotkey_id] = callback
            self._next_id += 1
        return bool(ok)

    def nativeEventFilter(self, event_type, message):
        if event_type == b"windows_generic_MSG":
            msg = wintypes.MSG.from_address(int(message))
            if msg.message == WM_HOTKEY and msg.wParam in self._actions:
                self._actions[msg.wParam]()
                return True, 0
        return False, 0

    def unregister_all(self) -> None:
        for hotkey_id in self._actions:
            ctypes.windll.user32.UnregisterHotKey(None, hotkey_id)
        self._actions.clear()

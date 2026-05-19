"""Persistent app settings — thin wrappers around QSettings.

Centralizes the SETTINGS_ORG / SETTINGS_APP names so every QSettings call
reads/writes the same store, and provides helpers for the most common
remembered values (e.g. the last directory used in an Open dialog).
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import QSettings

SETTINGS_ORG = "framewise"
SETTINGS_APP = "framewise-py"

_LAST_DIR_KEY = "last_dir"


def settings() -> QSettings:
    return QSettings(SETTINGS_ORG, SETTINGS_APP)


def get_last_dir() -> str:
    """Return the directory last used in an Open dialog, or "" if none."""
    return settings().value(_LAST_DIR_KEY, "", type=str)


def set_last_dir_from_path(path: str | Path) -> None:
    """Remember `path`'s parent directory (or `path` itself if a directory)."""
    p = Path(path)
    if not p.exists():
        return
    d = p if p.is_dir() else p.parent
    settings().setValue(_LAST_DIR_KEY, str(d))

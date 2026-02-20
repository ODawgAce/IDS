from __future__ import annotations


from pathlib import Path

import sys

import os


def is_frozen() -> bool:

    return getattr(sys, "frozen", False)


def base_dir() -> Path:
    """

    Dev:

      IDS/ (root repo)



    PyInstaller:

      sys._MEIPASS  -> katalog z _internal

    """

    if is_frozen():

        return Path(sys._MEIPASS).resolve()

    # dev mode → 2 poziomy w górę od app/

    return Path(__file__).resolve().parents[1]


def app_dir() -> Path:
    """

    Katalog wykonywalny (exe) albo root projektu

    """

    if is_frozen():

        return Path(sys.executable).resolve().parent

    return base_dir()


def p(*parts: str) -> str:
    """

    Do ZASOBÓW (scripts, tools, etc.) – tych spakowanych z exe

    """

    return str(base_dir().joinpath(*parts))


def ensure_data_dirs() -> dict[str, Path]:
    """

    Wymusza strukturę katalogów runtime w:

    C:\\Users\\<user>\\AppData\\Local\\IDS

    """

    local_appdata = Path(

        os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")

    )

    root = local_appdata / "IDS"

    artifacts = root / "artifacts"

    runtime = root / "runtime"

    logs = root / "logs"

    artifacts.mkdir(parents=True, exist_ok=True)

    runtime.mkdir(parents=True, exist_ok=True)

    logs.mkdir(parents=True, exist_ok=True)

    return {

        "root": root,

        "artifacts": artifacts,

        "runtime": runtime,

        "logs": logs,

    }

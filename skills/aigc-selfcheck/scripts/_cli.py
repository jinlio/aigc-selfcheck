"""Agent Skill 的本地 CLI 适配层。

只接受本地文件；不读取凭据；不把文本发到任何远端。
优先使用 PATH 上的 aigc-selfcheck，否则用项目内的 uv 环境。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def setup_stdio() -> None:
    """Windows 控制台默认不是 UTF-8；统一成 UTF-8，避免中文 JSON 乱码。"""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def ensure_local(path: str, label: str) -> None:
    if path.startswith(("http://", "https://", "ftp://", "file://")):
        raise SystemExit(f"{label} 必须是本地文件，不接受 URL：{path}")
    if not Path(path).exists():
        raise SystemExit(f"{label} 不存在：{path}")


def base_command() -> list[str]:
    exe = shutil.which("aigc-selfcheck")
    if exe:
        return [exe]
    root = repo_root()
    uv = shutil.which("uv")
    if uv and (root / "pyproject.toml").exists():
        return [uv, "run", "--project", str(root), "--quiet", "aigc-selfcheck"]
    return [sys.executable, "-m", "aigc_selfcheck.cli"]


def run(args: list[str]) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    return subprocess.run(
        base_command() + args,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        env=env,
    )


def fail(result: subprocess.CompletedProcess) -> int:
    if result.stderr:
        sys.stderr.write(result.stderr)
    return result.returncode or 1

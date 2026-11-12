"""Skills 3 & 4: read_file / write_file - sandboxed to the workspace folder.

Claude only ever sees paths relative to WORKSPACE. Any path that resolves
outside it (../, absolute paths, symlinks out) is rejected.
"""

from pathlib import Path

from . import skill

WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"
MAX_BYTES = 100_000


def _safe_path(path: str) -> Path:
    root = WORKSPACE.resolve()
    target = (root / path).resolve()
    if not target.is_relative_to(root):
        raise PermissionError(f"Path {path!r} is outside the workspace; only workspace files are allowed")
    return target


@skill(
    name="read_file",
    description="Read a text file from the workspace folder. The path is relative to the workspace.",
    properties={"path": {"type": "string", "description": "File path relative to the workspace, e.g. 'notes.txt'."}},
)
def read_file(path: str) -> str:
    target = _safe_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"No such file in workspace: {path!r}")
    if target.stat().st_size > MAX_BYTES:
        raise ValueError(f"File is larger than {MAX_BYTES} bytes")
    return target.read_text(encoding="utf-8")


@skill(
    name="write_file",
    description=(
        "Write text to a file in the workspace folder, creating folders as needed. "
        "Overwrites the file if it already exists."
    ),
    properties={
        "path": {"type": "string", "description": "File path relative to the workspace, e.g. 'notes.txt'."},
        "content": {"type": "string", "description": "The full text to write."},
    },
)
def write_file(path: str, content: str) -> str:
    data = content.encode("utf-8")
    if len(data) > MAX_BYTES:
        raise ValueError(f"Content is larger than {MAX_BYTES} bytes")
    target = _safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return f"Wrote {len(data)} bytes to {target.relative_to(WORKSPACE.resolve())}"

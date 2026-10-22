"""Run PyLingual in a subprocess."""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from . import bytecode as bc


@dataclass
class Decompiled:
    source: str | None
    info: dict


def pylingual_python() -> str | None:
    """Interpreter that has pylingual installed (env override, else this one)."""
    env = os.environ.get("PYDEOBF_PYLINGUAL_PYTHON")
    if env:
        return env
    try:
        import importlib.util
        if importlib.util.find_spec("pylingual") is not None:
            return sys.executable
    except Exception:
        pass
    return None


def _child_limits():
    # undo the parent's soft memory cap (torch maps large virtual regions)
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        resource.setrlimit(resource.RLIMIT_AS, (hard, hard))
    except (ValueError, OSError):
        pass


def decompile(blob: bc.CodeBlob, workdir: Path, stem: str, *, timeout=600, redis=None) -> Decompiled:
    py = pylingual_python()
    if py is None:
        return Decompiled(None, {"ok": False, "error": "pylingual not installed (pip install 'pydeobf[decompile]' or set PYDEOBF_PYLINGUAL_PYTHON)"})
    if blob.raw is None:
        return Decompiled(None, {"ok": False, "error": "no marshal bytes for this code object"})
    if blob.version < (3, 6):
        return Decompiled(None, {"ok": False, "error": f"PyLingual does not support Python {bc.vstr(blob.version)}"})
    workdir = Path(workdir).resolve()  # the worker runs with cwd=workdir
    workdir.mkdir(parents=True, exist_ok=True)
    # junk MAKE_FUNCTIONs crash PyLingual's masking, so fix them before the first run
    neutralized = bc.neutralize_bad_makefunc(blob)
    if neutralized is not None:
        blob = bc.CodeBlob(bc.unmarshal(neutralized[0], blob.version)[0], blob.version, neutralized[0])
    d = _run(py, blob.raw, blob.version, workdir, stem, timeout, redis)
    if d.source is None and "invalid target" in str(d.info.get("error", "")):
        padded = bc.pad_bad_jumps(blob)
        if padded is not None:
            raw, notes = padded
            d = _run(py, raw, blob.version, workdir, stem, timeout, redis)
            d.info["padded_jumps"] = notes
    if neutralized is not None:
        d.info["neutralized_makefunc"] = neutralized[1]
    return d


def _run(py, raw, version, workdir, stem, timeout, redis=None) -> Decompiled:
    pyc = workdir / f"{stem}.pyc"
    out = workdir / f"{stem}.pylingual.py"
    pyc.write_bytes(bc.make_pyc(raw, version))
    env = dict(os.environ)
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [py, "-m", "pydeobf.pylingual_worker", str(pyc), str(out), "--timeout", str(timeout)]
    if redis:
        cmd += ["--redis", redis]
    try:
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout + 120,
                              preexec_fn=_child_limits, cwd=str(workdir))
    except subprocess.TimeoutExpired:
        return Decompiled(None, {"ok": False, "error": f"pylingual timed out after {timeout}s"})
    info_path = Path(str(out) + ".json")
    info = json.loads(info_path.read_text()) if info_path.exists() else {
        "ok": False, "error": (proc.stderr or proc.stdout)[-2000:]}
    src = out.read_text(encoding="utf-8", errors="replace") if out.exists() else None
    info["pyc"] = str(pyc)
    return Decompiled(src, info)

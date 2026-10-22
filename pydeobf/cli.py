"""pydeobf - static, general-purpose Python deobfuscator.

Nothing from the sample is ever executed: layers are peeled by a sandboxed
bytecode emulator, and recovered bytecode is optionally decompiled with PyLingual.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
from pathlib import Path

from .engine import Engine, Options


def _version(s):
    parts = s.split(".")
    return (int(parts[0]), int(parts[1]))


def _inputs(paths):
    for p in paths:
        p = Path(p)
        if p.is_dir():
            for q in sorted(p.rglob("*")):
                if q.suffix in (".py", ".pyc", ".pyw") and q.is_file() and ".deobf" not in q.parts[-2]:
                    yield q
        else:
            yield p


def main(argv=None):
    ap = argparse.ArgumentParser(prog="pydeobf", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help=".py / .pyc files or directories")
    ap.add_argument("-o", "--out", help="output directory (default: <input>.deobf/; per-file subdirs for many inputs)")
    ap.add_argument("--py-version", type=_version, help="Python version hint for header-less marshal blobs, e.g. 3.9")
    ap.add_argument("--no-pylingual", action="store_true", help="never call PyLingual")
    ap.add_argument("--decompile-all", action="store_true", help="also decompile wrapper layers")
    ap.add_argument("--pylingual-timeout", type=int, default=600)
    ap.add_argument("--pylingual-redis", metavar="HOST[:PORT]", default=os.environ.get("PYDEOBF_PYLINGUAL_REDIS"),
                    help="Redis server PyLingual uses to cache statement translations across runs "
                    "(default port 6379; env PYDEOBF_PYLINGUAL_REDIS)")
    ap.add_argument("--max-steps", type=int, default=30_000_000, help="emulator instruction budget per layer")
    ap.add_argument("--max-seconds", type=float, default=120.0, help="emulator time budget per layer")
    ap.add_argument("--max-layers", type=int, default=400)
    ap.add_argument("--platform", choices=["windows", "linux"], default="windows",
                    help="platform the emulated sys/os modules pretend to be")
    ap.add_argument("--pyarmor-runtime", help="path to pyarmor_runtime.{so,pyd,dylib} for PyArmor samples "
                    "(auto-detected near the input; the built-in trial key is used otherwise)")
    ap.add_argument("--no-pyarmor", action="store_true", help="do not attempt static PyArmor decryption")
    ap.add_argument("--no-fold", action="store_true", help="don't fold encoded constant expressions")
    ap.add_argument("--no-rename", action="store_true", help="don't rename non-Latin identifiers")
    ap.add_argument("--no-blob-search", action="store_true")
    ap.add_argument("--mem-limit-gb", type=float, default=8.0,
                    help="soft address-space cap for this process (0 = none)")
    ap.add_argument("-q", "--quiet", action="store_true")
    a = ap.parse_args(argv)

    if a.mem_limit_gb:
        lim = int(a.mem_limit_gb * (1 << 30))
        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        if hard == resource.RLIM_INFINITY or lim < hard:
            resource.setrlimit(resource.RLIMIT_AS, (lim, hard))
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20000))

    opts = Options(version_hint=a.py_version, use_pylingual=not a.no_pylingual, decompile_all=a.decompile_all,
                   pylingual_timeout=a.pylingual_timeout, pylingual_redis=a.pylingual_redis, max_steps=a.max_steps,
                   max_seconds=a.max_seconds, max_layers=a.max_layers, platform=a.platform, fold=not a.no_fold,
                   rename=not a.no_rename, blob_search=not a.no_blob_search, quiet=a.quiet,
                   pyarmor_runtime=a.pyarmor_runtime, pyarmor=not a.no_pyarmor)
    files = list(_inputs(a.inputs))
    rc = 0
    used = set()
    for f in files:
        if a.out:
            name = f.name
            if name in used:
                name = f"{f.parent.name}_{f.name}"
            used.add(name)
            outdir = Path(a.out) if len(files) == 1 else Path(a.out) / (name + ".deobf")
        else:
            outdir = f.with_name(f.name + ".deobf")
        eng = Engine(opts)
        eng.workdir = outdir
        try:
            eng.run(f)
            report = eng.write(outdir)
        except Exception as e:  # keep going on batches
            print(f"[pydeobf] {f}: failed: {type(e).__name__}: {e}", file=sys.stderr)
            rc = 1
            continue
        print(eng.render(report))
        print(f"[pydeobf] results in {outdir}/ (final.py, report.txt, layers/)")
    return rc


if __name__ == "__main__":
    sys.exit(main())

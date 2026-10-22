"""Subprocess entry point that runs PyLingual on one .pyc.

Usage: python -m pydeobf.pylingual_worker IN.pyc OUT.py [--timeout S]
Writes OUT.py and OUT.py.json (equivalence summary).
"""

import argparse
import json
import logging
from pathlib import Path


class _CPython312Probe:
    """xdis treats magic 3531 as RustPython 3.13 when the code object at offset
    0x10 lacks FLAG_REF ('c' instead of 0xe3).  marshal omits that flag for
    objects with refcount 1, so genuine CPython 3.12 blobs (e.g. from
    marshal.dumps(compile(...))) get misrouted.  Answer that single probe read
    with 0xe3; every other read returns the real bytes."""

    def __init__(self, fp):
        self._fp = fp
        self._armed = True

    def read(self, n=-1):
        if self._armed and n == 1 and self._fp.tell() == 0x10:
            self._armed = False
            self._fp.read(1)
            return b"\xe3"
        return self._fp.read(n)

    def __getattr__(self, name):
        return getattr(self._fp, name)


def _patch_xdis_312_probe():
    import xdis.load

    orig = xdis.load.load_module_from_file_object

    def patched(fp, *a, **k):
        pos = fp.tell()
        head = fp.read(0x11)
        fp.seek(pos)
        if head[:4] == b"\xcb\r\r\n" and head[0x10:0x11] == b"c":
            fp = _CPython312Probe(fp)
        return orig(fp, *a, **k)

    xdis.load.load_module_from_file_object = patched


def _redis_reachable(host, port):
    import redis
    try:
        return redis.StrictRedis(host=host, port=port, socket_connect_timeout=2).ping()
    except redis.RedisError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pyc")
    ap.add_argument("out")
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--trust-lnotab", action="store_true")
    ap.add_argument("--redis", help="HOST[:PORT] of a Redis translation cache")
    a = ap.parse_args()

    logging.basicConfig(level="WARNING")
    _patch_xdis_312_probe()
    # Tolerate out-of-range operands in unreachable anti-disassembly junk so
    # PyLingual's xdis-based loader doesn't abort on dead bytecode.
    from pydeobf.bytecode import patch_xdis_lenient
    patch_xdis_lenient()
    from pylingual.decompiler import decompile

    out = Path(a.out)
    redis_host, redis_port = None, 6379
    if a.redis:
        redis_host, _, port = a.redis.partition(":")
        redis_port = int(port or 6379)
        if not _redis_reachable(redis_host, redis_port):
            logging.warning("Redis %s:%s unreachable; decompiling without the translation cache", redis_host, redis_port)
            redis_host = None
    summary = {"ok": False}
    try:
        res = decompile(pyc=Path(a.pyc), save_to=out, config_file=None, version=None, top_k=a.top_k,
                        trust_lnotab=a.trust_lnotab, timeout=a.timeout,
                        redis_cache_server_ip=redis_host, redis_port=redis_port)
        results = [r for r in res.equivalence_results if hasattr(r, "success")]

        def describe(r):
            name = r.names() if hasattr(r, "names") else type(r).__name__
            return f"{name}: {getattr(r, 'message', r)}"

        summary = {
            "ok": True,
            "success_rate": res.calculate_success_rate(),
            "code_objects": len(results),
            "failures": [describe(r) for r in results if not r.success][:50],
        }
    except Exception as e:  # report, don't crash the pipeline
        summary = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    Path(str(out) + ".json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Brute-force decoding of encoded blobs (fallback when emulation fails).

Given a long str/bytes constant, search short chains of reversible encodings
(reverse, base64/32/16/85, hex, zlib/gzip/bz2/lzma, marshal, rot13,
Kyrie-style shifts, int lists, escape sequences) for something that is valid
Python source or a marshalled code object.  Purely data transformations.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import zlib
from collections import Counter
from dataclasses import dataclass

from . import bytecode as bc
from . import safelib

MIN_BLOB = 48
MAX_DEPTH = 7


@dataclass
class Candidate:
    kind: str  # "source" | "code"
    value: object
    chain: list


_B64 = re.compile(rb"^[A-Za-z0-9+/=\s]+$")
_B64URL = re.compile(rb"^[A-Za-z0-9_\-=\s]+$")
_B32 = re.compile(rb"^[A-Z2-7=\s]+$")
_HEX = re.compile(rb"^(?:[0-9a-fA-F]{2}|\s)+$")
_INTS = re.compile(r"^[\s\[\(]*-?\d+(?:\s*[^\d\s\-]{1,3}\s*-?\d+){7,}[\s\]\)]*$")
_ESC = re.compile(rb"(?:\\x[0-9a-fA-F]{2}){4,}")


def _as_bytes(v):
    if isinstance(v, bytes):
        return v
    try:
        return v.encode("latin-1")
    except UnicodeEncodeError:
        return v.encode("utf-8", "surrogatepass")


def _strip(b):
    return re.sub(rb"\s+", b"", b)


def _t_reverse(v):
    return v[::-1]


def _t_b64(v):
    b = _as_bytes(v)
    if not _B64.match(b):
        return None
    s = _strip(b)
    if len(s) % 4:
        return None
    return base64.b64decode(s, validate=True)


def _t_b64url(v):
    b = _as_bytes(v)
    if not _B64URL.match(b) or not re.search(rb"[-_]", b):
        return None
    return base64.urlsafe_b64decode(_strip(b))


def _t_b32(v):
    b = _as_bytes(v)
    if not _B32.match(b):
        return None
    return base64.b32decode(_strip(b))


def _t_b16(v):
    b = _strip(_as_bytes(v))
    if not _HEX.match(b) or len(b) % 2:
        return None
    return binascii.unhexlify(b)


def _t_b85(v):
    b = _strip(_as_bytes(v))
    if len(b) < MIN_BLOB or not all(33 <= c <= 126 for c in b[:256]):
        return None
    return base64.b85decode(b)


def _t_a85(v):
    b = _strip(_as_bytes(v))
    if b.startswith(b"<~") and b.endswith(b"~>"):
        return base64.a85decode(b, adobe=True)
    if len(b) < MIN_BLOB or not all(33 <= c <= 117 or c == 122 for c in b[:256]):  # '!'..'u', 'z'
        return None
    return base64.a85decode(b)


def _t_zlib(v):
    b = _as_bytes(v)
    if b[:1] not in (b"x", b"\x08", b"\x18", b"\x28", b"\x38", b"\x48", b"\x58", b"\x68"):
        return None
    try:
        return safelib.zlib_decompress(b)
    except zlib.error as e:
        if "incorrect data check" not in str(e):
            raise
        # damaged Adler-32 trailer only: the deflate stream itself is intact
        return safelib.zlib_decompress(b[2:-4], -15)


def _t_gzip(v):
    b = _as_bytes(v)
    if not b.startswith(b"\x1f\x8b"):
        return None
    return safelib.gzip_decompress(b)


def _t_raw_deflate(v):
    b = _as_bytes(v)
    return safelib.zlib_decompress(b, -15)


def _t_bz2(v):
    b = _as_bytes(v)
    if not b.startswith(b"BZh"):
        return None
    return safelib.bz2_decompress(b)


def _t_lzma(v):
    b = _as_bytes(v)
    if not (b.startswith(b"\xfd7zXZ") or b[:3] == b"\x5d\x00\x00"):
        return None
    return safelib.lzma_decompress(b)


def _t_rot13(v):
    if not isinstance(v, str):
        return None
    return codecs.decode(v, "rot13")


def _t_escape(v):
    b = _as_bytes(v)
    if not _ESC.search(b[:4096]):
        return None
    return codecs.escape_decode(b)[0]


def _t_ints(v):
    s = v if isinstance(v, str) else v.decode("latin-1")
    if not _INTS.match(s[:200_000]):
        return None
    nums = [int(x) for x in re.findall(r"-?\d+", s)]
    best = None
    lo = min(nums)
    for off in {0, lo - 32, lo - 10, lo - 9}:
        try:
            out = "".join(chr(n - off) for n in nums)
        except (ValueError, OverflowError):
            continue
        if _printable_ratio(out) > 0.95:
            best = out
            break
    return best


_KYRIE = "abcdefghijklmnopqrstuvwxyz0123456789"


def _dkyrie(text):
    tbl = {c: _KYRIE[(i + 1) % len(_KYRIE)] for i, c in enumerate(_KYRIE)}
    return "".join(tbl.get(c, c) for c in text)


def _t_shift(v):
    """Kyrie-Eleison style: every char shifted by a constant key (newline -> 'ζ')."""
    if not isinstance(v, str) or len(v) < MIN_BLOB:
        return None
    if sum(1 for c in v[:2000] if ord(c) > 0x2FF) < 0.8 * min(len(v), 2000):
        return None
    common = Counter(c for c in v if c != "ζ").most_common(3)
    for ch, _ in common:
        key = ord(ch) - 32
        try:
            out = "".join("\n" if c == "ζ" else chr(ord(c) - key) for c in v)
        except ValueError:
            continue
        if _printable_ratio(out) > 0.97:
            return _dkyrie(out) if _looks_like_python(_dkyrie(out)) else out
    return None


TRANSFORMS = [
    ("marshal", None), ("zlib", _t_zlib), ("gzip", _t_gzip), ("bz2", _t_bz2), ("lzma", _t_lzma),
    ("b64", _t_b64), ("b64url", _t_b64url), ("b32", _t_b32), ("hex", _t_b16), ("b85", _t_b85), ("a85", _t_a85),
    ("escape", _t_escape), ("ints", _t_ints), ("shift", _t_shift), ("reverse", _t_reverse),
    ("raw-deflate", _t_raw_deflate), ("rot13", _t_rot13),
]


def _printable_ratio(s):
    if not s:
        return 0.0
    sample = s[:4000]
    ok = sum(1 for c in sample if c.isprintable() or c in "\n\r\t")
    return ok / len(sample)


_KW = re.compile(r"\b(import|def|class|print|exec|return|lambda|for|if|from)\b")


def _looks_like_python(s: str) -> bool:
    return len(_KW.findall(s[:20000])) >= 2


def as_python_source(v):
    """Return v as source text if it is non-trivial, compilable Python."""
    if isinstance(v, (bytes, bytearray)):
        try:
            s = bytes(v).decode("utf-8")
        except UnicodeDecodeError:
            return None
    else:
        s = v
    if len(s.strip()) < 8 or _printable_ratio(s) < 0.9:
        return None
    try:
        code = compile(s, "<blob>", "exec")  # parse only
    except (SyntaxError, ValueError, MemoryError, RecursionError):
        return None
    if not _looks_like_python(s) and len(code.co_names) < 2:
        return None
    return s


_ROUND_DECODERS = ("b64", "b85", "b32", "hex", "b64url", "reverse")
_ROUND_UNPACKERS = ("zlib", "lzma", "bz2", "gzip")
MAX_ROUNDS = 1000


def _peel_rounds(blob, version_hint):
    """Packers that repeat one decode + decompress pair N times (``for _ in range(100): d = b85(zlib(d))``):
    keep applying a pair while it works, then look for code / source. Either order is tried, since the loader
    may apply them in an order the data was not built for."""
    fns = dict(TRANSFORMS)
    for dec in _ROUND_DECODERS:
        for unp in _ROUND_UNPACKERS:
            for pair in ((dec, unp), (unp, dec)):
                value, rounds = blob, 0
                while rounds < MAX_ROUNDS:
                    try:
                        out = fns[pair[1]](fns[pair[0]](value))
                    except safelib._vm.VMStop:
                        out = None
                    except Exception:
                        out = None
                    if out is None or len(out) < 8 or out == value:
                        break
                    value, rounds = out, rounds + 1
                if rounds < 3:  # one or two rounds are the ordinary search's job
                    continue
                chain = [f"({'+'.join(pair)})x{rounds}"]
                if isinstance(value, (bytes, bytearray)):
                    cb = bc.detect_marshal(bytes(value), version_hint)
                    if cb is not None:
                        return Candidate("code", cb, chain + ["marshal"])
                src = as_python_source(value)
                if src is not None:
                    return Candidate("source", src, chain)
                rest = search(value, version_hint, max_depth=3, max_nodes=60, rounds=False)
                if rest is not None:
                    return Candidate(rest.kind, rest.value, chain + rest.chain)
    return None


def search(blob, version_hint=None, max_depth=MAX_DEPTH, max_nodes=400, rounds=True) -> Candidate | None:
    if blob is None or len(blob) < MIN_BLOB:
        return None
    # a stage carried as plain source that runs an encoded payload, e.g. written to a .pth / startup file:
    # b'import base64;exec(base64.b64decode(b"..."))'
    if len(blob) >= 200 and re.search(rb"\b(exec|eval)\s*\(" if isinstance(blob, (bytes, bytearray))
                                      else r"\b(exec|eval)\s*\(", blob):
        src = as_python_source(blob) or as_python_source(blob.strip())  # '\texec(...)' cut from an indented block
        if src is not None:
            return Candidate("source", src, ["embedded-source"])
    if rounds and len(blob) >= 1024:
        c = _peel_rounds(blob, version_hint)
        if c is not None:
            return c
    seen = set()
    frontier = [(blob, [])]
    nodes = 0
    while frontier and nodes < max_nodes:
        nxt = []
        for value, chain in frontier:
            nodes += 1
            for name, fn in TRANSFORMS:
                if chain and chain[-1] == name and name in ("reverse", "rot13"):
                    continue
                try:
                    if name == "marshal":
                        if not isinstance(value, (bytes, bytearray)):
                            continue
                        cb = bc.detect_marshal(bytes(value), version_hint)
                        if cb is not None:
                            return Candidate("code", cb, chain + ["marshal"])
                        if len(value) > 16 and value[2:4] == b"\r\n":  # a whole .pyc: 16-byte header, then marshal
                            cb = bc.detect_marshal(bytes(value[16:]), version_hint)
                            if cb is not None:
                                return Candidate("code", cb, chain + ["pyc"])
                        continue
                    out = fn(value)
                except safelib._vm.VMStop:
                    continue
                except Exception:
                    continue
                if out is None or len(out) < 8 or out == value:
                    continue
                newchain = chain + [name]
                src = as_python_source(out)
                if src is not None and name not in ("reverse", "rot13"):
                    return Candidate("source", src, newchain)
                if rounds and len(newchain) <= 2 and len(out) >= 1024:  # b64 outside, then N repeated rounds
                    c = _peel_rounds(out, version_hint)
                    if c is not None:
                        return Candidate(c.kind, c.value, newchain + c.chain)
                key = (bytes(out) if isinstance(out, (bytes, bytearray)) else out.encode("utf-8", "surrogatepass"))[:4096]
                if key in seen or len(newchain) >= max_depth:
                    continue
                seen.add(key)
                nxt.append((out, newchain))
        frontier = nxt
    return None


def search_many(blobs, version_hint=None, limit=20):
    out = []
    for b in sorted({b for b in blobs if isinstance(b, (str, bytes)) and len(b) >= MIN_BLOB}, key=len, reverse=True)[:limit]:
        c = search(b, version_hint)
        if c is not None:
            out.append(c)
    return out


# ------------------------------------------------------------------ keyed XOR

_KEYED_DECODES = [("b85", _t_b85), ("b64", _t_b64), ("a85", _t_a85), ("hex", _t_b16), ("raw", None)]
_PACKED_MAGIC = (b"\x78", b"BZh", b"\xfd7zXZ", b"\x1f\x8b", b"\x63", b"\xe3", b"\x5d\x00")


def _xor(data, key):
    n = len(key)
    reps = key * (len(data) // n + 1)
    return (int.from_bytes(data, "big") ^ int.from_bytes(reps[:len(data)], "big")).to_bytes(len(data), "big")


def search_keyed(pairs, version_hint=None, limit=300):
    """Blobs XORed with a repeating key stored right beside them (a short constant in the same code object):
    ``zlib.decompress(xor(b85decode(BLOB), KEY))``, e.g. one encrypted body per function."""
    out, done = [], set()
    for blob, keys in pairs[:limit]:
        if blob in done or not keys:
            continue
        done.add(blob)
        found = None
        for dname, dec in _KEYED_DECODES:
            try:
                data = _as_bytes(blob) if dec is None else dec(blob)
            except Exception:
                continue
            if not data or len(data) < 16:
                continue
            data = bytes(data)
            for k in keys:
                kb = bytes([k]) if isinstance(k, int) else k.encode("utf-8", "surrogatepass") if isinstance(k, str) else bytes(k)
                head = _xor(data[:8], kb[:8] if len(kb) >= 8 else (kb * 8)[:8])
                if not head.startswith(_PACKED_MAGIC):  # cheap check before XORing the whole blob
                    continue
                c = search(_xor(data, kb), version_hint, max_depth=3, max_nodes=30, rounds=False)
                if c is not None:
                    found = Candidate(c.kind, c.value, ([] if dec is None else [dname]) + ["xor-key"] + c.chain)
                    break
            if found:
                break
        if found:
            out.append(found)
    return out

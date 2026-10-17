"""Fingerprints for known Python obfuscators (used for reporting only)."""

from __future__ import annotations

import re

# name -> list of (regex over text, description)
SIGNATURES = {
    "Specter": [r"obfuscated with Specter", r"Func\.define\('__\d{4}__'", r"GG! You just deobfuscated Specter",
                r"billythegoat356/Specter"],
    "Berserker": [r"class Berserker\(\)", r"_sparkle='''", r"\bBerserker\b[\s\S]*\b_sparkle\b"],
    "Kramer": [r"class Kramer\(\)", r"\bKramer\b[\s\S]*\b_sparkle\b"],
    "Kyrie-Eleison": [r"Kyrie\._?(?:en|de)crypt", r"ζ"],
    "PyObfuscate": [r"Obfuscated with PyObfuscate", r"__import__\('marshal'\)\.loads\(__\[::-1\]\)",
                    r"exec\(str\(chr\(35\)(?:\+chr\(1\)){20}", r"_ = lambda __ : __import__\("],
    "pyminifier": [r"Created by pyminifier", r"exec\((?:bz2|zlib|lzma)\.decompress\(base64\.b64decode\('",
                   r"liftoff/pyminifier"],
    # PyArmor is decided structurally in pyarmor_evidence(); mentioning the
    # name in a string is not enough.
    "Hyperion": [r"billythegoat356/Hyperion", r"obfuscated with Hyperion", r"Hyperion\(\)"],
    "marshal-exec": [r"marshal\.loads\(", r"__import__\(['\"]marshal['\"]\)"],
    "base64/zlib-exec": [r"exec\(\s*(?:__import__\(['\"])?(?:zlib|base64|bz2|lzma)"],
}

_COMPILED = {k: [re.compile(p) for p in v] for k, v in SIGNATURES.items()}


_ARMOR_CALL = re.compile(r"__pyarmor__\s*\(\s*__name__\s*,\s*__file__\s*,\s*b['\"](?:PY\d{6}|PYARMOR)")
_ARMOR_IMPORT = re.compile(r"^\s*from\s+(?:\S*\.)?(pyarmor_runtime_\d{6}|pytransform)\s+import\s+(?:__pyarmor__|pyarmor_runtime)",
                           re.M)


def pyarmor_evidence_source(text: str) -> bool:
    """An actual armored module: runtime import plus __pyarmor__(..., b'PY...') call."""
    return bool(_ARMOR_IMPORT.search(text) and _ARMOR_CALL.search(text))


def pyarmor_evidence_code(names, strings) -> bool:
    has_call = "__pyarmor__" in names
    has_runtime = any(n == "pytransform" or re.fullmatch(r"pyarmor_runtime_\d{6}", n) for n in names)
    has_blob = any(isinstance(s, bytes) and (re.match(rb"PY\d{6}\x00", s) or s.startswith(b"PYARMOR"))
                   for s in strings)
    return has_call and has_runtime and has_blob


def detect_text(text: str) -> list[str]:
    found = []
    for name, pats in _COMPILED.items():
        hits = sum(1 for p in pats if p.search(text))
        if hits:
            found.append(name)
    # Generic markers are only interesting when nothing specific matched.
    specific = [f for f in found if f not in ("marshal-exec", "base64/zlib-exec")]
    if specific:
        found = specific
    if pyarmor_evidence_source(text):
        found.append("PyArmor")
    if "pyminifier" not in found and _nonlatin_identifiers(text):
        found.append("pyminifier (non-latin identifiers)")
    return found


_IDENT = re.compile(r"\b[^\W\d][\w]*\b", re.UNICODE)


def _nonlatin_identifiers(text: str) -> bool:
    # many distinct identifiers made of non-ASCII letters => pyminifier --nonlatin
    n = 0
    for m in _IDENT.finditer(text[:200_000]):
        w = m.group(0)
        if not w.isascii() and all(ord(c) > 0x2FF for c in w):
            n += 1
            if n > 25:
                return True
    return False


def detect_code_strings(strings, names=()) -> list[str]:
    text = "\n".join(s if isinstance(s, str) else s.decode("latin-1") for s in strings[:5000])
    text += "\n" + " ".join(names)
    found = [d for d in detect_text(text) if d != "PyArmor"]
    if pyarmor_evidence_code(set(names), strings):
        found.append("PyArmor")
    return found


PYARMOR_NOTE = (
    "PyArmor encrypts bytecode with keys held in its native runtime extension "
    "(pyarmor_runtime*.so/.pyd or _pytransform.*). It cannot be recovered by "
    "static emulation; the encrypted blob and runtime metadata are reported "
    "instead. Use a dedicated static unpacker (e.g. Pyarmor-Static-Unpack-1shot "
    "together with the sample's runtime library) or a controlled dynamic sandbox."
)


def pyarmor_info(data: bytes) -> dict:
    info = {}
    m = re.search(rb"#\s*Pyarmor\s+([\d.]+)\s*(\([^)]*\))?", data)
    if m:
        info["pyarmor_version"] = m.group(1).decode() + (" " + m.group(2).decode() if m.group(2) else "")
    m = re.search(rb"pyarmor_runtime_(\d{6})", data)
    if m:
        info["runtime_package"] = "pyarmor_runtime_" + m.group(1).decode()
        info["generation"] = "8+"
    if b"pytransform" in data:
        info["generation"] = info.get("generation", "<=7 (pytransform)")
    m = re.search(rb"PY(\d{6})\x00", data)
    if m:
        off = m.start()
        hdr = data[off:off + 32]
        info["blob_serial"] = m.group(1).decode()
        if len(hdr) > 10 and hdr[9] == 3 and 6 <= hdr[10] <= 15:
            info["python_version"] = f"3.{hdr[10]}"
    m = re.search(rb"PYARMOR\x00\x00(.)(.)", data, re.S)
    if m:
        info["legacy_header"] = True
        if m.group(1) == b"\x03":
            info["python_version"] = f"3.{m.group(2)[0]}"
    return info

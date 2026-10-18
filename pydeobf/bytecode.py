"""Version-aware bytecode helpers built on xdis.

Everything here is static: marshal data is parsed with xdis' pure-Python
unmarshaller (never the native ``marshal.loads`` on untrusted input) and code
objects are only disassembled, never executed.
"""

from __future__ import annotations

import contextlib
import io
import os
import struct
import sys
from dataclasses import dataclass, field

from xdis import magics
from xdis.bytecode import get_instructions_bytes
from xdis.op_imports import get_opcode_module
from xdis.unmarshal import load_code
from xdis.version_info import PythonImplementation

HOST_VERSION = sys.version_info[:2]

# Versions we try when a marshal blob comes without a pyc header.
CANDIDATE_VERSIONS = [(3, v) for v in range(15, 5, -1)]
# Versions the emulator knows the opcode semantics of.
EMULATED_VERSIONS = {(3, v) for v in range(6, 16)}


def vstr(version) -> str:
    return f"{version[0]}.{version[1]}"


def magic_for(version) -> bytes:
    return magics.by_version[vstr(version)]


def magic_int_for(version) -> int:
    return magics.magic2int(magic_for(version))


_OPC_CACHE: dict = {}


def opcode_module(version):
    key = tuple(version[:2])
    if key not in _OPC_CACHE:
        _OPC_CACHE[key] = get_opcode_module(key, PythonImplementation.CPython)
    return _OPC_CACHE[key]


@dataclass
class CodeBlob:
    """A code object together with the Python version it belongs to.

    ``raw`` is the marshal serialisation of ``code`` (no pyc header) when it is
    known; it is what gets wrapped into a .pyc for PyLingual.
    """

    code: object
    version: tuple
    raw: bytes | None = None
    source_hint: str | None = None  # source text if the code came from compile()
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------- pyc io

def looks_like_pyc(data: bytes) -> bool:
    if len(data) < 16 or data[2:4] != b"\r\n":
        return False
    try:
        return magics.magic2int(data[:4]) in magics.magicint2version
    except Exception:
        return False


def pyc_header_size(version) -> int:
    if version >= (3, 7):
        return 16
    if version >= (3, 3):
        return 12
    return 8


def read_pyc(data: bytes) -> CodeBlob:
    magic_int = magics.magic2int(data[:4])
    vtuple = magics.magic_int2tuple(magic_int)[:2]
    hdr = pyc_header_size(vtuple)
    raw = data[hdr:]
    code = load_code(io.BytesIO(raw), magic_int)
    return CodeBlob(code=code, version=tuple(vtuple), raw=raw, meta={"magic_int": magic_int})


def make_pyc(raw: bytes, version) -> bytes:
    hdr = magic_for(version)
    if version >= (3, 7):
        hdr += struct.pack("<III", 0, 0, len(raw) & 0xFFFFFFFF)
    else:
        hdr += struct.pack("<II", 0, len(raw) & 0xFFFFFFFF)
    return hdr + raw


# -------------------------------------------------------- marshal detection

def unmarshal(raw: bytes, version):
    """Load a header-less marshal blob as ``version``; returns (code, consumed)."""
    fp = io.BytesIO(raw)
    with open(os.devnull, "w") as null, contextlib.redirect_stderr(null), contextlib.redirect_stdout(null):
        code = load_code(fp, magic_int_for(version))
    return code, fp.tell()


def is_code(obj) -> bool:
    return hasattr(obj, "co_code") and hasattr(obj, "co_consts")


def iter_code_objects(code):
    stack = [code]
    while stack:
        c = stack.pop()
        yield c
        for k in c.co_consts:
            if is_code(k):
                stack.append(k)


def _opnames(code, version):
    opc = opcode_module(version)
    names = []
    for c in iter_code_objects(code):
        names.extend(i.opname for i in get_instructions_bytes(c, opc))
    return names


def validate_code(code, version) -> bool:
    """Heuristically check that ``code`` really is ``version`` bytecode."""
    opc = opcode_module(version)
    jump_ops = set(getattr(opc, "JUMP_OPS", None) or getattr(opc, "JUMP_OPs", ()) or ())
    max_cmp = 6 if version >= (3, 9) else 11
    cmp_shift = 5 if version >= (3, 13) else 4 if version >= (3, 12) else 0
    try:
        for c in iter_code_objects(code):
            n = len(c.co_code)
            if n == 0 or n % 2:
                return False
            nconst, nnames = len(c.co_consts), len(c.co_names)
            for ins in get_instructions_bytes(c, opc):
                name = ins.opname
                if name.startswith("<") or name in ("INVALID", "ILLEGAL"):
                    return False
                if name == "LOAD_CONST" and ins.arg is not None and ins.arg >= nconst:
                    return False
                if name in ("LOAD_NAME", "STORE_NAME", "IMPORT_NAME") and ins.arg is not None and ins.arg >= nnames:
                    return False
                if name == "COMPARE_OP" and ins.arg is not None and (ins.arg >> cmp_shift) >= max_cmp:
                    return False
                if ins.opcode in jump_ops and isinstance(ins.argval, int) and not (0 <= ins.argval <= n):
                    return False
    except Exception:
        return False
    return True


def _linetable_spans_code(code) -> bool:
    """3.10's co_linetable byte deltas sum to len(co_code); 3.8/3.9 lnotab doesn't."""
    tbl = getattr(code, "co_linetable", None) or getattr(code, "co_lnotab", None) or b""
    if isinstance(tbl, str):
        tbl = tbl.encode("latin-1")
    return len(tbl) >= 2 and sum(tbl[0::2]) == len(code.co_code)


def _disambiguate(cands: dict, hint):
    """Pick among versions that all load and validate the same blob."""
    vs = set(cands)
    if hint and tuple(hint[:2]) in vs:
        return tuple(hint[:2]), sorted(vs)
    if (3, 10) in vs and vs & {(3, 8), (3, 9)}:
        if _linetable_spans_code(cands[(3, 10)]):
            vs -= {(3, 8), (3, 9)}
        else:
            vs.discard((3, 10))
    if {(3, 6), (3, 7)} <= vs:
        ops = _opnames(cands[(3, 7)], (3, 7))
        if "LOAD_METHOD" in ops:
            vs.discard((3, 6))
        elif "LOAD_ATTR" in ops and "CALL_FUNCTION" in ops:
            vs.discard((3, 7))  # 3.7+ would have compiled obj.m() to LOAD_METHOD
    best = max(vs)
    return best, sorted(vs)


def detect_marshal(raw: bytes, hint=None) -> CodeBlob | None:
    """Find the Python version a header-less marshalled code object belongs to."""
    if not raw or raw[0] not in (0x63, 0xE3):  # TYPE_CODE, optionally FLAG_REF
        return None
    cands = {}
    for v in CANDIDATE_VERSIONS:
        try:
            code, used = unmarshal(raw, v)
        except Exception:
            continue
        if used == len(raw) and is_code(code) and validate_code(code, v):
            cands[v] = code
    if not cands:
        return None
    v, possible = _disambiguate(cands, hint)
    meta = {"detected": True}
    if len(possible) > 1:
        meta["ambiguous"] = [vstr(x) for x in possible]
    return CodeBlob(code=cands[v], version=v, raw=raw, meta=meta)


def load_marshal_best_effort(raw: bytes, hint=None) -> CodeBlob | None:
    """detect_marshal, else any version that parses the whole blob (hint first)."""
    blob = detect_marshal(raw, hint)
    if blob is not None:
        return blob
    order = ([tuple(hint[:2])] if hint else []) + [HOST_VERSION] + CANDIDATE_VERSIONS
    for v in order:
        try:
            code, used = unmarshal(raw, v)
        except Exception:
            continue
        if used == len(raw) and is_code(code):
            return CodeBlob(code=code, version=v, raw=raw, meta={"detected": False, "uncertain": True})
    return None


# ------------------------------------------------------------ instructions

@dataclass(slots=True)
class Ins:
    offset: int
    opname: str
    arg: int | None
    argval: object


_INS_CACHE: dict = {}


def instructions(code, version):
    """Decoded instruction list for ``code`` plus an offset->index map."""
    key = id(code)
    hit = _INS_CACHE.get(key)
    if hit is not None and hit[0] is code:
        return hit[1], hit[2]
    opc = opcode_module(version)
    out = []
    for i in get_instructions_bytes(code, opc):
        if i.opname in ("CACHE", "EXTENDED_ARG"):
            continue
        out.append(Ins(i.offset, i.opname, i.arg, i.argval))
    index = {ins.offset: n for n, ins in enumerate(out)}
    _INS_CACHE[key] = (code, out, index)
    return out, index


def pad_bad_jumps(blob):
    """Pad code objects whose jumps point past the end of their bytecode.

    Truncated or carved code (e.g. a module whose ``if __name__ == "__main__"``
    block lost its epilogue) makes decompilers reject the jump targets.  Each
    such code object gets NOPs up to the furthest target, then ``LOAD_CONST
    None; RETURN_VALUE`` when None is a constant.  Only for Python < 3.10
    (byte-offset jumps, no inline caches).  The blob itself is not modified.

    Returns ``(patched_raw, notes)``, or ``None`` if nothing needed padding.
    """
    if blob.raw is None or blob.version >= (3, 10):
        return None
    opc = opcode_module(blob.version)
    nop, load_const, ret = opc.opmap["NOP"], opc.opmap["LOAD_CONST"], opc.opmap["RETURN_VALUE"]
    raw = blob.raw
    notes = []
    for code in iter_code_objects(blob.code):
        co = bytes(code.co_code)
        far = max((int(i.argval) for i in get_instructions_bytes(code, opc)
                   if i.optype in ("jabs", "jrel") and isinstance(i.argval, int)), default=0)
        if far < len(co):
            continue
        pad = bytes([nop, 0]) * ((far - len(co)) // 2)
        if None in code.co_consts and code.co_consts.index(None) < 256:
            pad += bytes([load_const, code.co_consts.index(None), ret, 0])
        else:
            pad += bytes([nop, 0]) * 2
        patched = co + pad
        new_raw = _replace_co_code(raw, co, patched)
        if new_raw is not None:
            raw = new_raw
            notes.append(f"{code.co_name}: jump target {far} was past the end of its {len(co)} bytes "
                         f"of bytecode; padded with {len(pad)} bytes")
        else:
            notes.append(f"{code.co_name}: jump target {far} past end ({len(co)} bytes) "
                         "but its bytecode could not be located uniquely in the marshal data")
    if raw is blob.raw or raw == blob.raw:
        return None
    return raw, notes


def _replace_co_code(raw, co, patched, top_level=False):
    """Swap one code object's bytecode inside marshal data, or None if it is not found exactly once.

    The top-level code object's bytecode is the first string in its marshal data (only the
    fixed-size int fields come before it), so for it the first match is used even when the
    same bytecode appears again later, e.g. in an embedded copy of the next layer.
    """
    # marshal bytes: TYPE_STRING ('s', or with FLAG_REF 0xf3) + u32 length + data
    for tag in (b"s", b"\xf3"):
        old = tag + struct.pack("<I", len(co)) + co
        new = tag + struct.pack("<I", len(patched)) + patched
        if raw.count(old) == 1:
            return raw.replace(old, new)
        if top_level and 0 <= raw.find(old) < 64:
            return raw.replace(old, new, 1)
    return None


_PURE_LOADS = {"LOAD_CONST", "LOAD_NAME", "LOAD_GLOBAL", "LOAD_FAST", "LOAD_DEREF", "LOAD_CLOSURE"}


def neutralize_bad_makefunc(blob):
    """Rewrite MAKE_FUNCTIONs that have no code object to build a function from.

    Obfuscators plant these in dead branches (``foo = False; if foo: bar =
    <MAKE_FUNCTION of two ints>``) to crash decompilers, which look up the
    function's code object from the preceding LOAD_CONST.  Each one becomes a
    same-size ``BUILD_TUPLE n`` of the n values actually pushed before it.
    Python 3.6-3.12 only (3.13+ splits MAKE_FUNCTION into several ops).  The
    blob itself is not modified.

    Returns ``(patched_raw, notes)``, or ``None`` if nothing needed rewriting.
    """
    if blob.raw is None or not (3, 6) <= blob.version < (3, 13):
        return None
    opc = opcode_module(blob.version)
    build_tuple, ext = opc.opmap["BUILD_TUPLE"], opc.opmap["EXTENDED_ARG"]
    func_slot = 2 if blob.version < (3, 11) else 1  # code object is below the qualname before 3.11
    raw = blob.raw
    notes = []
    for code in iter_code_objects(blob.code):
        ins, _ = instructions(code, blob.version)
        co = bytearray(code.co_code)
        for i, inst in enumerate(ins):
            if inst.opname != "MAKE_FUNCTION":
                continue
            if i >= func_slot and ins[i - func_slot].opname == "LOAD_CONST" and is_code(ins[i - func_slot].argval):
                continue
            expected = func_slot + bin((inst.arg or 0) & 0xF).count("1")
            pushed = 0
            while pushed < expected and i - pushed > 0 and ins[i - pushed - 1].opname in _PURE_LOADS:
                pushed += 1
            off = inst.offset
            if pushed == 0 or (off >= 2 and co[off - 2] == ext):
                notes.append(f"{code.co_name}: MAKE_FUNCTION at {off} has no code object but could not be rewritten")
                continue
            co[off], co[off + 1] = build_tuple, pushed
            notes.append(f"{code.co_name}: MAKE_FUNCTION at {off} has no code object (anti-decompiler junk); "
                         f"rewritten as BUILD_TUPLE {pushed}")
        if co != code.co_code:
            new_raw = _replace_co_code(raw, bytes(code.co_code), bytes(co), top_level=code is blob.code)
            if new_raw is not None:
                raw = new_raw
            else:
                notes.append(f"{code.co_name}: bytecode could not be located uniquely in the marshal data")
    if not notes:
        return None
    return raw, notes


def _varint(data, pos):
    b = data[pos]
    val = b & 63
    pos += 1
    while b & 64:
        b = data[pos]
        pos += 1
        val = (val << 6) | (b & 63)
    return val, pos


def exception_table(code):
    """Parse a 3.11+ ``co_exceptiontable`` into (start, end, target, depth, lasti)."""
    data = getattr(code, "co_exceptiontable", b"") or b""
    out = []
    pos = 0
    try:
        while pos < len(data):
            start, pos = _varint(data, pos)
            length, pos = _varint(data, pos)
            target, pos = _varint(data, pos)
            dl, pos = _varint(data, pos)
            out.append((start * 2, (start + length) * 2, target * 2, dl >> 1, bool(dl & 1)))
    except IndexError:
        pass
    return out


def string_constants(code, min_len=1):
    """All str/bytes constants of ``code`` and its nested code objects."""
    out = []
    for c in iter_code_objects(code):
        stack = list(c.co_consts)
        while stack:
            k = stack.pop()
            if isinstance(k, (str, bytes)) and len(k) >= min_len:
                out.append(k)
            elif isinstance(k, (tuple, frozenset, list)):
                stack.extend(k)
    return out


def code_size_estimate(code) -> int:
    return sum(len(c.co_code) for c in iter_code_objects(code))


def disassemble(code, version, indent=0, out=None):
    """Plain, always-correct disassembly of a code object and its nested code
    objects (used as a reliable fallback when a decompiler produces noise)."""
    if out is None:
        out = []
    pad = "  " * indent
    qn = getattr(code, "co_qualname", None) or code.co_name
    out.append(f"{pad}# code object {qn} (argcount={code.co_argcount})")
    consts = code.co_consts
    nested = []
    for ins in instructions(code, version)[0]:
        if ins.opname in ("RESUME", "CACHE"):
            continue
        arg = "" if ins.arg is None else f"{ins.arg}"
        val = "" if ins.argval is None or ins.argval == ins.arg else f"  ({describe_const(ins.argval)})"
        out.append(f"{pad}  {ins.offset:>5} {ins.opname:<28} {arg}{val}")
    for k in consts:
        if is_code(k):
            out.append("")
            disassemble(k, version, indent + 1, out)
    return "\n".join(out) if indent == 0 else out


def describe_const(v, limit=60):
    r = repr(v)
    return r if len(r) <= limit else r[:limit - 3] + "..."

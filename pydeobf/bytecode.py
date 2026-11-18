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


def patch_xdis_lenient() -> None:
    """Make xdis' disassembly tolerate out-of-range operands in junk bytecode.

    Obfuscators (PyArmor's ``obfuscate code`` mode) plant unreachable garbage
    after an unconditional ``JUMP_FORWARD``.  Those dead bytes decode to
    instructions whose operands point past the real tables.  xdis' ``get_name_info``
    already guards its index, but ``get_const_info`` does not and raises a bare
    ``IndexError`` that aborts the whole disassembly (and, downstream, PyLingual).
    Guard it the same way: for an out-of-range index fall back to the raw index,
    exactly as ``get_name_info`` does.  Reachable code never references an
    out-of-range const, so this only ever affects dead junk.  Idempotent.
    """
    import xdis.bytecode as _xb

    if getattr(_xb.get_const_info, "_pydeobf_lenient", False):
        return
    _orig = _xb.get_const_info

    def get_const_info(const_index, const_list):
        if const_list is not None and not (0 <= const_index < len(const_list)):
            return const_index, repr(const_index)
        return _orig(const_index, const_list)

    get_const_info._pydeobf_lenient = True
    _xb.get_const_info = _xb._get_const_info = get_const_info


patch_xdis_lenient()


def patch_xdis_refs() -> None:
    """Fix two xdis unmarshal bugs that shift its back-reference table.

    * ``t_long`` returns early for a zero-length long without recording the
      reference CPython records (r_PyLong always ends in R_REF), so every later
      TYPE_REF resolves to the wrong object: constants come out swapped
      (``import marshal`` gets a string as its level, keys and blobs change).
    * ``t_object_reference`` treats ref #0 as invalid and returns the last
      object; CPython accepts 0 <= n < len(refs).
    * ``t_string`` decodes TYPE_STRING to str (utf-8, errors ignored) unless the
      caller passes bytes_for_s, which nothing does for values outside a code
      object: ``marshal.loads(marshal.dumps(b"..."))`` came back as a lossy str.
      In Python 3 marshal TYPE_STRING is always bytes.
    Also: big ints come back as xdis' LongTypeForPython3, whose str() carries a
    Python 2 ``L`` suffix and which the emulator's allow-list doesn't know; we only
    read Python 3, so make them plain ints. Idempotent.
    """
    import xdis.unmarshal as _xu
    from xdis.unmarshal import VersionIndependentUnmarshaller as U

    if getattr(U.t_long, "_pydeobf_fixed", False):
        return
    _xu.long = int
    _orig_long = U.t_long

    def t_long(self, save_ref, bytes_for_s=False):
        # 15-bit digits, least significant first. xdis adds them one by one (quadratic: a payload stored as a
        # 500 KB int never finishes); join them as a binary string instead, which int() parses in linear time
        n = struct.unpack("<i", self.fp.read(4))[0]
        if n == 0:
            return self.r_ref(0, save_ref)
        size = abs(n)
        raw = self.fp.read(2 * size)
        if len(raw) < 2 * size:
            raise EOFError("marshal data too short (long)")
        digits = struct.unpack(f"<{size}h", raw)
        d = int("".join(format(x & 0x7FFF, "015b") for x in reversed(digits)), 2)
        return self.r_ref(-d if n < 0 else d, save_ref)

    def t_object_reference(self, save_ref=None, bytes_for_s=False):
        refnum = self.read_uint32()
        if not (0 <= refnum < len(self.intern_objects)):
            return self.intern_objects[-1]
        return self.intern_objects[refnum]

    _orig_string = U.t_string

    def t_string(self, save_ref, bytes_for_s=False):
        return _orig_string(self, save_ref, bytes_for_s or tuple(self.version_triple[:1]) >= (3,))

    # * before 3.11 CPython marshals co_varnames, co_freevars, co_cellvars; some xdis
    #   releases read cellvars first, so the two come out swapped (closures over a
    #   variable that is both captured and passed on bind to the wrong cell). Whether
    #   this xdis does is probed once (see _xdis_swaps_cells), so a fixed xdis is left alone.
    _orig_code = U.t_code

    def t_code(self, save_ref, bytes_for_s=False):
        co = _orig_code(self, save_ref, bytes_for_s)
        if (2, 1) <= tuple(self.version_triple[:2]) < (3, 11) and hasattr(co, "co_cellvars") \
                and _xdis_swaps_cells():
            co.co_cellvars, co.co_freevars = co.co_freevars, co.co_cellvars
        return co

    # xdis grows tuples / sets with ``ret += (item,)``: quadratic, so a payload holding a tuple of a million ints
    # never finishes. Collect into a list and build once.
    def _seq(self, n, save_ref, bytes_for_s, make):
        _, i = self.r_ref_reserve(make(()), save_ref)
        items = [self.r_object(bytes_for_s=bytes_for_s) for _ in range(n)]
        return self.r_ref_insert(make(items), i)

    def t_tuple(self, save_ref, bytes_for_s=False):
        return _seq(self, self.read_uint32(), save_ref, bytes_for_s, tuple)

    def t_small_tuple(self, save_ref, bytes_for_s=False):
        return _seq(self, struct.unpack("B", self.fp.read(1))[0], save_ref, bytes_for_s, tuple)

    def t_set(self, save_ref, bytes_for_s=False):
        return _seq(self, self.read_uint32(), save_ref, bytes_for_s, set)

    t_long._pydeobf_fixed = True
    U.t_string = t_string
    U.t_object_reference = t_object_reference
    U.t_code = t_code
    # the linear readers below parse the marshal format themselves and only use these reader primitives;
    # if a future xdis renames them, keep xdis' own (slow but correct) readers
    if all(hasattr(U, a) for a in ("r_ref", "r_ref_reserve", "r_ref_insert", "r_object", "read_uint32")):
        fast_long = t_long

        def t_long(self, save_ref, bytes_for_s=False):
            return fast_long(self, save_ref, bytes_for_s) if hasattr(self, "fp") else \
                _orig_long(self, save_ref, bytes_for_s)
        t_long._pydeobf_fixed = True
        U.t_long = t_long
        U.t_tuple = t_tuple
        U.t_small_tuple = t_small_tuple
        U.t_set = t_set
    else:
        U.t_long = _mark_fixed(_orig_long)


def _mark_fixed(fn):
    def wrapped(self, *a, **k):
        return fn(self, *a, **k)
    wrapped._pydeobf_fixed = True
    return wrapped


_SWAPS_CELLS = None


def _xdis_swaps_cells() -> bool:
    """Does this xdis read a <= 3.10 code object's co_cellvars / co_freevars swapped? Probed once on a code
    object written here byte by byte (cellvars=('c',), freevars=('f',)), so the fix-up follows xdis
    rather than assuming its bug."""
    global _SWAPS_CELLS
    if _SWAPS_CELLS is None:
        _SWAPS_CELLS = False  # while probing, and if the probe cannot tell
        i32 = lambda v: struct.pack("<i", v)
        name = lambda t: b"z" + bytes([len(t)]) + t.encode()
        tup = lambda *xs: b")" + bytes([len(xs)]) + b"".join(xs)
        raw = (b"c" + i32(0) + i32(0) + i32(0) + i32(1) + i32(1) + i32(0)          # 3.8: argcount .. flags
               + b"s" + i32(4) + b"d\x00S\x00"                                     # LOAD_CONST 0; RETURN_VALUE
               + tup(b"N") + tup() + tup(name("v"))                                 # consts, names, varnames
               + tup(name("f")) + tup(name("c"))                                    # freevars, cellvars
               + name("p") + name("p") + i32(1) + b"s" + i32(0))                    # filename, name, line, lnotab
        try:
            co, _ = unmarshal(raw, (3, 8))
            _SWAPS_CELLS = tuple(co.co_cellvars) == ("f",) and tuple(co.co_freevars) == ("c",)
        except Exception:
            pass
    return _SWAPS_CELLS


patch_xdis_refs()


_OPC_CACHE: dict = {}


def _lenient_fmt(fn):
    """Wrap an xdis argrepr formatter so junk operands can't abort disassembly.

    xdis' arg formatters are purely cosmetic (they build the ``# ...`` comment),
    but several of them ``assert`` on operand ranges, e.g. ``format_RAISE_VARARGS``
    requires ``0 <= argc <= 2``.  Obfuscators plant unreachable junk bytes (after
    a ``JUMP_FORWARD`` over them) that decode to out-of-range operands, so a plain
    linear disassembly of the whole ``co_code`` trips the assert and raises a bare
    ``AssertionError``.  We never use the argrepr, so swallow any failure.
    """
    def wrapped(arg):
        try:
            return fn(arg)
        except Exception:
            return None
    return wrapped


class _Labels(list):
    """A list (what xdis expects from findlabels) whose ``in`` is a set lookup: xdis tests ``offset in labels``
    for every instruction, which is quadratic on a huge function."""

    def __init__(self, items=()):
        super().__init__(items)
        self._set = set(self)

    def __contains__(self, x):
        return x in self._set

    def append(self, x):
        super().append(x)
        self._set.add(x)

    def extend(self, xs):
        xs = list(xs)
        super().extend(xs)
        self._set.update(xs)


def _fast_findlabels(orig):
    """xdis's findlabels de-duplicates with ``label not in list``: quadratic on a function with 100k jumps.
    Linear, and computed from the opcode tables here (backward jumps included). Falls back to xdis' own
    if the helpers it borrows ever move."""
    try:
        from xdis.cross_dis import get_cache_size_313, unpack_opargs_bytecode_310
        from xdis.wordcode import unpack_opargs_wordcode
    except ImportError:
        return orig

    def findlabels(code, opc):
        v = tuple(opc.version_tuple[:2])
        if v < (3, 6):
            return orig(code, opc)
        unpack = unpack_opargs_wordcode if v < (3, 10) else unpack_opargs_bytecode_310
        labels = {}
        for offset, op, arg in unpack(code, opc):
            if arg is None:
                continue
            if op in opc.JREL_OPS:
                if v >= (3, 11) and opc.opname[op] in ("JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT"):
                    arg = -arg
                label = offset + 2 + (arg * 2 if v >= (3, 10) else arg)
                if v >= (3, 13):
                    label += 2 * get_cache_size_313(opc.opname[op])
            elif op in opc.JABS_OPS:
                label = arg * 2 if v >= (3, 10) else arg
            else:
                continue
            labels[label] = None
        return _Labels(labels)
    findlabels._orig = orig
    return findlabels


def opcode_module(version):
    key = tuple(version[:2])
    if key not in _OPC_CACHE:
        opc = get_opcode_module(key, PythonImplementation.CPython)
        if getattr(opc, "findlabels", None) is not None and not hasattr(opc.findlabels, "_orig"):
            opc.findlabels = _fast_findlabels(opc.findlabels)
        fmt = getattr(opc, "opcode_arg_fmt", None)
        if isinstance(fmt, dict):
            for name, fn in list(fmt.items()):
                if callable(fn):
                    fmt[name] = _lenient_fmt(fn)
        _OPC_CACHE[key] = opc
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
    try:
        code = load_code(io.BytesIO(raw), magic_int)
    except Exception:
        code = None
    if code is None or not validate_code(code, vtuple):
        # extracted pycs (e.g. pyinstxtractor) can carry the magic of the wrong Python version
        blob = detect_marshal(raw, tuple(vtuple))
        if blob is not None and blob.version != tuple(vtuple):
            blob.meta.update(magic_int=magic_int, header_version=vstr(vtuple))
            return blob
        if code is None:
            raise ValueError(f"cannot unmarshal .pyc as Python {vstr(vtuple)} (header magic {magic_int})")
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


# types.CodeType(...) positional fields, by the Python version whose bytecode makes the call
CODETYPE_FIELDS = {
    (3, 11): ("argcount", "posonlyargcount", "kwonlyargcount", "nlocals", "stacksize", "flags", "code", "consts",
              "names", "varnames", "filename", "name", "qualname", "firstlineno", "linetable", "exceptiontable",
              "freevars", "cellvars"),
    (3, 8): ("argcount", "posonlyargcount", "kwonlyargcount", "nlocals", "stacksize", "flags", "code", "consts",
             "names", "varnames", "filename", "name", "firstlineno", "linetable", "freevars", "cellvars"),
    (3, 7): ("argcount", "kwonlyargcount", "nlocals", "stacksize", "flags", "code", "consts", "names", "varnames",
             "filename", "name", "firstlineno", "linetable", "freevars", "cellvars"),
}


def codetype_fields(version):
    return CODETYPE_FIELDS[(3, 11) if version >= (3, 11) else (3, 8) if version >= (3, 8) else (3, 7)]


def dump_code(f: dict, version, dump_const) -> bytes:
    """Marshal a code object for ``version`` from its CodeType fields (no pyc header).
    ``dump_const`` serialises one constant (nested code objects included)."""
    i32 = lambda v: struct.pack("<i", v)
    obj = dump_const
    names = lambda seq: dump_const(tuple(seq))
    out = bytearray(b"c")
    if version >= (3, 11):
        # locals-plus: arguments/locals (0x20), cells (0x40, or 0x60 for a cell argument), frees (0x80)
        lp, kinds = list(f["varnames"]), [0x20] * len(f["varnames"])
        for c in f["cellvars"]:
            if c in lp:
                kinds[lp.index(c)] |= 0x40
            else:
                lp.append(c); kinds.append(0x40)
        lp += list(f["freevars"]); kinds += [0x80] * len(f["freevars"])
        out += i32(f["argcount"]) + i32(f["posonlyargcount"]) + i32(f["kwonlyargcount"]) + i32(f["stacksize"]) \
            + i32(f["flags"])
        out += obj(bytes(f["code"])) + obj(f["consts"]) + names(f["names"]) + names(lp) + obj(bytes(kinds))
        out += obj(f["filename"]) + obj(f["name"]) + obj(f["qualname"]) + i32(f["firstlineno"])
        out += obj(bytes(f["linetable"])) + obj(bytes(f["exceptiontable"]))
    else:
        out += i32(f["argcount"])
        if version >= (3, 8):
            out += i32(f["posonlyargcount"])
        out += i32(f["kwonlyargcount"]) + i32(f["nlocals"]) + i32(f["stacksize"]) + i32(f["flags"])
        out += obj(bytes(f["code"])) + obj(f["consts"]) + names(f["names"]) + names(f["varnames"])
        out += names(f["freevars"]) + names(f["cellvars"]) + obj(f["filename"]) + obj(f["name"])
        out += i32(f["firstlineno"]) + obj(bytes(f["linetable"]))
    return bytes(out)


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
    try:
        return all(_valid_one(c, version) for c in iter_code_objects(code))
    except Exception:
        return False


def _valid_one(c, version) -> bool:
    """validate_code for one code object, not its nested ones."""
    opc = opcode_module(version)
    jump_ops = set(getattr(opc, "JUMP_OPS", None) or getattr(opc, "JUMP_OPs", ()) or ())
    max_cmp = 6 if version >= (3, 9) else 11
    cmp_shift = 5 if version >= (3, 13) else 4 if version >= (3, 12) else 0
    try:
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


def trim_junk(code, version):
    """Drop anti-decompiler padding (random bytes after the last RETURN_VALUE, often an odd count) from every
    code object, 3.6-3.10 wordcode only. CPython never executes it. None when that doesn't make it valid."""
    if not (3, 6) <= tuple(version[:2]) <= (3, 10):
        return None
    ret = opcode_module(version).opmap.get("RETURN_VALUE")
    consts, changed = [], False
    for k in code.co_consts:
        if is_code(k) and not validate_code(k, version):
            k = trim_junk(k, version)
            if k is None:
                return None
            changed = True
        consts.append(k)
    c = code.replace(co_consts=tuple(consts)) if changed else code
    if _valid_one(c, version):
        return c
    co = bytes(c.co_code)
    for p in range((len(co) - 2) & ~1, -1, -2):
        if co[p] == ret:
            t = c.replace(co_code=co[:p + 2])
            if _valid_one(t, version):
                return t
    return None


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
    trimmed = False
    if not cands:  # junk-padded bytecode: valid once the bytes after the last return are dropped
        for v in CANDIDATE_VERSIONS:
            try:
                code, used = unmarshal(raw, v)
                t = trim_junk(code, v) if used == len(raw) and is_code(code) else None
            except Exception:
                continue
            if t is not None:
                cands[v] = t
        trimmed = bool(cands)
    if not cands:
        return None
    v, possible = _disambiguate(cands, hint)
    meta = {"detected": True}
    if trimmed:
        # the decompiler reads the marshal bytes, so cut the padding there too
        orig, _ = unmarshal(raw, v)
        for old, new in zip(iter_code_objects(orig), iter_code_objects(cands[v])):
            if len(old.co_code) != len(new.co_code):
                raw = _replace_co_code(raw, bytes(old.co_code), bytes(new.co_code)) or raw
        meta["junk_trimmed"] = True
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


_LINE_CACHE: dict = {}


def line_at(code, version, offset):
    """Source line of the instruction at ``offset`` (inspect.stack() / f_lineno as CPython reports it)."""
    hit = _LINE_CACHE.get(id(code))
    if hit is None or hit[0] is not code:
        starts = [(i.offset, i.starts_line) for i in get_instructions_bytes(code, opcode_module(version))
                  if isinstance(i.starts_line, int)]
        hit = _LINE_CACHE[id(code)] = (code, starts)
    line = getattr(code, "co_firstlineno", 0)
    for off, ln in hit[1]:
        if off > offset:
            break
        line = ln
    return line


def instructions(code, version):
    """Decoded instruction list for ``code`` plus an offset->index map."""
    key = id(code)
    hit = _INS_CACHE.get(key)
    if hit is not None and hit[0] is code:
        return hit[1], hit[2]
    opc = opcode_module(version)
    out = []
    index = {}
    pending = []  # offsets of skipped prefixes, which run as part of the next instruction
    for i in get_instructions_bytes(code, opc):
        if i.opname in ("CACHE", "EXTENDED_ARG"):
            pending.append(i.offset)
            continue
        # junk-jump obfuscation targets the EXTENDED_ARG in front of an instruction
        for off in pending:
            index.setdefault(off, len(out))
        pending.clear()
        index[i.offset] = len(out)
        out.append(Ins(i.offset, i.opname, i.arg, i.argval))
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
        try:
            ins, _ = instructions(code, blob.version)
        except Exception:
            # Unreachable anti-disassembly junk (jumped over at runtime) can decode
            # to operands xdis' linear sweep rejects (bad cmp_op/name indexes, etc.).
            # This pass is best-effort; skip a code object we cannot cleanly decode.
            continue
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


def keyed_blobs(code, min_blob, max_keys=24):
    """(blob, candidate keys) per code object: its big str/bytes constants, each with the short ones
    (4..256 chars) of the same code object, then of the module, then one-byte int constants, as XOR keys."""
    def strs(c):
        return [k for k in c.co_consts if isinstance(k, (str, bytes)) and 4 <= len(k) <= 256]

    module_keys = strs(code)
    pairs = []
    for c in iter_code_objects(code):
        big = [k for k in c.co_consts if isinstance(k, (str, bytes)) and len(k) >= min_blob and len(k) > 256]
        if big:
            ints = [k for k in c.co_consts if type(k) is int and 1 <= k <= 255]
            keys = list(dict.fromkeys(strs(c) + module_keys))[:max_keys] + list(dict.fromkeys(ints))[:16]
            pairs.extend((b, keys) for b in big)
    return pairs


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

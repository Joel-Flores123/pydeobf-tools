"""Static PyArmor <= 7 (pytransform) unpacking. Nothing is executed.

Target stub:

    from pytransform import pyarmor_runtime
    pyarmor_runtime()
    __pyarmor__(__name__, __file__, b'PYARMOR\\x00\\x00\\x03\\x08\\x00...', 2)

Blob header (little-endian):

    0   b'PYARMOR\\0\\0'
    9   Python major, minor
    12  .pyc magic
    16  runtime build tag
    28  payload offset, 32 payload length
    36  flags (top byte: restrict mode | entry bits)
    40  16 random bytes (key/IV material)

Wrap-mode code objects look like:

        LOAD_GLOBAL   __armor_enter__
        CALL_FUNCTION 0
        POP_TOP
        SETUP_FINALLY -> footer
        <body, possibly encrypted>
    footer:
        LOAD_GLOBAL   __armor_exit__
        CALL_FUNCTION 0
        POP_TOP
        END_FINALLY
"""

from __future__ import annotations

import ast
import keyword
import struct
from dataclasses import dataclass

from . import bytecode as bc

MAGIC = b"PYARMOR"
ARMOR_NAMES = ("__armor_enter__", "__armor_exit__", "__armor__", "__armor_wrap__")

NOTE_ENCRYPTED = (
    "PyArmor <= 7: payload is AES-128-GCM encrypted and no known capsule key decrypts it "
    "(likely a registered build with a per-user key). Only the header is reported."
)

# ------------------------------------------------------------- decryption
#
# Module payload: AES-128-GCM.  Key = capsule secret FIX ^ the header's random
# dwords (each offset by _KEY_ADJ); IV = header[40:52]; tag not checked.
# All trial builds (PyArmor 5.9.8-7.7.4) share one FIX; registered builds don't.

TRIAL_FIX = bytes.fromhex("6e9ca78879bf7d8e050d4afdd67f3ee7")

KNOWN_FIX: dict[str, bytes] = {"trial": TRIAL_FIX}

# ------------------------------------------------- obf_code=1 body cipher
#
# Wrap-mode bodies are XORed dword-wise with a 6-dword keystream (INNER_CONST
# mixed with the 16-byte nonce at co_code[-16:]), then offset by _INNER_ADD.
# Trailing len%4 bytes are left as-is.

INNER_CONST = bytes.fromhex("d67f3ee7b092982ab7fae0219c67ff5df428feb2f46d20e6")
_INNER_NONCE_ADJ = (0, -0x0B35, 0xD6AE, 0xE9C3)
_INNER_ADD = 0xDD15


def decrypt_body(body: bytes, nonce: bytes, const: bytes = INNER_CONST) -> bytes:
    """Decrypt an obf_code=1 body with its 16-byte nonce."""
    nd = len(body) // 4
    if nd == 0 or len(nonce) < 16:
        return body
    kb = list(struct.unpack("<6I", const))
    nw = struct.unpack("<4I", nonce[:16])
    for i in range(4):
        kb[i] ^= (nw[i] + _INNER_NONCE_ADJ[i]) & _U32
    cw = struct.unpack("<%dI" % nd, body[:nd * 4])
    out = bytes().join(struct.pack("<I", ((cw[i] ^ kb[i % 6]) + _INNER_ADD) & _U32) for i in range(nd))
    return out + body[nd * 4:]

_KEY_ADJ = (0, -0x3B22, 0x802F, 0x251A)
_U32 = 0xFFFFFFFF

# ------------------------------------------------- obf_code=2 body cipher
#
# Used by wrap_mode=0 ``__armor__`` functions: AES-128-CTR with a little-endian
# counter starting at the nonce.  Key derived like derive_key(), from OC2_CONST.
OC2_CONST = bytes.fromhex("a80b4e575d23298734c01984a56e9ca7")
_OC2_KEY_ADJ = (0, -0xF275, 0xB0B0, 0xCD59)


def oc2_key(nonce: bytes, const: bytes = OC2_CONST) -> bytes:
    """AES key for an obf_code=2 body."""
    c = list(struct.unpack("<4I", const))
    n = struct.unpack("<4I", nonce[:16])
    return struct.pack("<4I", *[(c[i] ^ ((n[i] + _OC2_KEY_ADJ[i]) & _U32)) & _U32 for i in range(4)])


def decrypt_ctr(body: bytes, nonce: bytes, const: bytes = OC2_CONST) -> bytes:
    """Decrypt an obf_code=2 body (AES-128-CTR)."""
    from Crypto.Cipher import AES
    ec = AES.new(oc2_key(nonce, const), AES.MODE_ECB)
    ctr = int.from_bytes(nonce[:16], "little")
    mask = (1 << 128) - 1
    ks = b"".join(ec.encrypt(((ctr + j) & mask).to_bytes(16, "little"))
                  for j in range((len(body) + 15) // 16))
    return bytes(a ^ b for a, b in zip(body, ks))


def derive_key(rand16: bytes, fix: bytes) -> bytes:
    """Payload AES key from the header's random bytes and a FIX."""
    r = struct.unpack("<4I", rand16)
    f = struct.unpack("<4I", fix)
    k = [(f[i] ^ ((r[i] + _KEY_ADJ[i]) & _U32)) for i in range(4)]
    return struct.pack("<4I", *k)


def decrypt_payload(blob: bytes, hdr: "Header", fix: bytes) -> bytes | None:
    """Decrypt the payload with ``fix``, or None."""
    from Crypto.Cipher import AES
    rand = blob[40:56]
    ct = blob[hdr.payload_offset:hdr.payload_offset + hdr.payload_length]
    if len(rand) < 16 or not ct:
        return None
    try:
        key = derive_key(rand, fix)
        return AES.new(key, AES.MODE_GCM, nonce=rand[:12]).decrypt(ct)
    except Exception:
        return None


def decrypted_module(blob: bytes, hdr: "Header", fixes: dict | None = None):
    """(CodeBlob, fix_name) for the first key that yields a valid module, else None."""
    for name, fix in (fixes or KNOWN_FIX).items():
        pt = decrypt_payload(blob, hdr, fix)
        if not pt or pt[0] not in (0x63, 0xE3):
            continue
        try:
            code, used = bc.unmarshal(pt, hdr.version)
        except Exception:
            continue
        if used == len(pt) and bc.is_code(code):
            return bc.CodeBlob(code=code, version=hdr.version, raw=pt,
                               meta={"pyarmor7": True, "capsule_key": name}), name
    return None

# top flags byte (minus entry/module-dict bits) -> restrict mode
_RESTRICT = {0x00: 0, 0x10: 1, 0x70: 2, 0xF0: 3, 0xB0: 4, 0xB4: 5}

CO_VARARGS, CO_VARKEYWORDS = 0x04, 0x08
CO_COROUTINE, CO_ASYNC_GENERATOR = 0x80, 0x200


# ---------------------------------------------------------------- detection

def find_blob_source(source: str):
    """(blob, last_arg) from a ``__pyarmor__(...)`` call in source, or None."""
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__pyarmor__":
            blob = next((a.value for a in node.args if isinstance(a, ast.Constant)
                         and isinstance(a.value, bytes) and a.value.startswith(MAGIC)), None)
            if blob is not None:
                last = node.args[-1]
                mode = last.value if isinstance(last, ast.Constant) and isinstance(last.value, int) else None
                return blob, mode
    return None


def find_blob_code(code):
    """The PYARMOR blob from a compiled stub's constants."""
    for s in bc.string_constants(code, 64):
        if isinstance(s, bytes) and s.startswith(MAGIC):
            return s
    return None


# ------------------------------------------------------------------- header

@dataclass
class Header:
    version: tuple
    magic_ok: bool
    runtime_tag: str
    payload_offset: int
    payload_length: int
    flags: int

    @property
    def restrict_mode(self):
        top = self.flags >> 24
        mode = _RESTRICT.get(top & ~0x0A)
        if mode is not None and top & 2:
            mode += 100
        return mode

    @property
    def entry(self) -> bool:
        return bool((self.flags >> 24) & 8)

    def info(self) -> dict:
        d = {"generation": "<=7 (pytransform)", "python_version": bc.vstr(self.version),
             "runtime_tag": self.runtime_tag, "flags": f"{self.flags:#010x}", "entry_script": self.entry}
        if self.restrict_mode is not None:
            d["restrict_mode"] = self.restrict_mode
        if not self.magic_ok:
            d["header_magic_mismatch"] = True
        return d


def parse_header(blob: bytes) -> Header:
    if not blob.startswith(MAGIC) or len(blob) < 64:
        raise ValueError("not a PyArmor <= 7 blob")
    version = (blob[9], blob[10])
    try:
        magic_ok = bc.magic_for(version) == blob[12:16]
    except KeyError:
        magic_ok = False
    off, ln = struct.unpack_from("<II", blob, 28)
    return Header(version, magic_ok, blob[16:20].hex(), off, ln, struct.unpack_from("<I", blob, 36)[0])


def plain_module(blob: bytes, hdr: Header) -> bc.CodeBlob | None:
    """The module code if the payload is unencrypted (--obf-mod 0)."""
    payload = blob[hdr.payload_offset:hdr.payload_offset + hdr.payload_length]
    if not payload or payload[0] not in (0x63, 0xE3):
        return None
    try:
        code, used = bc.unmarshal(payload, hdr.version)
    except Exception:
        return None
    if used != len(payload) or not bc.is_code(code):
        return None
    return bc.CodeBlob(code=code, version=hdr.version, raw=payload, meta={"pyarmor7": True})


# ---------------------------------------------------- wrap-mode stripping

def _cells(co: bytes, opc):
    """Decode wordcode to [(start, end, op, arg)], folding EXTENDED_ARG."""
    ext = opc.opmap["EXTENDED_ARG"]
    out, i, arg, start = [], 0, 0, None
    while i + 1 < len(co):
        op = co[i]
        if start is None:
            start = i
        arg = (arg << 8) | co[i + 1] if op >= opc.HAVE_ARGUMENT else 0
        i += 2
        if op == ext:
            continue
        out.append((start, i, op, arg if op >= opc.HAVE_ARGUMENT else None))
        arg, start = 0, None
    return out


def _unit(version):
    return 2 if version >= (3, 10) else 1  # 3.10 jump args count instructions


def find_wrapper(code, version):
    """(body_start, body_end) of a wrap-mode code object, or None."""
    if version < (3, 6) or "__armor_enter__" not in code.co_names:
        return None
    opc = bc.opcode_module(version)
    co = bytes(code.co_code)
    names = code.co_names
    cells = _cells(co, opc)
    op = opc.opmap
    for k, (_, _, o, a) in enumerate(cells):
        if not (o == op["LOAD_GLOBAL"] and a is not None and a < len(names) and names[a] == "__armor_enter__"):
            continue
        if k + 2 >= len(cells) or cells[k + 1][2] != op["CALL_FUNCTION"] or cells[k + 2][2] != op["POP_TOP"]:
            return None
        j = k + 3
        while j < len(cells) and cells[j][2] == op["NOP"]:
            j += 1
        if j >= len(cells) or cells[j][2] != op["SETUP_FINALLY"]:
            return None
        start = cells[j][1]
        end = start + cells[j][3] * _unit(version)
        footer = next((c for c in cells if c[0] == end), None)
        if footer is None or footer[2] != op["LOAD_GLOBAL"] or footer[3] >= len(names) \
                or names[footer[3]] != "__armor_exit__":
            return None
        return start, end
    return None


def body_is_plain(code, version, start, end, co=None, check_jumps=True) -> bool:
    """True if co[start:end] looks like valid (decrypted) bytecode.
    ``co`` defaults to code.co_code; ``check_jumps`` also bounds-checks jumps."""
    opc = bc.opcode_module(version)
    co = bytes(code.co_code) if co is None else bytes(co)
    unit = _unit(version)
    nconst, nname = len(code.co_consts), len(code.co_names)
    nlocal, nfree = len(code.co_varnames), len(code.co_cellvars) + len(code.co_freevars)
    cells = _cells(co[start:end], opc)
    if not cells:
        return False
    for s, e, o, a in cells:
        if opc.opname[o].startswith("<"):
            return False
        if a is None:
            continue
        if (o in opc.hasconst and a >= nconst) or (o in opc.hasname and a >= nname) \
                or (o in opc.haslocal and a >= nlocal) or (o in opc.hasfree and a >= nfree):
            return False
        if check_jumps:
            if o in opc.hasjabs and not start <= a * unit <= len(co):
                return False
            if o in opc.hasjrel and not e + a * unit <= len(co) - start:
                return False
    return True


def inner_decrypt(code, version):
    """co_code with the wrap-mode body decrypted in place (wrapper kept), or None."""
    span = find_wrapper(code, version)
    if span is None:
        return None
    start, end = span
    co = bytes(code.co_code)
    if body_is_plain(code, version, start, end, co, check_jumps=False):
        return None  # already plain
    dec = decrypt_body(co[start:end], co[-16:])
    cand = co[:start] + dec + co[end:]
    if not body_is_plain(code, version, start, end, cand, check_jumps=False):
        return None
    return cand


def strip_wrapper(code, version):
    """Original bytecode of a wrap-mode code object with the wrapper removed.

    Decrypts the body if needed, rebases in-body absolute jumps, and turns
    jumps out of the body into RETURN_VALUE.  Returns None if a conditional
    jump leaves the body."""
    span = find_wrapper(code, version)
    if span is None:
        return None
    start, end = span
    co = bytes(code.co_code)
    if not body_is_plain(code, version, start, end, co, check_jumps=False):
        dec = decrypt_body(co[start:end], co[-16:])
        cand = co[:start] + dec + co[end:]
        if not body_is_plain(code, version, start, end, cand, check_jumps=False):
            return None
        co = cand
    opc = bc.opcode_module(version)
    unit = _unit(version)
    ret, nop, jabs = opc.opmap["RETURN_VALUE"], opc.opmap["NOP"], opc.opmap["JUMP_ABSOLUTE"]
    body = bytearray(co[start:end])
    for s, e, o, a in _cells(bytes(body), opc):
        if a is None or o not in opc.hasjabs:
            continue
        target = a * unit
        if target < start or target >= end:  # exit via footer == return
            if o != jabs:
                return None
            body[s:e] = bytes([nop, 0]) * ((e - s) // 2 - 1) + bytes([ret, 0])
            continue
        new = (target - start) // unit
        for p in range(e - 1, s, -2):  # arg bytes, lowest first
            body[p] = new & 0xFF
            new >>= 8
    return bytes(body)


def find_armor_stub(code, version):
    """(body, nonce) of a wrap_mode=0 ``__armor__`` code object, or None.

    Layout: JUMP_FORWARD over the encrypted body to
    ``LOAD_GLOBAL __armor__; CALL_FUNCTION 0; POP_TOP; JUMP_ABSOLUTE 0``,
    followed by a 16-byte nonce."""
    if version < (3, 6) or "__armor__" not in code.co_names:
        return None
    opc = bc.opcode_module(version)
    op = opc.opmap
    co = bytes(code.co_code)
    if len(co) < 2 + 8 + 16 or co[0] not in (op["JUMP_FORWARD"], op["EXTENDED_ARG"]):
        return None
    i, ext = 0, 0
    while i < len(co) and co[i] == op["EXTENDED_ARG"]:
        ext = (ext << 8) | co[i + 1]
        i += 2
    if i >= len(co) or co[i] != op["JUMP_FORWARD"]:
        return None
    arg = (ext << 8) | co[i + 1]
    jsize = i + 2
    stub = jsize + arg * _unit(version)
    names = code.co_names
    if not (stub + 8 <= len(co)
            and co[stub] == op["LOAD_GLOBAL"] and co[stub + 1] < len(names)
            and names[co[stub + 1]] == "__armor__"
            and co[stub + 2] == op["CALL_FUNCTION"]
            and co[stub + 4] == op["POP_TOP"]
            and co[stub + 6] == op["JUMP_ABSOLUTE"]):
        return None
    return co[jsize:stub], co[-16:]


def recover_armor_stub(code, version):
    """Decrypted co_code of a wrap_mode=0 ``__armor__`` code object, or None.
    Tries both obf_code=1 and obf_code=2 ciphers."""
    span = find_armor_stub(code, version)
    if span is None:
        return None
    body, nonce = span
    for dec in (decrypt_body(body, nonce), decrypt_ctr(body, nonce)):
        if len(dec) == len(body) and body_is_plain(code, version, 0, len(dec), co=dec, check_jumps=True):
            return dec
    return None


def unwrap(blob: bc.CodeBlob):
    """Recover the bytecode of every wrapped code object in the module.

    Returns (new_raw | None, recovered, encrypted, notes), where ``encrypted``
    counts code objects that could not be recovered."""
    raw = blob.raw
    recovered = encrypted = 0
    notes = []
    for code in bc.iter_code_objects(blob.code):
        wrapped = find_wrapper(code, blob.version) is not None
        armor = find_armor_stub(code, blob.version) is not None
        if not wrapped and not armor:
            continue
        if wrapped:
            new = strip_wrapper(code, blob.version) or inner_decrypt(code, blob.version)
        else:
            new = recover_armor_stub(code, blob.version)
        if new is None:
            encrypted += 1
            continue
        out = bc._replace_co_code(raw, bytes(code.co_code), new) if raw is not None else None
        if out is None:
            notes.append(f"{code.co_name}: recovered bytecode could not be written back (not unique in marshal)")
            continue
        raw = out
        recovered += 1
    return (raw if recovered else None), recovered, encrypted, notes


# --------------------------------------------------------------- skeleton

def _ident(name, fallback):
    if isinstance(name, str) and name.isidentifier() and not keyword.iskeyword(name):
        return name
    return fallback


def _const_text(v, limit=300):
    if v is None or v is Ellipsis or bc.is_code(v):
        return None
    if isinstance(v, (tuple, frozenset)) and any(bc.is_code(x) for x in v):
        return None
    t = repr(v)
    return t if len(t) <= limit else t[:limit] + "..."


def _doc(code, extra=()):
    names = [n for n in code.co_names if n not in ARMOR_NAMES]
    consts = [t for t in (_const_text(c) for c in code.co_consts) if t is not None]
    lines = list(extra)
    if names:
        lines.append("names: " + ", ".join(names))
    if consts:
        lines.append("constants:")
        lines += [f"    {t}" for t in consts[:200]]
        if len(consts) > 200:
            lines.append(f"    ... {len(consts) - 200} more")
    return "\n".join(lines) or "(no names or constants)"


def _arguments(code):
    vn = list(code.co_varnames)
    pos = code.co_argcount
    posonly = getattr(code, "co_posonlyargcount", 0) or 0
    kwonly = code.co_kwonlyargcount
    def arg(i):
        return ast.arg(arg=_ident(vn[i] if i < len(vn) else None, f"arg{i}"))
    i = pos + kwonly
    vararg = kwarg = None
    if code.co_flags & CO_VARARGS:
        vararg, i = arg(i), i + 1
    if code.co_flags & CO_VARKEYWORDS:
        kwarg = arg(i)
    return ast.arguments(posonlyargs=[arg(k) for k in range(posonly)],
                         args=[arg(k) for k in range(posonly, pos)],
                         vararg=vararg, kwonlyargs=[arg(k) for k in range(pos, pos + kwonly)],
                         kw_defaults=[None] * kwonly, kwarg=kwarg, defaults=[])


def _is_class_body(code):
    return code.co_argcount == 0 and "__module__" in code.co_names and "__qualname__" in code.co_names


def _node(code, counter):
    kids = [_node(c, counter) for c in code.co_consts if bc.is_code(c)]
    body = [ast.Expr(ast.Constant(_doc(code)))] + kids + ([] if kids else [ast.Expr(ast.Constant(...))])
    raw_name = code.co_name
    counter[0] += 1
    name = _ident(raw_name, f"_{str(raw_name).strip('<>') or 'code'}_{counter[0]}")
    if name != raw_name:
        body[0] = ast.Expr(ast.Constant(f"was {raw_name}\n" + body[0].value.value))
    if _is_class_body(code):
        return ast.ClassDef(name=name, bases=[], keywords=[], body=body, decorator_list=[], type_params=[])
    cls = ast.AsyncFunctionDef if code.co_flags & (CO_COROUTINE | CO_ASYNC_GENERATOR) else ast.FunctionDef
    return cls(name=name, args=_arguments(code), body=body, decorator_list=[], returns=None, type_params=[])


def skeleton(blob: bc.CodeBlob) -> str:
    """Source stub of the code-object tree: signatures, names and constants."""
    code = blob.code
    intro = (f"PyArmor <= 7 module (Python {bc.vstr(blob.version)}).\n"
             "Bytecode is still encrypted; only signatures, names and constants are shown.\n")
    counter = [0]
    body = [ast.Expr(ast.Constant(_doc(code, [intro])))]
    body += [_node(c, counter) for c in code.co_consts if bc.is_code(c)]
    mod = ast.Module(body=body, type_ignores=[])
    ast.fix_missing_locations(mod)
    return ast.unparse(mod) + "\n"

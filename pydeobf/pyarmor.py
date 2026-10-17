"""Static PyArmor 8.0 - 9.x unpacking (no runtime loaded, nothing executed).

    __pyarmor__(__name__, __file__, b'PY000000...')
        -> AES-CTR decrypt of the outer blob (key derived from the bytes of
           pyarmor_runtime.{so,pyd,dylib}, or the shared trial key)
        -> PyArmor's marshal stream (standard marshal, but code objects carry
           an extra trailing field and their co_code is AES-CTR encrypted)
        -> per-code-object decryption + removal of the PyArmor extra field
        -> a STANDARD marshal stream / .pyc for the sample's Python version,
           handed to the normal pydeobf pipeline (emulator / PyLingual).

This is a pure data transformation: the runtime shared library is only read as
bytes (never imported), and the sample bytecode is only decrypted and
disassembled, never run.

Algorithm reference: the public, open-source Pyarmor-Static-Unpack-1shot
(Lil-House) - runtime.py, shot.py, pyc_code.cpp, pyc_module.cpp - and
cyber.wtf, "Unpacking Pyarmor v8+ scripts" (2025-02-12).
"""

from __future__ import annotations

import ast
import hashlib
import struct
from dataclasses import dataclass, field

from Crypto.Cipher import AES

from . import bytecode as bc

# RSA public certificate embedded in every pyarmor_runtime; part of the key MD5.
GLOBAL_CERT = bytes.fromhex(
    "3082010a0282010100bf6530f3bd67e7a69df8db18b2b9c1c05ffefbe54b91df6f38da51cceac4d3"
    "04bd952786c113ca7315444d97f510b95221721 6c8b2845f455632e7c26bad2bd9df52d6e9d12aba"
    "35e443ab54e791c5ced1f1baa59ff4cadb89043df89f6a8b8a2939f84c0db8a06d51c4742464fe1a"
    "2397f361eadec897dc576034be2c18503bd1763b492a399a3718538f1d4c82b1a033435719ad67e7"
    "af09fb0454a9eac0c1e9326c77927f9f7c087ce8a15da4fc40e66e18dbbf45534b5ca79df28f7e6c"
    "04b04dee99259a87846e9efe3c72ecb064dd2edbad32fa1d4b2c1a78857cbc2cd0d783775f92d5db"
    "591096532e5dc74212b861cb2c5f46149e93b05321a274342d0203010001".replace(" ", "")
)

# Trial / "free" PyArmor runtime (serial 000000): the key material is identical
# everywhere, so the AES key and mix-string nonce are fixed constants (verified
# against the runtime's own derivation).
_TRIAL_KEY = bytes.fromhex("ab738f35ffce23b13ae73d5a2c17a896")
_TRIAL_NONCE = bytes.fromhex("692e6e6f6e2d70726f666974")


def _md5_key(part_1: bytes, part_2: bytes, part_3: bytes) -> bytes:
    return hashlib.md5(part_1 + part_2 + part_3 + GLOBAL_CERT).digest()


def _dword(buf, off) -> int:
    return struct.unpack_from("<I", buf, off)[0]


@dataclass
class RuntimeKey:
    aes_key: bytes
    mix_str_nonce: bytes
    serial: str
    source: str

    @classmethod
    def trial(cls) -> "RuntimeKey":
        return cls(_TRIAL_KEY, _TRIAL_NONCE, "000000", "trial key (built-in)")

    @classmethod
    def from_runtime(cls, path) -> "RuntimeKey":
        """Derive the key from a pyarmor_runtime.{so,pyd,dylib}, reading its bytes
        only (the library is never loaded).  Mirrors 1shot's RuntimeInfo."""
        with open(path, "rb") as f:
            data = f.read(16 * 1024 * 1024)
        cur = data.find(b"pyarmor-vax")
        if cur == -1:
            raise ValueError("no 'pyarmor-vax' marker; not a PyArmor runtime (or packed/UPX)")
        if data[cur + 11:cur + 18] == b"\x00" * 7:
            raise ValueError("runtime template, not an instantiated runtime")
        data = bytearray(data[cur - 0x2C:cur - 0x2C + 1024 * 1024])
        if data[0x5C] & 1:
            raise ValueError('external key file ".pyarmor.ikey" is not supported')
        if _dword(data, 0x4C) != 0:
            xor_flag = 0x60 + _dword(data, 0x48)
            xor_target = 0x60 + _dword(data, 0x50)
            xor_length = int.from_bytes(data[xor_flag + 1:xor_flag + 4], "little")
            if data[xor_flag] == 1:
                for i in range(xor_length):
                    data[xor_target + i] ^= data[xor_flag + 4 + i]
        part_1 = bytes(data[0x2C:0x2C + 20])
        p2_off = 0x60 + _dword(data, 0x50)
        part_2 = bytes(data[p2_off:p2_off + _dword(data, 0x54)])
        var_a1 = 0x60 + _dword(data, 0x58)
        p3_len = _dword(data, var_a1 + 4)
        part_3 = bytes(data[var_a1 + 0x20:var_a1 + 0x20 + p3_len])
        serial = part_1[12:18].decode("utf-8", "replace")
        return cls(_md5_key(part_1, part_2, part_3), part_3[:12], serial, f"runtime {path}")


def _aes_ctr(data: bytes, key: bytes, nonce12: bytes) -> bytes:
    # GCM with a 12-byte IV encrypts from counter J0+1 = IV||0x00000002; with the
    # tag ignored this is exactly AES-CTR, 12-byte nonce, initial counter 2.
    return AES.new(key, AES.MODE_CTR, nonce=nonce12, initial_value=2).decrypt(data)


# ---------------------------------------------------------------- detection

def find_pyarmor_blob(source: str):
    """Return (blob, meta) for ``__pyarmor__(__name__, __file__, b'PY...')`` in
    ``source``, or None.  Only the literal bytes are read; nothing runs."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    runtime_pkg = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and (
            node.module.endswith("pyarmor_runtime") or "pyarmor_runtime_" in node.module
            or node.module == "pytransform"
        ):
            runtime_pkg = node.module
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "__pyarmor__":
            for a in node.args:
                if isinstance(a, ast.Constant) and isinstance(a.value, bytes) and a.value[:2] == b"PY":
                    return a.value, {"runtime_pkg": runtime_pkg}
    return None


# --------------------------------------------------------------- outer blob

@dataclass
class ArmoredModule:
    version: tuple
    marshal_stream: bytes
    co_code_nonce_xor_key: bytes | None
    bcc_segments: list = field(default_factory=list)  # (arch, elf_bytes)


def decrypt_blob(blob: bytes, key: RuntimeKey) -> ArmoredModule:
    if blob[:2] != b"PY":
        raise ValueError("not a PyArmor v8+ blob (missing 'PY' magic)")
    version = (blob[9], blob[10])
    protection = _dword(blob, 20)
    bcc_segments = []

    def decrypt_region(buf: bytes) -> bytes:
        cto, ctl = _dword(buf, 28), _dword(buf, 32)
        nonce = buf[36:40] + buf[44:52]
        dec = _aes_ctr(buf[cto:cto + ctl], key.aes_key, nonce)
        return buf[:cto] + dec + buf[cto + ctl:]

    if protection == 9:  # BCC mode: a native-code segment precedes the bytecode
        cto, ctl = _dword(blob, 28), _dword(blob, 32)
        nonce = blob[36:40] + blob[44:52]
        bcc = _aes_ctr(blob[cto:cto + ctl], key.aes_key, nonce)
        arch_map = {0x2001: "win-x64", 0x2003: "linux-x64", 0x3002: "darwin-arm64"}
        while len(bcc) >= 16:
            seg_off, seg_len, arch_id, nxt = struct.unpack_from("<IIII", bcc, 0)
            seg = bcc[seg_off:seg_off + seg_len]
            if seg.startswith(b"\x7fELF"):
                bcc_segments.append((arch_map.get(arch_id, hex(arch_id)), seg))
            if not nxt:
                break
            bcc = bcc[nxt:]
        remaining = blob[_dword(blob, 56):]
        module = decrypt_region(remaining)
    else:
        module = decrypt_region(blob)

    coo = _dword(module, 64)
    xor_len = _dword(module, 68)
    xor_key = None
    if xor_len > 0:
        procedure = module[64 + coo:64 + coo + xor_len]
        xor_key = _co_code_nonce_xor_key(procedure)
    marshal_stream = module[64 + coo + xor_len:]
    return ArmoredModule(version, marshal_stream, xor_key, bcc_segments)


def _co_code_nonce_xor_key(buf: bytes) -> bytes:
    """Interpret PyArmor's little register program that produces the 12-byte
    AES-nonce XOR key (ports pyarmorCoCodeAesNonceXorKeyCalculate)."""
    out = bytearray(12)
    regs = [0] * 8
    valid = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 7: 7}  # low-nibble -> register
    cur = 16

    def operand(cur):
        low = buf[cur + 1] & 0xF
        if low in valid:
            return regs[valid[low]], cur + 2
        size = buf[cur + 1] & 0x7
        if size == 1:
            return int.from_bytes(buf[cur + 2:cur + 3], "little", signed=True), cur + 3
        if size == 2:
            return int.from_bytes(buf[cur + 2:cur + 4], "little", signed=True), cur + 4
        return int.from_bytes(buf[cur + 2:cur + 6], "little", signed=True), cur + 6

    def i32(v):
        return v & 0xFFFFFFFF

    while cur < len(buf):
        op = buf[cur]
        if op == 1:
            cur += 1
        elif op in (2, 3, 4, 6, 7):
            hi = buf[cur + 1] >> 4
            val, cur = operand(cur)
            if op == 2:
                regs[hi] += val
            elif op == 3:
                regs[hi] -= val
            elif op == 4:
                regs[hi] = (regs[hi] * val)
            elif op == 6:
                regs[hi] ^= val
            elif op == 7:
                regs[hi] = val
        elif op == 5:
            hi = buf[cur + 1] >> 4
            val, cur = operand(cur)
            regs[hi] = 0 if val == 0 else int(regs[hi] / val)
            regs[0] = regs[hi]
        elif op == 8:
            cur += 2
        elif op == 9:
            reg = buf[cur + 1] & 0x7
            struct.pack_into("<i", out, 0, i32(regs[reg]) - (1 << 32) if regs[reg] & 0x80000000 else i32(regs[reg]))
            cur += 2
        elif op == 0xA:
            cur += 6
        elif op == 0xB:
            reg = buf[cur + 1] & 0x7
            dst = buf[cur + 2]
            struct.pack_into("<I", out, dst, i32(regs[reg]))
            cur += 3
        else:
            raise ValueError(f"unknown nonce-xor opcode {op:#x} at {cur}")
    return bytes(out)


# ------------------------------------------------------- marshal transcoder
#
# The PyArmor marshal stream is standard CPython marshal except that obfuscated
# code objects (flags & CO_OBFUSCATED) carry a 1-byte-length-prefixed extra
# field after co_exceptiontable/co_lnotab, and their co_code is AES-CTR
# encrypted.  We parse the stream, decrypt each co_code, drop the extra field
# and clear the flag, and re-serialise to STANDARD marshal.  Objects that are
# not an obfuscated code object (or an ancestor of one) are copied verbatim, so
# marshal back-references (TYPE_REF) stay valid.

CO_OBFUSCATED = 0x20000000
FLAG_REF = 0x80

# per-version code-object shape: (lead_i32_count, mid_obj_count, tail_obj_count)
# lead i32s end with `flags`; then co_code(obj); then mid objs; firstlineno(i32); tail objs
_CODE_SHAPE = {
    (3, 7): (5, 7, 1),
    (3, 8): (6, 7, 1), (3, 9): (6, 7, 1), (3, 10): (6, 7, 1),
    (3, 11): (5, 7, 2), (3, 12): (5, 7, 2), (3, 13): (5, 7, 2),
    (3, 14): (5, 7, 2), (3, 15): (5, 7, 2),
}


class _Node:
    __slots__ = ("kind", "raw", "head", "kids", "value", "dirty", "new_bytes")

    def __init__(self, kind):
        self.kind = kind          # 'scalar' | 'seq' | 'dict' | 'code'
        self.raw = b""            # exact input bytes (clean nodes copy this)
        self.head = b""           # container header (type byte + count)
        self.kids = []            # child _Node list (seq/dict/code-objs)
        self.value = None         # decoded value for scalars/tuples (descriptor lookup)
        self.dirty = False
        self.new_bytes = None     # rebuilt bytes for dirty code/string nodes


class _Transcoder:
    def __init__(self, data: bytes, version: tuple, key: RuntimeKey, nonce_xor_key, mix_str=True):
        self.d = data
        self.pos = 0
        self.version = tuple(version[:2])
        self.key = key
        self.nonce_xor_key = nonce_xor_key
        self.mix_str = mix_str
        if self.version not in _CODE_SHAPE:
            raise ValueError(f"unsupported Python version for PyArmor code objects: {self.version}")

    # -- primitive reads
    def _u8(self):
        v = self.d[self.pos]; self.pos += 1; return v

    def _i32(self):
        v = struct.unpack_from("<i", self.d, self.pos)[0]; self.pos += 4; return v

    def _u32(self):
        v = struct.unpack_from("<I", self.d, self.pos)[0]; self.pos += 4; return v

    def transcode(self) -> bytes:
        root = self._read()
        if self.pos != len(self.d):
            # trailing bytes after the top-level object: keep only the object
            pass
        return self._emit(root)

    # -- read one object, returning a _Node
    def _read(self) -> _Node:
        start = self.pos
        b = self._u8()
        flag = b & FLAG_REF
        t = chr(b & 0x7F)

        if t in "0NFTS.":                       # no payload
            n = _Node("scalar")
            n.value = {"N": None, "F": False, "T": True}.get(t, _MISS)
        elif t == "i":
            n = _Node("scalar"); n.value = self._i32()
        elif t == "g":
            n = _Node("scalar"); self.pos += 8; n.value = _MISS
        elif t == "y":
            n = _Node("scalar"); self.pos += 16; n.value = _MISS
        elif t == "I":
            n = _Node("scalar"); self.pos += 8; n.value = _MISS
        elif t == "l":                          # long
            n = _Node("scalar")
            ndig = self._i32()
            self.pos += 2 * abs(ndig); n.value = _MISS
        elif t in "stuaAzZ":                     # strings / bytes
            if t in "zZ":
                ln = self._u8()
            else:
                ln = self._u32()
            raw_data = self.d[self.pos:self.pos + ln]
            self.pos += ln
            n = _Node("scalar")
            if t == "s":
                n.value = bytes(raw_data)
            else:
                try:
                    n.value = raw_data.decode("utf-8", "surrogatepass")
                except Exception:
                    n.value = _MISS
            n.raw = self.d[start:self.pos]
            if self.mix_str:
                self._maybe_mix_str(n, b, t, raw_data)
            if n.raw == b"":
                n.raw = self.d[start:self.pos]
            return n
        elif t in "()[<>":                       # sequences
            n = _Node("seq")
            count = self._u8() if t == ")" else self._u32()
            n.head = self.d[start:self.pos]
            vals = []
            for _ in range(count):
                k = self._read(); n.kids.append(k)
                vals.append(k.value)
            n.value = tuple(vals) if t in "()" else _MISS
            if any(k.dirty for k in n.kids):
                n.dirty = True
            n.raw = self.d[start:self.pos]
            return n
        elif t == "{":                           # dict: pairs until NULL
            n = _Node("dict"); n.head = self.d[start:self.pos]
            while True:
                if self.d[self.pos] == ord("0"):
                    self.pos += 1
                    break
                n.kids.append(self._read())   # key
                n.kids.append(self._read())   # value
            if any(k.dirty for k in n.kids):
                n.dirty = True
            n.raw = self.d[start:self.pos]
            return n
        elif t == "r":                           # ref
            n = _Node("scalar"); self.pos += 4; n.value = _MISS
        elif t == "c":                           # code
            return self._read_code(start, b)
        else:
            raise ValueError(f"unsupported marshal type {t!r} ({b:#x}) at {start}")

        n.raw = self.d[start:self.pos]
        return n

    def _maybe_mix_str(self, n, type_byte, t, raw_data):
        # PyArmor "mix string": first byte has 0x80 set, low 7 bits in 1..4.
        if not raw_data or not (raw_data[0] & 0x80) or not (1 <= (raw_data[0] & 0x7F) <= 4):
            return
        body = bytearray(raw_data[1:])
        nonce = self.key.mix_str_nonce
        dec = _aes_ctr(bytes(body), self.key.aes_key, nonce)
        as_unicode = bool(raw_data[0] & 1)
        n.dirty = True
        if as_unicode:
            payload = dec
            n.new_bytes = bytes([type_byte & FLAG_REF | ord("u")]) + struct.pack("<I", len(payload)) + payload
            try:
                n.value = dec.decode("utf-8", "surrogatepass")
            except Exception:
                n.value = _MISS
        else:
            n.new_bytes = bytes([type_byte & FLAG_REF | ord("s")]) + struct.pack("<I", len(dec)) + dec
            n.value = bytes(dec)

    def _read_code(self, start, type_byte) -> _Node:
        lead, mid, tail = _CODE_SHAPE[self.version]
        n = _Node("code")
        # lead i32s (last is flags)
        lead_raw_start = self.pos
        vals = [self._i32() for _ in range(lead)]
        flags = vals[-1]
        obf = bool(flags & CO_OBFUSCATED)
        lead_raw = self.d[lead_raw_start:self.pos]
        # co_code (a bytes object)
        code_node = self._read()
        # mid objects
        mid_nodes = [self._read() for _ in range(mid)]
        # firstlineno i32
        firstlineno_raw = self.d[self.pos:self.pos + 4]; self.pos += 4
        # tail objects
        tail_nodes = [self._read() for _ in range(tail)]

        extra = None
        if obf:
            ex_len = self._u8()
            extra = self.d[self.pos:self.pos + ex_len]; self.pos += ex_len

        n.raw = self.d[start:self.pos]
        child_dirty = any(x.dirty for x in [code_node] + mid_nodes + tail_nodes)
        if not obf and not child_dirty:
            n.dirty = False
            return n

        n.dirty = True
        # decrypt co_code using the descriptor(s) from co_consts (first mid node)
        consts = mid_nodes[0].value
        new_code = bytearray(code_node.value)
        if obf:
            self._decrypt_co_code(new_code, extra, consts)
            _strip_markers(new_code, self.version, consts if isinstance(consts, tuple) else ())
            # drop PyArmor's now-dead synthetic exception handlers (3.11+)
            if self.version >= (3, 11) and tail_nodes:
                et = tail_nodes[-1]
                if isinstance(et.value, (bytes, bytearray)):
                    new_et = _filter_exc_table(bytes(new_code), self.version, bytes(et.value))
                    if new_et is not None:
                        hdr = et.raw[0:1] + struct.pack("<I", len(new_et))  # type byte + new u32 len
                        et.dirty = True
                        et.new_bytes = hdr + new_et
                        et.raw = hdr + new_et
        # rebuild the code bytes object (same type byte, FLAG_REF preserved)
        if not obf:
            cb = self._emit(code_node)
        else:
            cb = code_node.raw[:5] + bytes(new_code)

        # clear CO_OBFUSCATED in flags
        new_lead = bytearray(lead_raw)
        struct.pack_into("<i", new_lead, (lead - 1) * 4, flags & ~CO_OBFUSCATED)

        out = bytearray()
        out.append(type_byte)                    # 'c' with original FLAG_REF
        out += new_lead
        out += cb
        for x in mid_nodes:
            out += self._emit(x)
        out += firstlineno_raw
        for x in tail_nodes:
            out += self._emit(x)
        # extra field dropped
        n.new_bytes = bytes(out)
        return n

    def _decrypt_co_code(self, code: bytearray, extra: bytes, consts):
        fn_count = extra[0] & 3
        desc_count = (extra[0] >> 2) & 3
        p = 4
        for _ in range(fn_count):
            p += (extra[p] >> 6) + 2
        for _ in range(desc_count):
            item_len = (extra[p] >> 6) + 2
            item_end = p + item_len
            p += 1
            idx = 0
            while p < item_end:
                idx = (idx << 8) | extra[p]; p += 1
            descriptor = consts[idx]
            if not isinstance(descriptor, (bytes, bytearray)) or len(descriptor) < 20:
                continue
            self._apply_descriptor(code, descriptor)

    def _apply_descriptor(self, code: bytearray, descriptor: bytes):
        flags, short_nonce_index, _, decrypt_begin = descriptor[8:12]
        decrypt_length = struct.unpack_from("<I", descriptor, 12)[0]
        copy_prologue = flags & 0x8
        xor_nonce = flags & 0x4
        short_code = flags & 0x2
        nonce_index = short_nonce_index if short_code else short_nonce_index + decrypt_begin + decrypt_length
        nonce = bytearray(code[nonce_index:nonce_index + 12])
        if xor_nonce:
            if not self.nonce_xor_key:
                raise ValueError("co_code nonce XOR required but no XOR key was derived")
            for i in range(12):
                nonce[i] ^= self.nonce_xor_key[i]
        dec = _aes_ctr(bytes(code[decrypt_begin:decrypt_begin + decrypt_length]), self.key.aes_key, bytes(nonce))
        code[decrypt_begin:decrypt_begin + decrypt_length] = dec
        if copy_prologue:
            code[0:decrypt_begin] = code[decrypt_length:decrypt_length + decrypt_begin]
            nop = {(3, 13): 30, (3, 14): 27}.get(self.version, 9)
            for i in range(decrypt_begin):
                code[decrypt_length + i] = nop

    def _emit(self, n: _Node) -> bytes:
        if not n.dirty:
            return n.raw
        if n.new_bytes is not None:
            return n.new_bytes
        if n.kind == "seq":
            out = bytearray(n.head)
            for k in n.kids:
                out += self._emit(k)
            return bytes(out)
        if n.kind == "dict":
            out = bytearray(n.head)
            for k in n.kids:
                out += self._emit(k)
            out += b"0"   # TYPE_NULL terminator
            return bytes(out)
        return n.raw


_MISS = object()


def transcode_to_standard_marshal(module: ArmoredModule, key: RuntimeKey, mix_str=True) -> bytes:
    tc = _Transcoder(module.marshal_stream, module.version, key, module.co_code_nonce_xor_key, mix_str)
    return tc.transcode()


@dataclass
class Unpacked:
    version: tuple
    marshal_stream: bytes      # standard marshal
    pyc: bytes                 # standard .pyc (header + marshal)
    key: RuntimeKey
    bcc_segments: list
    runtime_pkg: str | None


def unpack(source: str, runtime_path=None, mix_str=True) -> Unpacked | None:
    """Statically unpack a PyArmor-obfuscated .py source string.

    runtime_path: a pyarmor_runtime.{so,pyd,dylib}; if omitted the built-in
    trial key is used (correct for the free/trial PyArmor, serial 000000).
    Returns None if the source is not PyArmor-obfuscated.
    """
    found = find_pyarmor_blob(source)
    if not found:
        return None
    blob, meta = found
    key = None
    if runtime_path:
        try:
            key = RuntimeKey.from_runtime(runtime_path)
        except Exception:
            key = None
    if key is None:
        key = RuntimeKey.trial()
    module = decrypt_blob(blob, key)
    std = transcode_to_standard_marshal(module, key, mix_str)
    pyc = bc.make_pyc(std, module.version)
    return Unpacked(module.version, std, pyc, key, module.bcc_segments, meta.get("runtime_pkg"))


_NOP_OP = {(3, 13): 30, (3, 14): 27}  # NOP is 9 for <= 3.12
# opcodes used inside PyArmor marker groups; a group may only contain these
_MARKER_SAFE = {
    "LOAD_CONST", "LOAD_SMALL_INT", "PUSH_NULL", "BUILD_TUPLE", "CALL_FUNCTION_EX", "CALL_FUNCTION",
    "CALL", "STORE_FAST", "POP_TOP", "PRECALL", "KW_NAMES", "NOP", "CACHE", "RESUME",
}


def _decode_cells(code: bytes, opc, version):
    """Minimal linear decode -> list of (start, end, opname, arg). 3.11+ CACHE
    cells (opcode 0) are folded into the preceding instruction's span."""
    opname = opc.opname
    cache = opc.opmap.get("CACHE", -1)
    ext = opc.opmap.get("EXTENDED_ARG", -1)
    wordcode = version >= (3, 6)
    out = []
    i = 0
    n = len(code)
    ext_arg = 0
    while i < n:
        op = code[i]
        if wordcode:
            if op == cache:
                if out:
                    out[-1] = out[-1][:1] + (i + 2,) + out[-1][2:]
                i += 2
                continue
            arg = (ext_arg << 8) | code[i + 1] if op >= opc.HAVE_ARGUMENT else None
            start = i
            i += 2
        else:
            arg = None
            start = i
            i += 1
            if op >= opc.HAVE_ARGUMENT:
                arg = (ext_arg << 16) | code[i] | (code[i + 1] << 8)
                i += 2
        name = opname[op] if 0 <= op < len(opname) else f"<{op}>"
        if op == ext:
            ext_arg = (ext_arg << 8) | (arg & 0xFF)
            out.append((start, i, name, arg))
            continue
        ext_arg = 0
        out.append((start, i, name, arg))
    return out


def _strip_markers(code: bytearray, version, consts):
    """NOP out PyArmor's __pyarmor_enter/leave/exit/assert__ marker instruction
    groups.  Size-preserving (each group is stack-neutral), so jump targets and
    the exception table stay valid; the real logic is untouched."""
    opc = bc.opcode_module(version)
    nop = _NOP_OP.get(tuple(version[:2]), 9)
    try:
        cells = _decode_cells(bytes(code), opc, version)
    except Exception:
        return
    load_const = opc.opmap.get("LOAD_CONST", -1)

    def is_marker(cell):
        start, end, name, arg = cell
        return (name == "LOAD_CONST" and arg is not None and 0 <= arg < len(consts)
                and isinstance(consts[arg], str) and consts[arg].startswith("__pyarmor_"))

    k = 0
    while k < len(cells):
        if not is_marker(cells[k]):
            k += 1
            continue
        group = [cells[k]]
        ok = False
        j = k + 1
        while j < len(cells):
            name = cells[j][2]
            if name not in _MARKER_SAFE:
                break
            group.append(cells[j])
            if name in ("POP_TOP", "STORE_FAST"):
                ok = True
                break
            j += 1
        if ok:
            for (start, end, _n, _a) in group:
                code[start] = nop
                for p in range(start + 1, end):
                    code[p] = 0
            k = j + 1
        else:
            k += 1


# ------------------------------- exception-table surgery (3.11+) -----------
# After NOP-ing the enter/exit markers, PyArmor's synthetic catch-all handler
# (which re-encrypts co_code at runtime) is dead but still referenced by
# co_exceptiontable, so decompilers wrap every function in try/finally.  We drop
# ONLY the entries whose handler is pure exception plumbing (no real user code),
# leaving genuine try/except intact.

_PLUMBING = {
    "PUSH_EXC_INFO", "RERAISE", "COPY", "POP_EXCEPT", "POP_TOP", "NOP", "RESUME",
    "LOAD_CONST", "LOAD_COMMON_CONSTANT", "RETURN_VALUE", "RETURN_CONST", "SWAP",
}


def _exc_varint_decode(data, pos):
    b = data[pos]; val = b & 0x3F; pos += 1
    while b & 0x40:
        b = data[pos]; val = (val << 6) | (b & 0x3F); pos += 1
    return val, pos


def _parse_exc_words(data):
    out, pos = [], 0
    while pos < len(data):
        start, pos = _exc_varint_decode(data, pos)
        length, pos = _exc_varint_decode(data, pos)
        target, pos = _exc_varint_decode(data, pos)
        dl, pos = _exc_varint_decode(data, pos)
        out.append((start, length, target, dl))
    return out


def _exc_varint_encode(val):
    chunks = []
    while True:
        chunks.append(val & 0x3F)
        val >>= 6
        if not val:
            break
    chunks.reverse()
    out = bytearray()
    for i, c in enumerate(chunks):
        out.append(c | (0x40 if i != len(chunks) - 1 else 0))
    return out


def _encode_exc_table(entries):
    out = bytearray()
    for start, length, target, dl in entries:
        blob = bytearray()
        for v in (start, length, target, dl):
            blob += _exc_varint_encode(v)
        blob[0] |= 0x80
        out += blob
    return bytes(out)


def _filter_exc_table(code: bytes, version, table_bytes: bytes):
    """Return a new exception table with PyArmor's synthetic plumbing handlers
    removed, or None if nothing changed."""
    if not table_bytes:
        return None
    opc = bc.opcode_module(version)
    try:
        cells = _decode_cells(code, opc, version)
        entries = _parse_exc_words(table_bytes)
    except Exception:
        return None
    by_off = {c[0]: c[2] for c in cells}        # offset -> opname
    offs = sorted(by_off)

    def handler_is_plumbing(target_byte):
        # scan from the handler target until RERAISE (handler end)
        i = 0
        while i < len(offs) and offs[i] < target_byte:
            i += 1
        seen_real = False
        while i < len(offs):
            name = by_off[offs[i]]
            if name == "RERAISE":
                return not seen_real
            if name not in _PLUMBING:
                seen_real = True
            i += 1
        return not seen_real

    kept = [e for e in entries if not handler_is_plumbing(e[2] * 2)]
    if len(kept) == len(entries):
        return None
    return _encode_exc_table(kept)

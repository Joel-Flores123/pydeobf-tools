"""Allow-lists for the emulator: which host callables/attributes are safe.

The rule is simple: emulated code can only reach host functionality through
objects defined here.  Everything is side-effect free (no I/O, no process,
network or filesystem access, no arbitrary attribute walking).
"""

from __future__ import annotations

import base64
import binascii
import builtins as _b
import bz2
import codecs
import collections
import hashlib
import hmac
import itertools
import json
import lzma
import math
import operator
import random
import re
import string
import struct
import sys
import types
import zlib
import ntpath
import posixpath
import ast as _ast

from . import vm as _vm

MAX_OUT = 512 * 1024 * 1024

PRIMITIVE_TYPES = (
    str, bytes, bytearray, int, float, complex, bool, list, tuple, dict, set, frozenset, range, slice,
    type(None), memoryview, type(Ellipsis), type(NotImplemented),
)
_ITER_TYPES = tuple({
    type(iter("")), type(iter(b"")), type(iter([])), type(iter(())), type(iter({})), type(iter(set())),
    type(iter(range(1))), type(iter(range(1 << 70))), type(reversed([])), type(reversed(range(1))),
    type({}.keys()), type({}.values()), type({}.items()), type(iter({}.items())), type(iter({}.values())),
    type(iter(bytearray())), type(reversed("")) if hasattr(type(reversed("")), "__next__") else type(iter("")),
    map, filter, zip, enumerate, reversed,
    itertools.chain, itertools.islice, itertools.product, itertools.permutations, itertools.combinations,
    itertools.accumulate, itertools.starmap, itertools.zip_longest, itertools.groupby, itertools.takewhile,
    itertools.dropwhile, itertools.compress, itertools.cycle, itertools.count, itertools.repeat,
    type(re.compile("")), type(re.match("", "")), type(sys.version_info),
    type(hashlib.md5()), type(hashlib.sha256()), type(hashlib.blake2b()),
    collections.OrderedDict, collections.deque, collections.Counter, collections.defaultdict,
    type(memoryview(b"")),
})
# sys.version_info as reported to emulated code that targets another Python version
VersionInfo = collections.namedtuple("version_info", "major minor micro releaselevel serial")
FrameInfo = collections.namedtuple("FrameInfo", "frame filename lineno function code_context index")  # inspect.stack()


class SysModules(dict):
    """sys.modules: lookups of modules the sample expects to be loaded (its own runtime
    helpers, injected by a packer) give an opaque value instead of KeyError; membership
    tests (`'pdb' in sys.modules`) still see only what was put there."""

    def __missing__(self, key):
        return _vm.Opaque(f"sys.modules[{key!r}]")


SAFE_INSTANCE_TYPES = frozenset(PRIMITIVE_TYPES + _ITER_TYPES + (VersionInfo, FrameInfo, SysModules))

# samples built for <=3.10 turn int-encoded payloads into text with str(n); 3.11+ caps that at 4300 digits
if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

_EXC_TYPES = [v for v in vars(_b).values() if isinstance(v, type) and issubclass(v, BaseException)]

SAFE_TYPES = frozenset(PRIMITIVE_TYPES + (object, enumerate, zip, map, filter, reversed)
                       + tuple(_EXC_TYPES) + tuple(t for t in _ITER_TYPES if isinstance(t, type)))

SAFE_DUNDERS = frozenset(
    "__len__ __getitem__ __contains__ __iter__ __next__ __add__ __radd__ __mul__ __rmul__ __mod__ __rmod__ "
    "__eq__ __ne__ __lt__ __le__ __gt__ __ge__ __hash__ __str__ __repr__ __format__ __name__ __qualname__ "
    "__doc__ __class__ __sub__ __rsub__ __xor__ __rxor__ __and__ __rand__ __or__ __ror__ __neg__ __invert__ "
    "__int__ __float__ __bool__ __index__ __reversed__ __setitem__ __delitem__ __truediv__ __floordiv__ "
    "__pow__ __lshift__ __rshift__ __abs__ __divmod__ __round__ __iadd__ __sizeof__".split()
)

_SAFE_IDS: dict[int, object] = {}


def _allow(*objs):
    for o in objs:
        _SAFE_IDS[id(o)] = o
    return objs[0] if len(objs) == 1 else objs


_PURE_BUILTINS = [
    "abs", "all", "any", "ascii", "bin", "chr", "divmod", "format", "hash", "hex", "id", "len", "max", "min",
    "oct", "ord", "pow", "repr", "round", "sorted", "sum", "next",
]
_allow(*[getattr(_b, n) for n in _PURE_BUILTINS])
_allow(str.maketrans, bytes.maketrans, bytearray.maketrans)


def is_safe_callable(f) -> bool:
    if id(f) in _SAFE_IDS and _SAFE_IDS[id(f)] is f:
        return True
    if isinstance(f, type):
        return f in SAFE_TYPES
    if isinstance(f, (types.BuiltinMethodType, types.MethodWrapperType, types.MethodDescriptorType,
                      types.WrapperDescriptorType, types.ClassMethodDescriptorType)):
        owner = getattr(f, "__self__", None)
        if owner is None or isinstance(owner, types.ModuleType):
            owner = getattr(f, "__objclass__", None)
            if owner is None:
                return False
        if isinstance(owner, type):
            return owner in SAFE_TYPES and (not f.__name__.startswith("__") or f.__name__ in SAFE_DUNDERS)
        if f.__name__.startswith("__") and f.__name__ not in SAFE_DUNDERS:
            return False
        return type(owner) in SAFE_INSTANCE_TYPES
    return False


def attr_allowed(obj, name) -> bool:
    if name.startswith("__") and name not in SAFE_DUNDERS:
        return False
    if isinstance(obj, type):
        return obj in SAFE_TYPES
    return type(obj) in SAFE_INSTANCE_TYPES


# ------------------------------------------------------------------ guards

_SIZE_SENSITIVE = {"ljust", "rjust", "center", "zfill", "expandtabs", "to_bytes", "repeat", "bytes", "bytearray",
                   "join", "replace"}


_MEMORY_BOMB = 1 << 32  # bytes; anti-emulation bombs ([x] * 987654321, ...) rely on MemoryError


def guard_binop(op, a, b):
    o = op.rstrip("=")
    if o == "*":
        seq, n = (a, b) if isinstance(b, int) else (b, a)
        if isinstance(seq, (str, bytes, bytearray, list, tuple)) and isinstance(n, int) and len(seq) * n > _vm.SIZE_LIMIT:
            if len(seq) * n * (8 if isinstance(seq, (list, tuple)) else 1) > _MEMORY_BOMB:
                raise _vm.VMRaise(MemoryError())
            raise _vm.VMStop("refusing huge sequence repetition")
    elif o == "**" and isinstance(a, int) and isinstance(b, int) and b > 0 and abs(a) > 1:
        if a.bit_length() * b > 50_000_000:
            raise _vm.VMStop("refusing huge power")
    elif o == "<<" and isinstance(b, int) and b > 50_000_000:
        raise _vm.VMStop("refusing huge shift")


def guard_call(vm, fn, args, kwargs):
    name = getattr(fn, "__name__", "")
    if fn in (list, tuple, set, frozenset, bytes, bytearray, sorted, dict) and args:
        a = args[0]
        if isinstance(a, range) and len(a) > _vm.SIZE_LIMIT // 8:
            raise _vm.VMStop("refusing to materialise huge range")
        if isinstance(a, int) and fn in (bytes, bytearray) and a > _vm.SIZE_LIMIT:
            raise _vm.VMStop("refusing huge allocation")
    # str(n)/int(n) and a seed are as big as the int already is; only allocations scale with it
    if name in _SIZE_SENSITIVE or (isinstance(fn, type) and fn not in (SeededRandom, str, int)):
        for a in list(args) + list(kwargs.values()):
            if isinstance(a, int) and not isinstance(a, bool) and a > _vm.SIZE_LIMIT:
                raise _vm.VMStop(f"refusing huge argument to {name}")


# --------------------------------------------------- bounded decompressors

def _bounded(dobj_factory, data, what):
    out = []
    total = 0
    data = bytes(data)
    while True:
        d = dobj_factory()
        chunk = d.decompress(data, MAX_OUT - total) if what != "zlib" else d.decompress(data, MAX_OUT - total)
        total += len(chunk)
        out.append(chunk)
        if total >= MAX_OUT:
            raise _vm.VMStop("decompression bomb")
        if not d.eof:
            if what == "zlib":
                raise zlib.error("Error -5 while decompressing data: incomplete or truncated stream")
            raise EOFError("Compressed data ended before the end-of-stream marker was reached")
        rest = d.unused_data
        if what == "zlib" or not rest:
            if what == "zlib":
                out.append(d.flush())
            return b"".join(out)
        data = rest


def zlib_decompress(data, wbits=15, bufsize=16384):
    return _bounded(lambda: zlib.decompressobj(wbits), data, "zlib")


def bz2_decompress(data):
    return _bounded(bz2.BZ2Decompressor, data, "bz2")


def lzma_decompress(data, format=lzma.FORMAT_AUTO, memlimit=None, filters=None):
    return _bounded(lambda: lzma.LZMADecompressor(format, memlimit, filters), data, "lzma")


def gzip_decompress(data):
    return _bounded(lambda: zlib.decompressobj(31), data, "zlib")


class StreamDecompressor(_vm.HelperObject):
    """zlib.decompressobj() / bz2.BZ2Decompressor() / lzma.LZMADecompressor(), output capped at MAX_OUT."""

    ALLOWED = frozenset({"decompress", "flush", "eof", "unused_data", "unconsumed_tail", "needs_input"})

    def __init__(self, d):
        self._d = d
        self._total = 0

    def _count(self, chunk):
        self._total += len(chunk)
        if self._total > MAX_OUT:
            raise _vm.VMStop("decompression bomb")
        return chunk

    def decompress(self, data, max_length=-1):
        room = MAX_OUT - self._total + 1
        return self._count(self._d.decompress(bytes(data), room if max_length in (-1, 0) else min(max_length, room)))

    def flush(self, *a):
        return self._count(self._d.flush(*a)) if hasattr(self._d, "flush") else b""

    def __getattr__(self, name):
        if name in ("eof", "unused_data", "unconsumed_tail", "needs_input"):
            return getattr(self._d, name)
        raise AttributeError(name)


def zlib_decompressobj(wbits=15, zdict=b""):
    return StreamDecompressor(zlib.decompressobj(wbits, zdict=zdict) if zdict else zlib.decompressobj(wbits))


def bz2_decompressor():
    return StreamDecompressor(bz2.BZ2Decompressor())


def lzma_decompressor(format=lzma.FORMAT_AUTO, memlimit=None, filters=None):
    return StreamDecompressor(lzma.LZMADecompressor(format, memlimit, filters))


_allow(zlib_decompressobj, bz2_decompressor, lzma_decompressor)


def hashlib_pbkdf2_hmac(hash_name, password, salt, iterations, dklen=None):
    if not isinstance(iterations, int) or iterations > 5_000_000:  # key derivation, not a CPU sink
        raise _vm.VMStop("refusing huge pbkdf2 iteration count")
    return hashlib.pbkdf2_hmac(hash_name, password, salt, iterations, dklen)


def hashlib_scrypt(password, *, salt, n, r, p, maxmem=0, dklen=64):
    if not all(isinstance(v, int) for v in (n, r, p)) or n * r * p > 1 << 22:
        raise _vm.VMStop("refusing huge scrypt cost")
    return hashlib.scrypt(password, salt=salt, n=n, r=r, p=p, maxmem=max(maxmem, 256 * n * r + (1 << 20)),
                          dklen=dklen)


_allow(hashlib_pbkdf2_hmac, hashlib_scrypt)


_COMPRESS_CODECS = {"zlib": zlib_decompress, "zlib_codec": zlib_decompress, "zip": zlib_decompress,
                    "bz2": bz2_decompress, "bz2_codec": bz2_decompress}


def codecs_decode(obj, encoding="utf-8", errors="strict"):
    enc = str(encoding).lower().replace("-", "_")
    if enc in _COMPRESS_CODECS:
        return _COMPRESS_CODECS[enc](obj)
    return codecs.decode(obj, encoding, errors)


_allow(zlib_decompress, bz2_decompress, lzma_decompress, gzip_decompress, codecs_decode)


# ------------------------------------------------------------------ modules

def _mod(name, attrs):
    attrs = dict(attrs)
    for k, v in attrs.items():
        if callable(v) and not isinstance(v, _vm.VMObject):
            _allow(v)
            if isinstance(v, types.FunctionType):
                # our Python stand-in for a C builtin: look like one (str/type/__module__), call it guarded
                attrs[k] = _vm.Special(k, v, name, host=True)
    return _vm.SafeModule(name, attrs)


def _pick(module, names):
    return {n: getattr(module, n) for n in names if hasattr(module, n)}


_rng = random.Random(1337)


# The emulated process runs <FAKE_CWD>/<vm.script_name>: getcwd(), abspath/realpath, __file__, sys.argv and
# the outermost frame agree, since obfuscators check realpath(__file__) == getcwd() + sep + name.
FAKE_CWD = {"windows": "C:\\Users\\user\\Desktop", "linux": "/home/user"}


FAKE_ENV = {
    "windows": {"USERNAME": "user", "USERPROFILE": "C:\\Users\\user", "HOMEDRIVE": "C:", "HOMEPATH": "\\Users\\user",
                "APPDATA": "C:\\Users\\user\\AppData\\Roaming", "LOCALAPPDATA": "C:\\Users\\user\\AppData\\Local",
                "TEMP": "C:\\Users\\user\\AppData\\Local\\Temp", "TMP": "C:\\Users\\user\\AppData\\Local\\Temp",
                "COMPUTERNAME": "DESKTOP-PC", "OS": "Windows_NT", "PROCESSOR_ARCHITECTURE": "AMD64",
                "NUMBER_OF_PROCESSORS": "8", "SystemRoot": "C:\\Windows", "windir": "C:\\Windows",
                "SystemDrive": "C:", "ProgramFiles": "C:\\Program Files", "PATH": "C:\\Windows\\system32;C:\\Windows"},
    "linux": {"USER": "user", "HOME": "/home/user", "SHELL": "/bin/bash", "PATH": "/usr/local/bin:/usr/bin:/bin",
              "LANG": "en_US.UTF-8", "TMPDIR": "/tmp"},
}

# What a clean, undebugged desktop answers to environment probes that are otherwise opaque.
# Anti-debug/anti-VM checks branch on these; an opaque answer sends them down the "detected" path.
ENV_PROBES = {
    "ctypes.windll.kernel32.IsDebuggerPresent": 0,
    "platform.machine": {"windows": "AMD64", "linux": "x86_64"},
    "platform.system": {"windows": "Windows", "linux": "Linux"},
    "platform.release": {"windows": "10", "linux": "6.5.0"},
    "platform.node": "DESKTOP-PC", "socket.gethostname": "DESKTOP-PC",
    "platform.architecture": ("64bit", "WindowsPE"),
    "multiprocessing.cpu_count": 8, "os.cpu_count": 8,
    "os.getpid": 4242, "os.getppid": 4100, "os.getlogin": "user", "getpass.getuser": "user",
    "traceback.format_stack": [], "traceback.extract_stack": [],
}


def env_probe(vm, path):
    """(True, value) for a known environment probe ``path`` (e.g. 'platform.machine'), else (False, None)."""
    if path not in ENV_PROBES:
        return False, None
    v = ENV_PROBES[path]
    return True, (v[vm.platform] if isinstance(v, dict) else v)


def script_path(vm):
    return (ntpath if vm.platform == "windows" else posixpath).join(FAKE_CWD[vm.platform], vm.script_name)


def _static_modules(platform):
    win = platform == "windows"
    pathmod = ntpath if win else posixpath
    path_attrs = _pick(pathmod, ["join", "basename", "dirname", "splitext", "split", "normpath", "isabs",
                                 "normcase", "splitdrive", "sep", "extsep", "altsep", "pathsep", "curdir", "pardir"])

    def abspath(p, *, strict=False):
        return pathmod.normpath(pathmod.join(FAKE_CWD[platform], p))

    env = FAKE_ENV[platform]
    known_dirs = {abspath(d).lower() for d in [FAKE_CWD[platform]] + [v for v in env.values() if pathmod.isabs(v)]}

    def isdir(p):  # a fresh machine: the user's standard folders exist, nothing else
        return isinstance(p, str) and abspath(p).lower() in known_dirs

    path_attrs.update(abspath=_vm.Special("abspath", abspath, pathmod.__name__),
                      realpath=_vm.Special("realpath", abspath, pathmod.__name__),
                      isdir=_vm.Special("isdir", isdir, pathmod.__name__))
    return {
        "base64": _pick(base64, ["b64decode", "b64encode", "b32decode", "b32encode", "b16decode", "b16encode",
                                 "b85decode", "b85encode", "a85decode", "a85encode", "urlsafe_b64decode",
                                 "urlsafe_b64encode", "standard_b64decode", "standard_b64encode", "decodebytes",
                                 "encodebytes", "b32hexdecode", "b32hexencode"]),
        "zlib": {**_pick(zlib, ["compress", "crc32", "adler32", "MAX_WBITS", "DEFLATED", "error"]),
                 "decompress": zlib_decompress, "decompressobj": zlib_decompressobj},
        "bz2": {"compress": bz2.compress, "decompress": bz2_decompress, "BZ2Decompressor": bz2_decompressor},
        "lzma": {"compress": lzma.compress, "decompress": lzma_decompress, "LZMADecompressor": lzma_decompressor,
                 **_pick(lzma, ["FORMAT_AUTO", "FORMAT_XZ", "FORMAT_ALONE", "FORMAT_RAW", "LZMAError"])},
        "gzip": {"decompress": gzip_decompress, "compress": __import__("gzip").compress},
        "binascii": _pick(binascii, ["unhexlify", "hexlify", "a2b_base64", "b2a_base64", "a2b_hex", "b2a_hex",
                                     "crc32", "a2b_uu", "b2a_uu", "a2b_qp", "b2a_qp", "crc_hqx", "Error"]),
        "codecs": {**_pick(codecs, ["encode", "BOM_UTF8", "lookup"]), "decode": codecs_decode},
        "string": _pick(string, ["ascii_letters", "ascii_lowercase", "ascii_uppercase", "digits", "hexdigits",
                                 "octdigits", "punctuation", "printable", "whitespace", "capwords"]),
        "math": {k: v for k, v in vars(math).items() if not k.startswith("_")},
        "operator": {k: v for k, v in vars(operator).items()
                     if not k.startswith("_") and k not in ("attrgetter", "methodcaller", "call")},
        "functools": {"reduce": functools_reduce, "partial": __import__("functools").partial},
        "itertools": _pick(itertools, ["chain", "islice", "product", "permutations", "combinations",
                                       "combinations_with_replacement", "accumulate", "starmap", "zip_longest",
                                       "groupby", "takewhile", "dropwhile", "compress", "cycle", "count", "repeat",
                                       "pairwise", "batched"]),
        "re": _pick(re, ["compile", "sub", "subn", "match", "search", "findall", "finditer", "split", "fullmatch",
                         "escape", "I", "IGNORECASE", "M", "MULTILINE", "S", "DOTALL", "X", "VERBOSE", "A",
                         "ASCII", "error"]),
        "struct": _pick(struct, ["pack", "unpack", "calcsize", "iter_unpack", "unpack_from", "error"]),
        "hmac": {"new": hmac_new, "HMAC": hmac_new, "compare_digest": hmac.compare_digest, "digest": hmac.digest},
        "hashlib": {**_pick(hashlib, ["md5", "sha1", "sha224", "sha256", "sha384", "sha512", "blake2b", "blake2s",
                                      "sha3_256", "sha3_512", "new"]),
                    "pbkdf2_hmac": hashlib_pbkdf2_hmac, "scrypt": hashlib_scrypt},
        "random": {**{n: getattr(_rng, n) for n in ["random", "randint", "choice", "shuffle", "randrange", "sample",
                                                    "seed", "uniform", "choices", "getrandbits"]},
                   "Random": SeededRandom},
        "json": _pick(json, ["loads", "dumps", "JSONDecodeError"]),
        "ast": {"literal_eval": _ast.literal_eval},
        "collections": _pick(collections, ["OrderedDict", "deque", "Counter", "defaultdict"]),
        "os.path": path_attrs,
        "ntpath": _pick(ntpath, list(path_attrs)),
        "posixpath": _pick(posixpath, list(path_attrs)),
    }


_CRYPTO_ROOTS = ("Crypto", "Cryptodome", "cryptography")


def _load_crypto(vm):
    """All crypto modules at once, so each package holds its submodules as attributes."""
    from . import cryptolib
    built = {name: _mod(name, dict(attrs)) for name, attrs in cryptolib.all_modules().items()}
    for name, mod in built.items():
        parent, _, leaf = name.rpartition(".")
        if parent in built:
            built[parent].attrs.setdefault(leaf, mod)
    for name, mod in built.items():
        vm.modules.setdefault(name, mod)


class SeededRandom(_vm.HelperObject):
    """random.Random(seed): deterministic, so keys derived from it are recoverable."""

    ALLOWED = frozenset({"random", "randint", "choice", "shuffle", "randrange", "sample", "seed", "uniform",
                         "choices", "getrandbits", "randbytes"})

    def __init__(self, seed=None):
        self._r = random.Random(seed if seed is not None else 1337)


def _delegate(name):
    def m(self, *a, **k):
        return getattr(self._r, name)(*a, **k)
    m.__name__ = name
    return m


for _n in SeededRandom.ALLOWED:
    setattr(SeededRandom, _n, _delegate(_n))


_allow(SeededRandom)


def hmac_new(key, msg=None, digestmod=""):
    from .cryptolib import HashObj
    return HashObj(hmac.new(key, msg, digestmod))


_allow(hmac_new)


def functools_reduce(fn, seq, *init):
    import functools as ft
    return ft.reduce(fn, seq, *init)


_allow(functools_reduce)


def _sys_version(version):
    """version_info, version and hexversion as seen by code compiled for ``version``."""
    if tuple(version) == sys.version_info[:2]:
        return sys.version_info, sys.version, sys.hexversion
    major, minor = version
    info = VersionInfo(major, minor, 0, "final", 0)
    return info, f"{major}.{minor}.0 (main, Jan  1 2024, 00:00:00) [GCC 11.4.0]", (major << 24) | (minor << 16) | 0xF0


# modules an inner stage may use without importing (its host program did)
AUTO_IMPORT = frozenset({"marshal", "base64", "zlib", "bz2", "lzma", "binascii", "codecs", "os", "sys", "time",
                         "random", "hashlib", "json", "re", "string", "math", "struct", "types", "dis"})


def _fs_noop(vm, name):
    def fn(path, *a, **k):
        vm.event(f"os.{name}({_vm.describe(path, 80)}) - not performed")
    return fn


def get_module(vm, name):
    # obfuscated samples compare sys.version with the version they were compiled for and exit on a mismatch
    version = vm.frames[-1].version if vm.frames else _vm.HOST
    if version == _vm.HOST and vm.version_hint:  # exec'd source is compiled by the host
        version = tuple(vm.version_hint)
    key = (name, version) if name == "sys" else name
    if key in vm.modules:
        return vm.modules[key]
    if not hasattr(vm, "_static_mods"):
        vm._static_mods = _static_modules(vm.platform)
    mod = None
    if name == "builtins":
        mod = vm.builtins_module
    elif name == "marshal":
        mod = _vm.SafeModule("marshal", {"loads": _vm.Special("loads", vm.marshal_loads, "marshal"),
                                         "dumps": _vm.Special("dumps", vm.marshal_dumps, "marshal"),
                                         "load": _vm.Special("load", lambda f: vm.marshal_loads(f.read()))})
    elif name == "sys":
        version_info, version_str, hexversion = _sys_version(version)
        mod = _vm.SafeModule("sys", {
            "version_info": version_info, "version": version_str, "hexversion": hexversion,
            "platform": "win32" if vm.platform == "windows" else "linux", "argv": [vm.script_name],
            "maxsize": sys.maxsize, "byteorder": "little", "executable": "python",
            "exit": vm.builtins["exit"], "modules": SysModules(), "flags": _vm.Opaque("sys.flags"),
            "getrecursionlimit": _vm.Special("getrecursionlimit", lambda: 1000),
            # no debugger/profiler attached: anti-debug checks take the normal path
            "gettrace": _vm.Special("gettrace", lambda: None),
            "getprofile": _vm.Special("getprofile", lambda: None),
            "settrace": _vm.Special("settrace", lambda *a: None),
            "setprofile": _vm.Special("setprofile", lambda *a: None),
            "setrecursionlimit": _vm.Special("setrecursionlimit", lambda n: None),
            "getdefaultencoding": _vm.Special("getdefaultencoding", lambda: "utf-8"),
            "stdout": _vm.Opaque("sys.stdout"), "stderr": _vm.Opaque("sys.stderr"), "stdin": _vm.Opaque("sys.stdin"),
        })
    elif name == "os":
        win = vm.platform == "windows"
        mod = _vm.SafeModule("os", {
            "name": "nt" if win else "posix", "sep": "\\" if win else "/", "linesep": "\r\n" if win else "\n",
            "path": get_module(vm, "os.path"), "_exit": vm.builtins["exit"],
            "getcwd": _vm.Special("getcwd", lambda: FAKE_CWD[vm.platform], "nt" if win else "posix"),
            "environ": dict(FAKE_ENV[vm.platform]),
            "getenv": _vm.Special("getenv", lambda k, d=None: FAKE_ENV[vm.platform].get(k, d), "os"),
            # recorded, never performed (self-deleting samples; anti-tamper checks also probe their repr)
            **{n: _vm.Special(n, _fs_noop(vm, n), "nt" if win else "posix") for n in ("unlink", "remove", "rmdir")},
            "pathsep": ";" if win else ":", "curdir": ".", "pardir": "..",
        })
    elif name == "threading":
        # a single main thread: anti-debug checks count threads (debuggers/injectors add some)
        one = _vm.Special("active_count", lambda: 1, "threading")
        mod = _vm.SafeModule("threading", {"active_count": one, "activeCount": one})
    elif name == "inspect":
        def stack(context=1):
            # the executing offset of each emulated frame lives in its _exec generator's locals
            offsets, fr = {}, sys._getframe(1)
            while fr:
                if fr.f_code.co_name == "_exec" and "ins" in fr.f_locals and "f" in fr.f_locals:
                    offsets.setdefault(id(fr.f_locals["f"]), fr.f_locals["ins"].offset)
                fr = fr.f_back
            return [FrameInfo(_vm.Opaque("frame"), f.code.co_filename,
                              _vm.bc.line_at(f.code, f.version, offsets.get(id(f), 0)), f.code.co_name, None, None)
                    for f in reversed(vm.frames)]

        mod = _vm.SafeModule("inspect", {"stack": _vm.Special("stack", stack, "inspect")})
    elif name == "time":
        clock = [1_700_000_000.0]

        def _t():
            clock[0] += 0.001
            return clock[0]

        def _ns():
            return int(_t() * 1e9)

        mod = _vm.SafeModule("time", {
            "time": _vm.Special("time", _t), "perf_counter": _vm.Special("perf_counter", _t),
            "monotonic": _vm.Special("monotonic", _t), "process_time": _vm.Special("process_time", _t),
            **{n: _vm.Special(n, _ns) for n in ("time_ns", "perf_counter_ns", "monotonic_ns", "process_time_ns")},
            "sleep": _vm.Special("sleep", lambda s=0: vm.event(f"time.sleep({_vm.describe(s)})")),
        })
    elif name == "types":
        mod = _vm.SafeModule("types", {
            "FunctionType": _vm.Special("FunctionType", vm.make_function_type),
            "LambdaType": _vm.Special("LambdaType", vm.make_function_type),
            "ModuleType": _vm.Special("ModuleType", lambda name, doc=None: _vm.SafeModule(
                str(name), {"__name__": name, "__doc__": doc})),
            # isinstance(exec, BuiltinFunctionType) integrity checks
            "BuiltinFunctionType": types.BuiltinFunctionType, "BuiltinMethodType": types.BuiltinMethodType,
            "MethodType": types.MethodType, "CodeType": types.CodeType,
        })
    elif name in ("_bz2", "_lzma", "_hashlib"):  # the C modules behind bz2/lzma/hashlib
        mod = _mod(name, vm._static_mods[name[1:]])
    elif name in vm._static_mods:
        mod = _mod(name, vm._static_mods[name])
        if name in ("os.path", "ntpath", "posixpath"):
            isdir = mod.attrs["isdir"].fn
            here = mod.attrs["abspath"].fn(vm.script_name).lower()

            def isfile(p):  # only the script itself exists
                return isinstance(p, str) and mod.attrs["abspath"].fn(p).lower() == here

            mod.attrs.update(isfile=_vm.Special("isfile", isfile, name),
                             exists=_vm.Special("exists", lambda p: isfile(p) or isdir(p), name))
    elif name.split(".")[0] in _CRYPTO_ROOTS:
        _load_crypto(vm)
        return vm.modules.get(name)
    if mod is not None:
        vm.modules[key] = mod
    return mod


# ----------------------------------------------------------------- builtins

def make_builtins(vm):
    V = _vm
    B = {}
    for n in _PURE_BUILTINS:
        B[n] = getattr(_b, n)
    for t in (str, bytes, bytearray, int, float, complex, bool, list, tuple, dict, set, frozenset, range, slice,
              object, memoryview, enumerate, zip, map, filter, reversed):
        B[t.__name__] = t
    for n, e in vars(_b).items():  # by builtins name: IOError/EnvironmentError are aliases of OSError
        if isinstance(e, type) and issubclass(e, BaseException):
            B[n] = e
    if vm.platform == "windows":
        B["WindowsError"] = OSError
    B.update({"None": None, "True": True, "False": False, "Ellipsis": Ellipsis, "NotImplemented": NotImplemented,
              "__debug__": True, "__name__": "builtins"})

    def _exit(*a):
        vm.event(f"exit({', '.join(V.describe(x) for x in a)})")
        raise V.VMStop("exit() called")

    def _print(*args, sep=" ", end="\n", file=None, flush=False):
        parts = []
        for a in args:
            try:
                parts.append(str(a) if not isinstance(a, V.Unknown) else V.describe(a))
            except V.VMSignal:
                parts.append("<?>")
        vm.event("print: " + V.describe(sep.join(parts) if isinstance(sep, str) else " ".join(parts), 300))

    def _input(prompt=""):
        vm.event(f"input({V.describe(prompt)})")
        return ""

    def _globals():
        return vm.current_globals()

    def _locals():
        f = vm.frame
        if f is None:
            return vm.new_globals()
        if f.locals is not None:
            return f.locals
        return f.globals if f.is_module else dict(f.fast)

    def _vars(*a):
        if not a:
            return _locals()
        o = a[0]
        if isinstance(o, V.VMInstance):
            return o.dict
        if isinstance(o, V.VMClass):
            return o.ns
        if isinstance(o, V.SafeModule):
            return o.attrs
        raise V.VMUnsupported("vars() of host object")

    def _getattr(o, name, *default):
        if not isinstance(name, str):
            if isinstance(name, V.Unknown):
                return V.Unknown("getattr with unknown name")
            raise V.VMRaise(TypeError("attribute name must be string"))
        try:
            return vm.getattr(o, name)
        except V.VMRaise as e:
            if default and isinstance(e.exc, AttributeError):
                return default[0]
            raise

    def _hasattr(o, name):
        try:
            vm.getattr(o, name)
            return True
        except V.VMRaise as e:
            if isinstance(e.exc, AttributeError):
                return False
            raise

    def _setattr(o, name, value):
        vm.setattr(o, name, value)

    def _delattr(o, name):
        vm.delattr(o, name)

    def _isinstance(o, t):
        if isinstance(t, tuple):
            return any(_isinstance(o, x) for x in t)
        if isinstance(t, V.VMClass):
            return isinstance(o, V.VMInstance) and t in o.cls.mro()
        if isinstance(o, V.Unknown):
            return V.Unknown("isinstance of unknown")
        if isinstance(t, type):
            if isinstance(o, V.VMInstance):
                return t is object
            if isinstance(o, V.VMFunction):
                return t in (object, types.FunctionType)
            if isinstance(o, V.Special):
                return t in (object, types.BuiltinFunctionType)
            if isinstance(o, V.BoundMethod):
                return t in (object, types.MethodType)
            if isinstance(o, V.CodeRef):
                return t in (object, types.CodeType)
            if isinstance(o, V.VMObject):
                return t is object or (t is type and isinstance(o, V.VMClass))
            return isinstance(o, t)
        if isinstance(t, V.Special) and t.__name__ == "type":
            return isinstance(o, (type, V.VMClass))
        if isinstance(t, V.Special) and t.__name__ in ("FunctionType", "LambdaType"):
            return isinstance(o, V.VMFunction)
        if isinstance(t, V.Special) and t.__name__ == "ModuleType":
            return isinstance(o, V.SafeModule)
        raise V.VMRaise(TypeError("isinstance() arg 2 must be a type or tuple of types"))

    def _issubclass(c, t):
        if isinstance(t, tuple):
            return any(_issubclass(c, x) for x in t)
        if isinstance(c, V.VMClass):
            return t is object or (isinstance(t, V.VMClass) and t in c.mro())
        if isinstance(c, type) and isinstance(t, type):
            return issubclass(c, t)
        return False

    def _type(*a):
        if len(a) == 1:
            o = a[0]
            if isinstance(o, V.VMInstance):
                return o.cls
            if isinstance(o, V.Unknown):
                return V.Unknown("type of unknown")
            # the real types, so anti-tamper checks like str(type(print)) see what CPython shows;
            # calling them stays emulated (type) or refused (FunctionType)
            if isinstance(o, (type, V.VMClass)):
                return type
            if isinstance(o, V.VMFunction):
                return types.FunctionType
            if isinstance(o, V.Special):  # emulated type/super/... are classes in CPython
                return type if isinstance(getattr(_b, o.__name__, None), type) else types.BuiltinFunctionType
            t = type(o)
            if t in SAFE_TYPES:
                return t
            return V.Opaque(f"type<{t.__name__}>")
        if len(a) == 3:
            name, bases, ns = a
            return V.VMClass(vm, name, bases, dict(ns))
        raise V.VMRaise(TypeError("type() takes 1 or 3 arguments"))

    def _callable(o):
        if isinstance(o, V.Unknown):
            return V.Unknown("callable of unknown")
        if isinstance(o, (V.VMFunction, V.VMClass, V.BoundMethod, V.Special)):
            return True
        if isinstance(o, V.VMInstance):
            return o.cls.lookup("__call__") is not V.EMPTY
        return callable(o)

    def _iter(o, *sentinel):
        if sentinel:
            if not isinstance(o, (V.VMFunction, V.BoundMethod)):
                raise V.VMUnsupported("iter(callable, sentinel) with host callable")
            return iter(o, sentinel[0])
        return vm.iter_(o)

    def _dir(*a):
        if not a:
            return sorted(_locals().keys())
        o = a[0]
        if isinstance(o, V.SafeModule):
            return sorted(o.attrs)
        if isinstance(o, V.VMInstance):
            return sorted(set(o.dict) | {k for c in o.cls.mro() for k in c.ns})
        if isinstance(o, V.VMObject) or isinstance(o, V.Unknown):
            return []
        if type(o) in SAFE_INSTANCE_TYPES or o in SAFE_TYPES:
            return dir(o)
        raise V.VMUnsupported("dir of host object")

    def _import(name, globals=None, locals=None, fromlist=(), level=0):
        return vm.import_(name, fromlist, level)

    def _noop(*a, **k):
        return None

    specials = {
        "exec": lambda o, g=None, l=None, **kw: vm.do_exec(o, g, l),
        "eval": vm.do_eval,
        "compile": vm.do_compile,
        "globals": _globals,
        "locals": _locals,
        "vars": _vars,
        "getattr": _getattr,
        "hasattr": _hasattr,
        "setattr": _setattr,
        "delattr": _delattr,
        "isinstance": _isinstance,
        "issubclass": _issubclass,
        "type": _type,
        "callable": _callable,
        "iter": _iter,
        "dir": _dir,
        "__import__": _import,
        "__build_class__": vm.build_class,
        "open": vm.do_open,
        "print": _print,
        "input": _input,
        "exit": _exit,
        "quit": _exit,
        "breakpoint": _noop,
        "help": _noop,
        "staticmethod": lambda f: V.StaticMethod(f),
        "classmethod": lambda f: V.ClassMethod(f),
        "property": lambda fget=None, fset=None, fdel=None, doc=None: V.Property(fget, fset, fdel, doc),
        "super": lambda *a: (_ for _ in ()).throw(V.VMUnsupported("super()")),
    }
    for k, fn in specials.items():
        B[k] = V.Special(k, fn)
    return B

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
SAFE_INSTANCE_TYPES = frozenset(PRIMITIVE_TYPES + _ITER_TYPES)

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


def guard_binop(op, a, b):
    o = op.rstrip("=")
    if o == "*":
        seq, n = (a, b) if isinstance(b, int) else (b, a)
        if isinstance(seq, (str, bytes, bytearray, list, tuple)) and isinstance(n, int) and len(seq) * n > _vm.SIZE_LIMIT:
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
    if name in _SIZE_SENSITIVE or isinstance(fn, type):
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
    for v in attrs.values():
        if callable(v) and not isinstance(v, _vm.VMObject):
            _allow(v)
    return _vm.SafeModule(name, attrs)


def _pick(module, names):
    return {n: getattr(module, n) for n in names if hasattr(module, n)}


_rng = random.Random(1337)


def _static_modules(platform):
    win = platform == "windows"
    pathmod = ntpath if win else posixpath
    path_attrs = _pick(pathmod, ["join", "basename", "dirname", "splitext", "split", "normpath", "isabs",
                                 "normcase", "splitdrive", "sep", "extsep", "altsep", "pathsep", "curdir", "pardir"])
    return {
        "base64": _pick(base64, ["b64decode", "b64encode", "b32decode", "b32encode", "b16decode", "b16encode",
                                 "b85decode", "b85encode", "a85decode", "a85encode", "urlsafe_b64decode",
                                 "urlsafe_b64encode", "standard_b64decode", "standard_b64encode", "decodebytes",
                                 "encodebytes", "b32hexdecode", "b32hexencode"]),
        "zlib": {**_pick(zlib, ["compress", "crc32", "adler32", "MAX_WBITS", "DEFLATED", "error"]),
                 "decompress": zlib_decompress},
        "bz2": {"compress": bz2.compress, "decompress": bz2_decompress},
        "lzma": {"compress": lzma.compress, "decompress": lzma_decompress,
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
        "hashlib": _pick(hashlib, ["md5", "sha1", "sha224", "sha256", "sha384", "sha512", "blake2b", "blake2s",
                                   "sha3_256", "sha3_512"]),
        "random": {n: getattr(_rng, n) for n in ["random", "randint", "choice", "shuffle", "randrange", "sample",
                                                  "seed", "uniform", "choices", "getrandbits"]},
        "json": _pick(json, ["loads", "dumps", "JSONDecodeError"]),
        "ast": {"literal_eval": _ast.literal_eval},
        "collections": _pick(collections, ["OrderedDict", "deque", "Counter", "defaultdict"]),
        "os.path": path_attrs,
        "ntpath": _pick(ntpath, list(path_attrs)),
        "posixpath": _pick(posixpath, list(path_attrs)),
    }


def functools_reduce(fn, seq, *init):
    import functools as ft
    return ft.reduce(fn, seq, *init)


_allow(functools_reduce)


def get_module(vm, name):
    if name in vm.modules:
        return vm.modules[name]
    if not hasattr(vm, "_static_mods"):
        vm._static_mods = _static_modules(vm.platform)
    mod = None
    if name == "builtins":
        mod = vm.builtins_module
    elif name == "marshal":
        mod = _vm.SafeModule("marshal", {"loads": _vm.Special("loads", vm.marshal_loads),
                                         "load": _vm.Special("load", lambda f: vm.marshal_loads(f.read()))})
    elif name == "sys":
        mod = _vm.SafeModule("sys", {
            "version_info": sys.version_info, "version": sys.version, "hexversion": sys.hexversion,
            "platform": "win32" if vm.platform == "windows" else "linux", "argv": ["main.py"],
            "maxsize": sys.maxsize, "byteorder": "little", "executable": "python",
            "exit": vm.builtins["exit"], "modules": {}, "flags": _vm.Opaque("sys.flags"),
            "getrecursionlimit": _vm.Special("getrecursionlimit", lambda: 1000),
            "setrecursionlimit": _vm.Special("setrecursionlimit", lambda n: None),
            "getdefaultencoding": _vm.Special("getdefaultencoding", lambda: "utf-8"),
            "stdout": _vm.Opaque("sys.stdout"), "stderr": _vm.Opaque("sys.stderr"), "stdin": _vm.Opaque("sys.stdin"),
        })
    elif name == "os":
        win = vm.platform == "windows"
        mod = _vm.SafeModule("os", {
            "name": "nt" if win else "posix", "sep": "\\" if win else "/", "linesep": "\r\n" if win else "\n",
            "path": get_module(vm, "os.path"), "_exit": vm.builtins["exit"],
            "pathsep": ";" if win else ":", "curdir": ".", "pardir": "..",
        })
    elif name == "time":
        clock = [1_700_000_000.0]

        def _t():
            clock[0] += 0.001
            return clock[0]

        mod = _vm.SafeModule("time", {
            "time": _vm.Special("time", _t), "perf_counter": _vm.Special("perf_counter", _t),
            "monotonic": _vm.Special("monotonic", _t),
            "sleep": _vm.Special("sleep", lambda s=0: vm.event(f"time.sleep({_vm.describe(s)})")),
        })
    elif name == "types":
        mod = _vm.SafeModule("types", {"FunctionType": _vm.Special("FunctionType", vm.make_function_type),
                                       "LambdaType": _vm.Special("LambdaType", vm.make_function_type)})
    elif name in vm._static_mods:
        mod = _mod(name, vm._static_mods[name])
    if mod is not None:
        vm.modules[name] = mod
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
    for e in _EXC_TYPES:
        B[e.__name__] = e
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
            if isinstance(o, V.VMObject):
                return t is object or (t is type and isinstance(o, V.VMClass))
            return isinstance(o, t)
        if isinstance(t, V.Special) and t.__name__ == "type":
            return isinstance(o, (type, V.VMClass))
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
            if isinstance(o, V.VMFunction):
                return V.Opaque("function")
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

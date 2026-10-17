"""A sandboxed CPython bytecode emulator (3.6 - 3.15).

The emulator never hands sample code to the host interpreter.  It walks the
bytecode itself and only ever calls host functions that are on an explicit
allow-list of side-effect free primitives (str/bytes methods, base64, zlib,
...).  ``exec``/``eval``/``marshal.loads`` and friends are replaced by
interceptors that *record* their argument as a new layer instead of running
it.  Anything else (imports of real modules, file/network/os access, ...) is
represented by opaque placeholder values and logged as an event.
"""

from __future__ import annotations

import functools
import operator
import time
import types
from dataclasses import dataclass, field

from . import bytecode as bc
from . import safelib

# ----------------------------------------------------------------- signals


class VMSignal(BaseException):
    """Base for emulator control-flow signals (BaseException so that host
    code which catches ``Exception`` can never swallow them)."""


class VMUnknownError(VMSignal):
    """An operation needed the concrete value of something unknown."""


class VMUnsupported(VMSignal):
    """The emulator does not (or refuses to) model this construct."""


class VMStop(VMSignal):
    """Emulation must stop (exit() called, budget exhausted, ...)."""


class VMRaise(VMSignal):
    """A Python-level exception raised inside emulated code."""

    def __init__(self, exc):
        super().__init__(exc)
        self.exc = exc


# ------------------------------------------------------------------ values


class _Null:
    __slots__ = ()

    def __repr__(self):
        return "<NULL>"


NULL = _Null()
EMPTY = object()


def _unknown_op(self, *a, **k):
    raise VMUnknownError(self.reason)


class Unknown:
    """A value the emulator cannot know.  Any attempt by host code to inspect
    it raises ``VMUnknownError``."""

    __slots__ = ("reason",)

    def __init__(self, reason="unknown"):
        object.__setattr__(self, "reason", reason)

    for _n in (
        "__str__ __repr__ __format__ __bool__ __iter__ __len__ __hash__ __eq__ __ne__ __lt__ __le__ __gt__ __ge__ "
        "__int__ __float__ __index__ __add__ __radd__ __sub__ __rsub__ __mul__ __rmul__ __mod__ __rmod__ __truediv__ "
        "__floordiv__ __pow__ __xor__ __rxor__ __and__ __or__ __lshift__ __rshift__ __neg__ __invert__ __getitem__ "
        "__contains__ __call__ __bytes__ __reversed__ __abs__ __setitem__"
    ).split():
        locals()[_n] = _unknown_op
    del _n


class Opaque(Unknown):
    """An unknown value with a provenance path (e.g. ``os.getenv``)."""

    __slots__ = ()


def describe(v, limit=120) -> str:
    try:
        if isinstance(v, Opaque):
            return f"<{v.reason}>"
        if isinstance(v, Unknown):
            return "<?>"
        if isinstance(v, (str, bytes, int, float, bool, type(None))):
            r = repr(v)
        elif isinstance(v, (list, tuple, dict, set)):
            r = None
            if len(v) <= 16:
                try:
                    items = v.items() if isinstance(v, dict) else v
                    if isinstance(v, dict):
                        r = "{" + ", ".join(f"{describe(k, 40)}: {describe(x, 60)}" for k, x in items) + "}"
                    else:
                        inner = ", ".join(describe(x, 60) for x in items)
                        r = {list: "[%s]", set: "{%s}"}.get(type(v), "(%s)") % (inner + ("," if isinstance(v, tuple) and len(v) == 1 else ""))
                except VMSignal:
                    r = None
            if r is None:
                r = type(v).__name__ + f"[{len(v)}]"
        else:
            r = f"<{type(v).__name__}>"
    except VMSignal:
        r = "<?>"
    return r if len(r) <= limit else r[: limit - 3] + "..."


class Cell:
    __slots__ = ("value",)

    def __init__(self, value=EMPTY):
        self.value = value


class VMObject:
    """Base for emulator-owned objects; host code may hold but never run them."""


class VMFunction(VMObject):
    def __init__(self, vm, code, version, globals_, name=None, qualname=None, defaults=None,
                 kwdefaults=None, closure=None, annotations=None, stub=False):
        self.vm = vm
        self.code = code
        self.version = version
        self.globals = globals_
        self.__name__ = name or getattr(code, "co_name", "<fn>")
        self.__qualname__ = qualname or self.__name__
        self.defaults = tuple(defaults) if defaults else ()
        self.kwdefaults = dict(kwdefaults) if kwdefaults else {}
        self.closure = tuple(closure) if closure else ()
        self.annotations = annotations
        self.attrs = {}
        self.stub = stub
        self.payload = None  # set when built from a payload code object

    def __call__(self, *args, **kwargs):
        return self.vm.call(self, args, kwargs)

    def __repr__(self):
        return f"<VMFunction {self.__qualname__}>"


class BoundMethod(VMObject):
    def __init__(self, func, obj):
        self.func = func
        self.obj = obj

    def __call__(self, *args, **kwargs):
        return self.func.vm.call(self.func, (self.obj,) + args, kwargs) if isinstance(self.func, VMFunction) \
            else _call_host(self.func, (self.obj,) + args, kwargs)


def _call_host(f, args, kwargs):
    return f(*args, **kwargs)


class StaticMethod(VMObject):
    def __init__(self, f):
        self.f = f


class ClassMethod(VMObject):
    def __init__(self, f):
        self.f = f


class Property(VMObject):
    def __init__(self, fget=None, fset=None, fdel=None, doc=None):
        self.fget, self.fset = fget, fset


class VMClass(VMObject):
    def __init__(self, vm, name, bases, ns):
        self.vm = vm
        self.__name__ = name
        self.bases = tuple(b for b in bases if b is not object)
        self.ns = ns

    def mro(self):
        out, seen = [], set()

        def walk(c):
            if id(c) in seen:
                return
            seen.add(id(c))
            out.append(c)
            for b in getattr(c, "bases", ()):
                if isinstance(b, VMClass):
                    walk(b)

        walk(self)
        return out

    def lookup(self, name):
        for c in self.mro():
            if name in c.ns:
                return c.ns[name]
        return EMPTY

    def native_bases(self):
        return [b for c in self.mro() for b in c.bases if not isinstance(b, VMClass)]

    def __call__(self, *args, **kwargs):
        return self.vm.call(self, args, kwargs)

    def __repr__(self):
        return f"<VMClass {self.__name__}>"


class VMInstance(VMObject):
    def __init__(self, cls):
        self.cls = cls
        self.dict = {}

    def _dunder(self, name, *args):
        f = self.cls.lookup(name)
        if f is EMPTY:
            raise AttributeError(name)
        return self.cls.vm.call(f, (self,) + args, {})

    def __call__(self, *args, **kwargs):
        return self._dunder("__call__", *args) if not kwargs else self.cls.vm.call(self.cls.lookup("__call__"), (self,) + args, kwargs)

    def __str__(self):
        try:
            return self._dunder("__str__")
        except AttributeError:
            return f"<{self.cls.__name__} object>"

    __repr__ = __str__

    def __iter__(self):
        return iter(self._dunder("__iter__"))

    def __next__(self):
        return self._dunder("__next__")

    def __len__(self):
        return self._dunder("__len__")

    def __getitem__(self, k):
        return self._dunder("__getitem__", k)


class VMGenerator(VMObject):
    def __init__(self, vm, gen, name):
        self.vm = vm
        self.gen = gen
        self.__name__ = name
        self.started = False

    def __iter__(self):
        return self

    def __next__(self):
        return self.send(None)

    def send(self, value):
        if not self.started:
            self.started = True
            return next(self.gen)
        return self.gen.send(value)

    def close(self):
        self.gen.close()


class SafeModule(VMObject):
    def __init__(self, name, attrs):
        self.__name__ = name
        self.attrs = attrs

    def get(self, name):
        if name in self.attrs:
            return self.attrs[name]
        if name == "__name__":
            return self.__name__
        if name == "__dict__":
            return self.attrs
        return Opaque(f"{self.__name__}.{name}")

    def __repr__(self):
        return f"<module {self.__name__}>"


class Special(VMObject):
    """An emulator-implemented builtin (exec, eval, globals, open, ...)."""

    def __init__(self, name, fn):
        self.__name__ = name
        self.fn = fn

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)

    def __repr__(self):
        return f"<built-in function {self.__name__}>"


class CodeRef(VMObject):
    """A code object produced inside the emulator (e.g. by marshal.loads)."""

    def __init__(self, blob: bc.CodeBlob):
        self.blob = blob

    @property
    def code(self):
        return self.blob.code


class VMCompiled(VMObject):
    """Result of ``compile()`` inside the emulator: host code + its source."""

    def __init__(self, code, source, mode, filename):
        self.code = code
        self.source = source
        self.mode = mode
        self.filename = filename


class HelperObject(VMObject):
    """Emulator-side helper whose listed methods emulated code may call."""

    ALLOWED: frozenset = frozenset()


class FakeFile(HelperObject):
    ALLOWED = frozenset({"read", "readlines", "readline", "close", "seek", "tell", "name"})

    def __init__(self, data, name):
        self.data = data
        self.pos = 0
        self.name = name

    def read(self, n=-1):
        if n is None or n < 0:
            out = self.data[self.pos:]
        else:
            out = self.data[self.pos:self.pos + n]
        self.pos += len(out)
        return out

    def readline(self):
        nl = "\n" if isinstance(self.data, str) else b"\n"
        i = self.data.find(nl, self.pos)
        end = len(self.data) if i < 0 else i + 1
        out = self.data[self.pos:end]
        self.pos = end
        return out

    def readlines(self):
        out = self.read()
        return out.splitlines(True)

    def seek(self, pos, whence=0):
        self.pos = pos

    def tell(self):
        return self.pos

    def close(self):
        pass


# ------------------------------------------------------------------ result


@dataclass
class Payload:
    kind: str  # "source" | "code"
    value: object  # str or bc.CodeBlob
    via: str
    globals: dict | None = None
    children: list = field(default_factory=list)
    events: list = field(default_factory=list)
    error: str | None = None  # emulator limitation (unsupported / unknown value)
    raised: str | None = None  # the payload raised a Python exception
    stop: str | None = None
    emulated: bool = False  # inline emulation ran to completion


@dataclass
class EmuResult:
    payloads: list = field(default_factory=list)
    events: list = field(default_factory=list)
    error: str | None = None
    stop_reason: str | None = None
    steps: int = 0
    assumptions: int = 0
    globals: dict | None = None


# ------------------------------------------------------------------- frame


class Frame:
    __slots__ = ("code", "version", "ins", "index", "globals", "locals", "fast", "cells", "stack",
                 "blocks", "kwnames", "handled", "etable", "is_module")

    def __init__(self, code, version, globals_, locals_=None):
        self.code = code
        self.version = tuple(version)
        self.ins, self.index = bc.instructions(code, self.version)
        self.globals = globals_
        self.locals = locals_
        self.fast = {}
        self.cells = {}
        self.stack = []
        self.blocks = []
        self.kwnames = ()
        self.handled = []
        self.etable = bc.exception_table(code) if self.version >= (3, 11) else None
        self.is_module = False


class Block:
    __slots__ = ("kind", "handler", "level")

    def __init__(self, kind, handler, level):
        self.kind, self.handler, self.level = kind, handler, level


BINOPS = {
    "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv, "//": operator.floordiv,
    "%": operator.mod, "**": operator.pow, "<<": operator.lshift, ">>": operator.rshift, "&": operator.and_,
    "|": operator.or_, "^": operator.xor, "@": operator.matmul,
    "+=": operator.iadd, "-=": operator.isub, "*=": operator.imul, "/=": operator.itruediv,
    "//=": operator.ifloordiv, "%=": operator.imod, "**=": operator.ipow, "<<=": operator.ilshift,
    ">>=": operator.irshift, "&=": operator.iand, "|=": operator.ior, "^=": operator.ixor, "@=": operator.imatmul,
}
OLD_BINOPS = {
    "ADD": "+", "SUBTRACT": "-", "MULTIPLY": "*", "TRUE_DIVIDE": "/", "FLOOR_DIVIDE": "//", "MODULO": "%",
    "POWER": "**", "LSHIFT": "<<", "RSHIFT": ">>", "AND": "&", "OR": "|", "XOR": "^", "MATRIX_MULTIPLY": "@",
}
NB_OPS = ["+", "&", "//", "<<", "@", "*", "%", "|", "**", ">>", "-", "/", "^"]
NB_OPS = NB_OPS + [o + "=" for o in NB_OPS]
DUNDER_FOR = {"+": "__add__", "-": "__sub__", "*": "__mul__", "%": "__mod__", "==": "__eq__", "!=": "__ne__"}
COMPARES = {
    "<": operator.lt, "<=": operator.le, "==": operator.eq, "!=": operator.ne, ">": operator.gt, ">=": operator.ge,
}
SIZE_LIMIT = 1 << 28
HOST = bc.HOST_VERSION


class Emulator:
    def __init__(self, *, version_hint=None, max_steps=30_000_000, max_seconds=120.0, self_file=None,
                 self_file_data=None, platform="windows", max_depth=300):
        self.version_hint = version_hint
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.self_file = self_file
        self.self_file_data = self_file_data
        self.platform = platform
        self.max_depth = max_depth
        self.depth = 0
        self.steps = 0
        self.deadline = None
        self.frames: list[Frame] = []
        self.payloads: list[Payload] = []
        self.ctx: list[Payload] = []
        self._payload_keys = set()
        self.events: list[str] = []
        self.assumptions = 0
        self.code_versions = {}
        self.builtins = safelib.make_builtins(self)
        self.builtins_module = SafeModule("builtins", self.builtins)
        self.modules = {}

    # ------------------------------------------------------------ api

    def new_globals(self, name="__main__"):
        return {
            "__name__": name, "__file__": self.self_file or "<pydeobf>", "__doc__": None, "__package__": None,
            "__spec__": None, "__loader__": None, "__builtins__": self.builtins_module, "__cached__": None,
        }

    def run(self, code, version, globals_=None) -> EmuResult:
        res = EmuResult()
        g = globals_ if globals_ is not None else self.new_globals()
        g.setdefault("__builtins__", self.builtins_module)
        self.deadline = time.monotonic() + self.max_seconds
        try:
            self.run_code(code, version, g, None, module=True)
        except VMStop as e:
            res.stop_reason = str(e)
        except VMUnsupported as e:
            res.error = f"unsupported: {e}"
        except VMUnknownError as e:
            res.error = f"needed unknown value: {e}"
        except VMRaise as e:
            res.error = f"uncaught {type(e.exc).__name__}: {describe(str(e.exc) if not isinstance(e.exc, VMSignal) else '')}"
        except RecursionError:
            res.error = "recursion limit"
        except MemoryError:
            res.error = "memory limit"
        res.payloads = self.payloads
        res.events = self.events
        res.steps = self.steps
        res.assumptions = self.assumptions
        res.globals = g
        return res

    def event(self, msg):
        target = self.ctx[-1].events if self.ctx else self.events
        if len(target) < 2000:
            target.append(msg)

    def add_payload(self, kind, value, via, globals_) -> Payload:
        """Record a payload in the current payload tree (once per content)."""
        p = Payload(kind, value, via, globals_)
        key = (kind, value if kind == "source" else value.raw or id(value.code))
        if key not in self._payload_keys:
            self._payload_keys.add(key)
            (self.ctx[-1].children if self.ctx else self.payloads).append(p)
        return p

    def run_payload(self, p: Payload, g, l=None):
        """Emulate an exec'd payload inline (inside the sandbox) so that its side
        effects on ``g`` are visible to the code that exec'd it."""
        if p.kind == "source":
            try:
                code = compile(p.value, "<exec>", "exec", dont_inherit=True)  # compile only
            except (SyntaxError, ValueError) as e:
                p.raised = f"{type(e).__name__}: {e}"
                p.error = "payload does not compile"
                raise VMRaise(e if isinstance(e, SyntaxError) else SyntaxError(str(e)))
            version = HOST
        else:
            code, version = p.value.code, p.value.version
            if version not in bc.EMULATED_VERSIONS:
                p.error = f"Python {bc.vstr(version)} bytecode is not emulated"
                return
        self.ctx.append(p)
        try:
            f = Frame(code, version, g, l)
            f.is_module = True
            self.run_frame(f)
            p.emulated = True
        except VMRaise as e:
            p.raised = f"{type(e.exc).__name__}: {describe(str(e.exc), 200)}"
            raise
        except VMStop as e:
            p.stop = str(e)
            raise
        except (VMUnsupported, VMUnknownError) as e:
            p.error = f"{'unsupported' if isinstance(e, VMUnsupported) else 'needed unknown value'}: {e}"
        except (RecursionError, MemoryError) as e:
            p.error = type(e).__name__
        finally:
            self.ctx.pop()

    @property
    def frame(self):
        return self.frames[-1] if self.frames else None

    def current_globals(self):
        return self.frame.globals if self.frames else self.new_globals()

    # ------------------------------------------------------ execution

    def run_code(self, code, version, globals_, locals_, module=False):
        f = Frame(code, version, globals_, locals_)
        f.is_module = module
        if not module and locals_ is not None:
            f.is_module = True  # exec-style frame using a namespace dict
        return self.run_frame(f)

    def run_frame(self, f):
        gen = self._exec(f)
        try:
            next(gen)
        except StopIteration as e:
            return e.value
        raise VMUnsupported("yield outside generator")

    def _enter(self):
        self.depth += 1
        if self.depth > self.max_depth:
            self.depth -= 1
            raise VMStop("call depth limit")

    def _exec(self, f: Frame):
        self._enter()
        self.frames.append(f)
        try:
            ins_list, index, stack = f.ins, f.index, f.stack
            i = 0
            n = len(ins_list)
            handlers = HANDLERS
            while True:
                if i >= n:
                    raise VMUnsupported("fell off end of code")
                self.steps += 1
                if self.steps > self.max_steps:
                    raise VMStop("instruction budget exhausted")
                if not self.steps & 0xFFFF and time.monotonic() > self.deadline:
                    raise VMStop("time budget exhausted")
                ins = ins_list[i]
                op = ins.opname
                try:
                    if op == "RETURN_VALUE":
                        return stack.pop()
                    if op == "RETURN_CONST":
                        return f.code.co_consts[ins.arg]
                    if op == "YIELD_VALUE":
                        val = stack.pop()
                        sent = yield val
                        stack.append(sent)
                        i += 1
                        continue
                    if op == "FOR_ITER":
                        # 3.15: [iterator, index-or-NULL] under the next value
                        it = stack[-2] if f.version >= (3, 15) else stack[-1]
                        try:
                            stack.append(self.next_(it))
                            i += 1
                        except StopIteration:
                            i = index[ins.argval]
                            if f.version >= (3, 14):
                                # 3.14: skip END_FOR, POP_ITER pops the iterator
                                if i < n and ins_list[i].opname == "END_FOR":
                                    i += 1
                                continue
                            stack.pop()
                            if i < n and ins_list[i].opname == "END_FOR":
                                i += 1
                                if f.version >= (3, 13) and i < n and ins_list[i].opname == "POP_TOP":
                                    i += 1
                        continue
                    h = handlers.get(op)
                    if h is None:
                        raise VMUnsupported(f"opcode {op} ({bc.vstr(f.version)})")
                    r = h(self, f, ins)
                    i = i + 1 if r is None else index[r]
                except VMRaise as e:
                    target = self._unwind(f, ins, e.exc)
                    if target is None:
                        raise
                    i = index[target]
        finally:
            self.frames.pop()
            self.depth -= 1

    def _unwind(self, f, ins, exc):
        if f.version >= (3, 11):
            for start, end, target, depth, lasti in f.etable:
                if start <= ins.offset < end:
                    del f.stack[depth:]
                    if lasti:
                        f.stack.append(ins.offset)
                    f.stack.append(exc)
                    return target
            return None
        while f.blocks:
            b = f.blocks.pop()
            if b.kind == "loop":
                continue
            if b.kind == "handler":
                del f.stack[b.level:]
                if f.handled:
                    f.handled.pop()
                continue
            del f.stack[b.level:]
            f.blocks.append(Block("handler", None, len(f.stack)))
            prev = f.handled[-1] if f.handled else None
            f.stack.extend([None, prev, type(prev) if prev is not None else None])
            f.stack.extend([None, exc, type(exc)])
            f.handled.append(exc)
            return b.handler
        return None

    # --------------------------------------------------------- calls

    def call(self, fn, args, kwargs):
        if isinstance(fn, VMFunction):
            return self.call_function(fn, tuple(args), kwargs)
        if isinstance(fn, BoundMethod):
            return self.call(fn.func, (fn.obj,) + tuple(args), kwargs)
        if isinstance(fn, Special):
            return fn.fn(*args, **kwargs)
        if isinstance(fn, VMClass):
            return self.instantiate(fn, args, kwargs)
        if isinstance(fn, StaticMethod):
            return self.call(fn.f, args, kwargs)
        if isinstance(fn, Opaque):
            self.event(f"call {fn.reason}({', '.join([describe(a, 80) for a in args] + [f'{k}={describe(v, 80)}' for k, v in kwargs.items()])})")
            return Opaque(f"{fn.reason}()")
        if isinstance(fn, Unknown):
            return Unknown("result of unknown call")
        if isinstance(fn, VMInstance):
            f = fn.cls.lookup("__call__")
            if f is EMPTY:
                raise VMRaise(TypeError(f"'{fn.cls.__name__}' object is not callable"))
            return self.call(f, (fn,) + tuple(args), kwargs)
        if isinstance(fn, functools.partial):
            return self.call(fn.func, fn.args + tuple(args), {**fn.keywords, **kwargs})
        if isinstance(fn, types.MethodType) and isinstance(fn.__self__, HelperObject):
            return fn(*args, **kwargs)
        if not safelib.is_safe_callable(fn):
            raise VMUnsupported(f"refusing to call non-allow-listed {describe(fn)} {getattr(fn, '__qualname__', '')}")
        for a in args:
            if isinstance(a, Unknown):
                return Unknown("call with unknown argument")
        for a in kwargs.values():
            if isinstance(a, Unknown):
                return Unknown("call with unknown argument")
        safelib.guard_call(self, fn, args, kwargs)
        try:
            return fn(*args, **kwargs)
        except VMUnknownError:
            return Unknown("host call touched unknown value")
        except VMSignal:
            raise
        except RecursionError:
            raise
        except MemoryError:
            raise VMStop("memory limit")
        except Exception as e:  # a genuine Python-level exception
            raise VMRaise(e)

    def bind_args(self, fn: VMFunction, args, kwargs):
        code = fn.code
        argc = code.co_argcount
        posonly = getattr(code, "co_posonlyargcount", 0) or 0
        kwonly = code.co_kwonlyargcount
        flags = code.co_flags
        names = code.co_varnames
        params = names[:argc]
        kwonly_names = names[argc:argc + kwonly]
        j = argc + kwonly
        varargs = varkw = None
        if flags & 0x04:
            varargs = names[j]
            j += 1
        if flags & 0x08:
            varkw = names[j]
        fast = {}
        for name, val in zip(params, args):
            fast[name] = val
        extra = args[argc:]
        if extra and not varargs:
            raise VMRaise(TypeError(f"{fn.__name__}() takes {argc} positional arguments but {len(args)} were given"))
        if varargs:
            fast[varargs] = tuple(extra)
        kwd = {}
        for k, v in kwargs.items():
            if (k in params[posonly:]) or k in kwonly_names:
                if k in fast:
                    raise VMRaise(TypeError(f"{fn.__name__}() got multiple values for argument '{k}'"))
                fast[k] = v
            elif varkw:
                kwd[k] = v
            else:
                raise VMRaise(TypeError(f"{fn.__name__}() got an unexpected keyword argument '{k}'"))
        d = fn.defaults
        for idx, name in enumerate(params):
            if name not in fast:
                di = idx - (argc - len(d))
                if di < 0:
                    raise VMRaise(TypeError(f"{fn.__name__}() missing required argument: '{name}'"))
                fast[name] = d[di]
        for name in kwonly_names:
            if name not in fast:
                if name not in fn.kwdefaults:
                    raise VMRaise(TypeError(f"{fn.__name__}() missing required keyword-only argument: '{name}'"))
                fast[name] = fn.kwdefaults[name]
        if varkw:
            fast[varkw] = kwd
        return fast

    def setup_cells(self, f: Frame, closure):
        code = f.code
        cellvars = tuple(getattr(code, "co_cellvars", ()) or ())
        freevars = tuple(getattr(code, "co_freevars", ()) or ())
        for name in cellvars:
            f.cells[name] = Cell(f.fast.get(name, EMPTY))
        if closure:
            names = freevars if len(freevars) == len(closure) else (cellvars + freevars)[-len(closure):]
            for name, cell in zip(names, closure):
                f.cells[name] = cell
        for name in freevars:
            if name not in f.cells:
                f.cells[name] = Cell(f.fast.get(name, EMPTY))

    def call_function(self, fn: VMFunction, args, kwargs):
        if fn.stub:
            return None
        if fn.payload is not None and not args and not kwargs:
            self.run_payload(fn.payload, fn.globals)
            return None
        fast = self.bind_args(fn, args, kwargs)
        f = Frame(fn.code, fn.version, fn.globals, None)
        f.fast = fast
        self.setup_cells(f, fn.closure)
        flags = fn.code.co_flags
        if flags & (0x80 | 0x100 | 0x200):
            raise VMUnsupported("coroutines / async generators")
        if flags & 0x20:
            return VMGenerator(self, self._exec(f), fn.__name__)
        return self.run_frame(f)

    def instantiate(self, cls: VMClass, args, kwargs):
        nb = cls.native_bases()
        if nb:
            # Subclass of a host type (e.g. Exception): only allow pure data classes
            if all(isinstance(b, type) and issubclass(b, BaseException) for b in nb):
                return nb[0](*args)
            raise VMUnsupported(f"instantiating class with host base {nb}")
        if cls.lookup("__new__") is not EMPTY:
            raise VMUnsupported("custom __new__")
        inst = VMInstance(cls)
        init = cls.lookup("__init__")
        if init is not EMPTY:
            self.call(init, (inst,) + tuple(args), kwargs)
        return inst

    def build_class(self, func, name, *bases, metaclass=None, **kw):
        if metaclass is not None:
            raise VMUnsupported("metaclass")
        if not isinstance(func, VMFunction):
            raise VMUnsupported("class body is not emulated code")
        ns = {}
        f = Frame(func.code, func.version, func.globals, ns)
        self.setup_cells(f, func.closure)
        ret = self.run_frame(f)
        cls = VMClass(self, name, bases, ns)
        if isinstance(ret, Cell):
            ret.value = cls
        return cls

    # ------------------------------------------------- object model

    def getattr(self, obj, name):
        if isinstance(obj, Opaque):
            return Opaque(f"{obj.reason}.{name}")
        if isinstance(obj, Unknown):
            return Unknown("attribute of unknown")
        if isinstance(obj, SafeModule):
            return obj.get(name)
        if isinstance(obj, VMInstance):
            if name == "__dict__":
                return obj.dict
            if name == "__class__":
                return obj.cls
            if name in obj.dict:
                return obj.dict[name]
            v = obj.cls.lookup(name)
            if v is EMPTY:
                raise VMRaise(AttributeError(f"'{obj.cls.__name__}' object has no attribute '{name}'"))
            if isinstance(v, VMFunction):
                return BoundMethod(v, obj)
            if isinstance(v, ClassMethod):
                return BoundMethod(v.f, obj.cls)
            if isinstance(v, StaticMethod):
                return v.f
            if isinstance(v, Property):
                return self.call(v.fget, (obj,), {})
            return v
        if isinstance(obj, VMClass):
            if name == "__name__":
                return obj.__name__
            if name == "__dict__":
                return obj.ns
            v = obj.lookup(name)
            if v is EMPTY:
                raise VMRaise(AttributeError(f"type object '{obj.__name__}' has no attribute '{name}'"))
            if isinstance(v, ClassMethod):
                return BoundMethod(v.f, obj)
            if isinstance(v, StaticMethod):
                return v.f
            return v
        if isinstance(obj, VMFunction):
            if name in ("__name__", "__qualname__"):
                return getattr(obj, name)
            if name == "__code__":
                return CodeRef(bc.CodeBlob(obj.code, obj.version))
            if name == "__globals__":
                return obj.globals
            if name == "__defaults__":
                return obj.defaults or None
            if name in obj.attrs:
                return obj.attrs[name]
            raise VMRaise(AttributeError(name))
        if isinstance(obj, BoundMethod):
            if name == "__self__":
                return obj.obj
            if name == "__func__":
                return obj.func
            return self.getattr(obj.func, name)
        if isinstance(obj, VMGenerator):
            if name in ("send", "close", "__next__"):
                return getattr(obj, name if name != "__next__" else "__next__")
            raise VMUnsupported(f"generator attribute {name}")
        if isinstance(obj, CodeRef):
            if name.startswith("co_"):
                return getattr(obj.code, name)
            raise VMUnsupported(f"code attribute {name}")
        if isinstance(obj, HelperObject):
            if name in obj.ALLOWED:
                return getattr(obj, name)
            raise VMUnsupported(f"helper attribute {name}")
        if isinstance(obj, Special):
            if name in ("__name__", "__qualname__"):
                return obj.__name__
            raise VMUnsupported(f"attribute {name} of builtin")
        if isinstance(obj, BaseException):
            if name in ("args", "__class__", "msg", "errno", "strerror", "filename", "value", "code"):
                return getattr(obj, name)
            raise VMRaise(AttributeError(name))
        if safelib.attr_allowed(obj, name):
            try:
                return getattr(obj, name)
            except AttributeError as e:
                raise VMRaise(e)
        raise VMUnsupported(f"attribute {name!r} of {type(obj).__name__}")

    def setattr(self, obj, name, value):
        if isinstance(obj, VMInstance):
            p = obj.cls.lookup(name)
            if isinstance(p, Property) and p.fset is not None:
                self.call(p.fset, (obj, value), {})
                return
            obj.dict[name] = value
        elif isinstance(obj, VMClass):
            obj.ns[name] = value
        elif isinstance(obj, VMFunction):
            obj.attrs[name] = value
        elif isinstance(obj, Unknown):
            if isinstance(obj, Opaque):
                self.event(f"setattr {obj.reason}.{name} = {describe(value)}")
        else:
            raise VMUnsupported(f"setattr on {type(obj).__name__}")

    def delattr(self, obj, name):
        if isinstance(obj, VMInstance):
            obj.dict.pop(name, None)
        elif isinstance(obj, VMClass):
            obj.ns.pop(name, None)
        elif not isinstance(obj, Unknown):
            raise VMUnsupported("delattr")

    def subscr(self, obj, key):
        if isinstance(obj, Opaque):
            return Opaque(f"{obj.reason}[{describe(key, 40)}]")
        if isinstance(obj, Unknown) or isinstance(key, Unknown):
            return Unknown("subscript of unknown")
        if isinstance(obj, VMInstance):
            return self.call(self.getattr(obj, "__getitem__"), (key,), {})
        if isinstance(obj, VMObject):
            raise VMUnsupported(f"subscript of {type(obj).__name__}")
        try:
            return obj[key]
        except VMSignal:
            raise
        except Exception as e:
            raise VMRaise(e)

    def store_subscr(self, obj, key, value):
        if isinstance(obj, Unknown):
            return
        if isinstance(obj, VMInstance):
            self.call(self.getattr(obj, "__setitem__"), (key, value), {})
            return
        if not isinstance(obj, (dict, list, bytearray)):
            raise VMUnsupported(f"item assignment on {type(obj).__name__}")
        try:
            obj[key] = value
        except VMSignal:
            raise
        except Exception as e:
            raise VMRaise(e)

    def binop(self, op, a, b):
        if isinstance(a, Unknown) or isinstance(b, Unknown):
            return Unknown("arithmetic on unknown")
        if isinstance(a, VMInstance) or isinstance(b, VMInstance):
            if isinstance(a, VMInstance) and op.rstrip("=") in DUNDER_FOR:
                return self.call(self.getattr(a, DUNDER_FOR[op.rstrip("=")]), (b,), {})
            raise VMRaise(TypeError(f"unsupported operand type(s) for {op}"))
        safelib.guard_binop(op, a, b)
        try:
            return BINOPS[op](a, b)
        except VMSignal:
            raise
        except MemoryError:
            raise VMStop("memory limit")
        except Exception as e:
            raise VMRaise(e)

    def compare(self, op, a, b):
        if op == "is":
            return a is b
        if op == "is not":
            return a is not b
        if op in ("in", "not in"):
            r = self.contains(b, a)
            return r if isinstance(r, Unknown) else (r if op == "in" else not r)
        if op == "exception-match":
            return self.exc_match(a, b)
        if isinstance(a, Unknown) or isinstance(b, Unknown):
            return Unknown("comparison with unknown")
        if isinstance(a, VMObject) or isinstance(b, VMObject):
            if op == "==":
                return a is b
            if op == "!=":
                return a is not b
            raise VMRaise(TypeError(f"'{op}' not supported between instances of '{type(a).__name__}' and '{type(b).__name__}'"))
        try:
            return COMPARES[op](a, b)
        except VMSignal:
            raise
        except Exception as e:
            raise VMRaise(e)

    def contains(self, container, item):
        if isinstance(container, Unknown) or isinstance(item, Unknown):
            return Unknown("membership with unknown")
        if isinstance(container, VMObject):
            raise VMUnsupported("membership on emulated object")
        try:
            return item in container
        except VMSignal:
            raise
        except Exception as e:
            raise VMRaise(e)

    def exc_match(self, exc, typ):
        if isinstance(typ, tuple):
            return any(self.exc_match(exc, t) for t in typ)
        if isinstance(exc, BaseException):
            exc = type(exc)
        if isinstance(typ, type) and isinstance(exc, type):
            return issubclass(exc, typ)
        return False

    def truth(self, v):
        if isinstance(v, Unknown):
            raise VMUnknownError("branch on unknown value")
        if isinstance(v, VMInstance):
            for n in ("__bool__", "__len__"):
                if v.cls.lookup(n) is not EMPTY:
                    return bool(self.call(self.getattr(v, n), (), {}))
            return True
        if isinstance(v, VMObject):
            return True
        try:
            return bool(v)
        except VMSignal:
            raise
        except Exception as e:
            raise VMRaise(e)

    def iter_(self, v):
        if isinstance(v, Unknown):
            raise VMUnknownError("iteration over unknown value")
        if isinstance(v, (VMGenerator, VMInstance)):
            return iter(v)
        if isinstance(v, VMObject):
            raise VMUnsupported(f"iterating {type(v).__name__}")
        try:
            return iter(v)
        except VMSignal:
            raise
        except Exception as e:
            raise VMRaise(e)

    def next_(self, it):
        try:
            return next(it)
        except StopIteration:
            raise
        except VMSignal:
            raise
        except Exception as e:
            raise VMRaise(e)

    def unknown_branch(self, f, ins, default_jump):
        """Called when a conditional jump depends on an unknown value."""
        self.assumptions += 1
        if self.assumptions > 2000:
            raise VMStop("too many unknown branches")
        self.event(f"assumed branch at {f.code.co_name}:{ins.offset}")
        return default_jump

    # ------------------------------------------------ interceptors

    def _decode_src(self, obj):
        if isinstance(obj, (bytes, bytearray)):
            data = bytes(obj)
            try:
                return data.decode("utf-8")
            except UnicodeDecodeError:
                return data.decode("latin-1")
        return obj

    def do_exec(self, obj, g=None, l=None, *, via="exec"):
        if g is None:
            g = self.current_globals()
        if isinstance(g, Unknown):
            g = self.current_globals()
        if isinstance(l, Unknown):
            l = None
        if isinstance(obj, (str, bytes, bytearray)):
            self.event(f"{via}(<{len(obj)} chars>)")
            p = self.add_payload("source", self._decode_src(obj), via, g)
        elif isinstance(obj, VMCompiled):
            self.event(f"{via}(compile(<{len(obj.source)} chars>))")
            p = self.add_payload("source", obj.source, via, g)
        elif isinstance(obj, CodeRef):
            self.event(f"{via}(<code object {getattr(obj.code, 'co_name', '?')}>)")
            p = self.add_payload("code", obj.blob, via, g)
        elif bc.is_code(obj):
            ver = self.code_versions.get(id(obj), self.frame.version if self.frames else HOST)
            p = self.add_payload("code", bc.CodeBlob(obj, ver), via, g)
        else:
            self.event(f"{via}({describe(obj)}) - argument not recoverable")
            return None
        self.run_payload(p, g, l)
        return None

    def do_eval(self, obj, g=None, l=None):
        if isinstance(obj, (str, bytes, bytearray)):
            src = self._decode_src(obj)
            try:
                code = compile(src.strip(), "<eval>", "eval")  # parse only; run by the emulator
            except SyntaxError as e:
                raise VMRaise(e)
            except ValueError as e:
                raise VMRaise(e)
            self.event(f"eval({describe(src, 80)})")
            return self._eval_code(code, HOST, g, l)
        if isinstance(obj, VMCompiled):
            if obj.mode == "eval":
                return self._eval_code(obj.code, HOST, g, l)
            return self.do_exec(obj, g, l, via="eval")
        if isinstance(obj, CodeRef) or bc.is_code(obj):
            return self.do_exec(obj, g, l, via="eval")
        self.event(f"eval({describe(obj)}) - argument not recoverable")
        return Unknown("eval of unknown")

    def _eval_code(self, code, version, g, l):
        if g is None:
            fr = self.frame
            g = fr.globals
            if l is None and fr is not None:
                if fr.locals is not None:
                    l = fr.locals
                elif fr.is_module:
                    l = fr.globals  # module level: locals() is globals()
                else:
                    l = dict(fr.fast)
        f = Frame(code, version, g, l if l is not None else g)
        f.is_module = True
        return self.run_frame(f)

    def do_compile(self, source, filename="<string>", mode="exec", flags=0, dont_inherit=False, optimize=-1, **kw):
        if isinstance(source, Unknown):
            return Unknown("compile of unknown")
        if not isinstance(source, (str, bytes, bytearray)):
            raise VMUnsupported("compile() of non-string")
        src = self._decode_src(source)
        try:
            code = compile(src, str(filename), mode, dont_inherit=True)  # parse/compile only
        except (SyntaxError, ValueError) as e:
            raise VMRaise(e)
        return VMCompiled(code, src, mode, filename)

    def marshal_loads(self, data, *a, **k):
        if isinstance(data, Unknown):
            return Unknown("marshal of unknown")
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise VMRaise(TypeError("marshal.loads() argument must be bytes-like"))
        data = bytes(data)
        hint = self.version_hint
        if self.frames and self.frames[-1].version != HOST:
            hint = self.frames[-1].version
        blob = bc.load_marshal_best_effort(data, hint) if data[:1] in (b"c", b"\xe3") else None
        if blob is None:
            # not a code object: decode plain data (pure-Python unmarshaller)
            try:
                from xdis.unmarshal import load_code
                val = load_code(data, bc.magic_int_for(hint or HOST))
            except Exception:
                raise VMRaise(ValueError("bad marshal data"))
            if bc.is_code(val):
                raise VMUnsupported("marshal code object of unknown version")
            return val
        self.version_hint = blob.version
        self.code_versions[id(blob.code)] = blob.version
        extra = ""
        if blob.meta.get("ambiguous"):
            extra = f", bytecode also valid as {'/'.join(blob.meta['ambiguous'])}"
        if blob.meta.get("uncertain"):
            extra = ", version detection uncertain"
        self.event(f"marshal.loads -> code object (Python {bc.vstr(blob.version)}{extra}, {len(data)} bytes)")
        return CodeRef(blob)

    def make_function_type(self, code, globals_=None, name=None, argdefs=None, closure=None):
        if isinstance(code, CodeRef):
            blob = code.blob
        elif isinstance(code, VMCompiled):
            p = self.add_payload("source", code.source, "FunctionType", globals_)
            fn = VMFunction(self, code.code, HOST, globals_ if isinstance(globals_, dict) else self.current_globals())
            fn.payload = p
            return fn
        elif bc.is_code(code):
            blob = bc.CodeBlob(code, self.code_versions.get(id(code), self.frame.version))
        else:
            raise VMUnsupported("FunctionType of non-code")
        if getattr(blob.code, "co_name", "") == "<module>":
            g = globals_ if isinstance(globals_, dict) else self.current_globals()
            p = self.add_payload("code", blob, "FunctionType", g)
            fn = VMFunction(self, blob.code, blob.version, g)
            fn.payload = p
            return fn
        return VMFunction(self, blob.code, blob.version, globals_ if isinstance(globals_, dict) else self.current_globals(),
                          name=name, defaults=argdefs, closure=closure)

    def import_(self, name, fromlist=None, level=0):
        if isinstance(name, Unknown):
            return Opaque("<dynamic import>")
        if level:
            self.event(f"import (relative) {name}")
            return Opaque(f"<relative>.{name}")
        top = name.split(".")[0]
        mod = safelib.get_module(self, name)
        if mod is None:
            self.event(f"import {name}")
            return Opaque(name if fromlist else top)
        if fromlist:
            return mod
        return safelib.get_module(self, top) or mod

    def import_star(self, mod, ns):
        if isinstance(mod, SafeModule):
            for k, v in mod.attrs.items():
                if not k.startswith("_"):
                    ns[k] = v
        else:
            self.event(f"from {describe(mod)} import *")

    def do_open(self, file, mode="r", *a, **k):
        try:
            same = isinstance(file, str) and self.self_file is not None and file == self.self_file
        except VMSignal:
            same = False
        if same and not any(c in str(mode) for c in "wax+"):
            data = self.self_file_data or b""
            if "b" in str(mode):
                return FakeFile(data, file)
            return FakeFile(data.decode("utf-8", errors="ignore"), file)
        self.event(f"open({describe(file)}, {describe(mode)})")
        return Opaque(f"open({describe(file, 60)})")


# ================================================================ handlers

HANDLERS = {}


def handler(*names):
    def deco(fn):
        for n in names:
            HANDLERS[n] = fn
        return fn
    return deco


@handler("NOP", "RESUME", "PRECALL", "MAKE_CELL", "COPY_FREE_VARS", "SETUP_ANNOTATIONS_NOP", "EXTENDED_ARG",
         "NOT_TAKEN", "CACHE", "INSTRUMENTED_RESUME")
def _nop(vm, f, ins):
    if ins.opname == "SETUP_ANNOTATIONS_NOP":
        pass


@handler("POP_TOP")
def _pop_top(vm, f, ins):
    f.stack.pop()


@handler("PUSH_NULL")
def _push_null(vm, f, ins):
    f.stack.append(NULL)


@handler("GEN_START")
def _gen_start(vm, f, ins):
    if f.stack:
        f.stack.pop()


@handler("RETURN_GENERATOR")
def _ret_gen(vm, f, ins):
    f.stack.append(None)


@handler("ROT_TWO")
def _rot2(vm, f, ins):
    s = f.stack
    s[-1], s[-2] = s[-2], s[-1]


@handler("ROT_THREE")
def _rot3(vm, f, ins):
    s = f.stack
    s[-1], s[-2], s[-3] = s[-2], s[-3], s[-1]


@handler("ROT_FOUR")
def _rot4(vm, f, ins):
    s = f.stack
    s[-1], s[-2], s[-3], s[-4] = s[-2], s[-3], s[-4], s[-1]


@handler("ROT_N")
def _rotn(vm, f, ins):
    s = f.stack
    n = ins.arg
    s[-n:] = [s[-1]] + s[-n:-1]


@handler("DUP_TOP")
def _dup(vm, f, ins):
    f.stack.append(f.stack[-1])


@handler("DUP_TOP_TWO")
def _dup2(vm, f, ins):
    f.stack.extend(f.stack[-2:])


@handler("COPY")
def _copy(vm, f, ins):
    f.stack.append(f.stack[-ins.arg])


@handler("SWAP")
def _swap(vm, f, ins):
    s = f.stack
    s[-1], s[-ins.arg] = s[-ins.arg], s[-1]


@handler("LOAD_CONST")
def _load_const(vm, f, ins):
    f.stack.append(f.code.co_consts[ins.arg])


@handler("LOAD_SMALL_INT")
def _load_small_int(vm, f, ins):
    f.stack.append(ins.arg)


@handler("LOAD_ASSERTION_ERROR")
def _load_ae(vm, f, ins):
    f.stack.append(AssertionError)


# ---- names


def _builtin(vm, name):
    if name in vm.builtins:
        return vm.builtins[name]
    raise VMRaise(NameError(f"name '{name}' is not defined"))


@handler("LOAD_NAME")
def _load_name(vm, f, ins):
    name = ins.argval
    if f.locals is not None and name in f.locals:
        f.stack.append(f.locals[name])
    elif name in f.globals:
        f.stack.append(f.globals[name])
    else:
        f.stack.append(_builtin(vm, name))


@handler("STORE_NAME")
def _store_name(vm, f, ins):
    v = f.stack.pop()
    (f.locals if f.locals is not None else f.globals)[ins.argval] = v


@handler("DELETE_NAME")
def _del_name(vm, f, ins):
    ns = f.locals if f.locals is not None else f.globals
    if ins.argval not in ns:
        raise VMRaise(NameError(ins.argval))
    del ns[ins.argval]


@handler("LOAD_GLOBAL")
def _load_global(vm, f, ins):
    name = ins.argval
    v = f.globals[name] if name in f.globals else _builtin(vm, name)
    push_null = f.version >= (3, 11) and ins.arg & 1
    if push_null and f.version < (3, 13):
        f.stack.append(NULL)
    f.stack.append(v)
    if push_null and f.version >= (3, 13):
        f.stack.append(NULL)


@handler("STORE_GLOBAL")
def _store_global(vm, f, ins):
    f.globals[ins.argval] = f.stack.pop()


@handler("DELETE_GLOBAL")
def _del_global(vm, f, ins):
    f.globals.pop(ins.argval, None)


@handler("LOAD_FAST", "LOAD_FAST_CHECK", "LOAD_FAST_BORROW")
def _load_fast(vm, f, ins):
    name = ins.argval
    if f.version >= (3, 13) and name in f.cells:
        # 3.13 emits LOAD_CLOSURE as LOAD_FAST on the cell slot
        f.stack.append(f.cells[name])
    elif name in f.fast:
        f.stack.append(f.fast[name])
    elif name in f.cells and f.cells[name].value is not EMPTY:
        f.stack.append(f.cells[name].value)
    else:
        raise VMRaise(UnboundLocalError(f"local variable '{name}' referenced before assignment"))


@handler("LOAD_FAST_AND_CLEAR")
def _load_fast_clear(vm, f, ins):
    f.stack.append(f.fast.pop(ins.argval, NULL))


@handler("LOAD_FAST_LOAD_FAST", "LOAD_FAST_BORROW_LOAD_FAST_BORROW")
def _load_fast2(vm, f, ins):
    for name in ins.argval:
        if name not in f.fast:
            raise VMRaise(UnboundLocalError(name))
        f.stack.append(f.fast[name])


def _set_fast(f, name, v):
    if v is NULL:
        f.fast.pop(name, None)
    else:
        f.fast[name] = v


@handler("STORE_FAST")
def _store_fast(vm, f, ins):
    _set_fast(f, ins.argval, f.stack.pop())


@handler("STORE_FAST_LOAD_FAST")
def _store_load_fast(vm, f, ins):
    a, b = ins.argval
    _set_fast(f, a, f.stack.pop())
    f.stack.append(f.fast[b])


@handler("STORE_FAST_STORE_FAST")
def _store_fast2(vm, f, ins):
    a, b = ins.argval
    _set_fast(f, a, f.stack.pop())
    _set_fast(f, b, f.stack.pop())


@handler("DELETE_FAST")
def _del_fast(vm, f, ins):
    f.fast.pop(ins.argval, None)


def _cell(f, name):
    c = f.cells.get(name)
    if c is None:
        c = f.cells[name] = Cell(f.fast.get(name, EMPTY))
    return c


@handler("LOAD_CLOSURE")
def _load_closure(vm, f, ins):
    f.stack.append(_cell(f, ins.argval))


@handler("LOAD_DEREF")
def _load_deref(vm, f, ins):
    v = _cell(f, ins.argval).value
    if v is EMPTY:
        raise VMRaise(NameError(f"free variable '{ins.argval}' referenced before assignment"))
    f.stack.append(v)


@handler("LOAD_CLASSDEREF")
def _load_classderef(vm, f, ins):
    if f.locals is not None and ins.argval in f.locals:
        f.stack.append(f.locals[ins.argval])
    else:
        _load_deref(vm, f, ins)


@handler("LOAD_FROM_DICT_OR_DEREF")
def _load_dict_or_deref(vm, f, ins):
    d = f.stack.pop()
    if isinstance(d, dict) and ins.argval in d:
        f.stack.append(d[ins.argval])
    else:
        _load_deref(vm, f, ins)


@handler("LOAD_FROM_DICT_OR_GLOBALS")
def _load_dict_or_globals(vm, f, ins):
    d = f.stack.pop()
    name = ins.argval
    if isinstance(d, dict) and name in d:
        f.stack.append(d[name])
    elif name in f.globals:
        f.stack.append(f.globals[name])
    else:
        f.stack.append(_builtin(vm, name))


@handler("LOAD_LOCALS")
def _load_locals(vm, f, ins):
    f.stack.append(f.locals if f.locals is not None else f.fast)


@handler("STORE_DEREF")
def _store_deref(vm, f, ins):
    _cell(f, ins.argval).value = f.stack.pop()


@handler("DELETE_DEREF")
def _del_deref(vm, f, ins):
    _cell(f, ins.argval).value = EMPTY


# ---- attributes


@handler("LOAD_ATTR")
def _load_attr(vm, f, ins):
    obj = f.stack.pop()
    v = vm.getattr(obj, ins.argval)
    method = f.version >= (3, 12) and ins.arg & 1
    if method and f.version < (3, 13):
        f.stack.append(NULL)
    f.stack.append(v)
    if method and f.version >= (3, 13):
        f.stack.append(NULL)


@handler("LOAD_METHOD")
def _load_method(vm, f, ins):
    obj = f.stack.pop()
    f.stack.append(NULL)
    f.stack.append(vm.getattr(obj, ins.argval))


@handler("STORE_ATTR")
def _store_attr(vm, f, ins):
    obj = f.stack.pop()
    v = f.stack.pop()
    vm.setattr(obj, ins.argval, v)


@handler("DELETE_ATTR")
def _del_attr(vm, f, ins):
    vm.delattr(f.stack.pop(), ins.argval)


# ---- operators


def _binop_handler(sym):
    def h(vm, f, ins):
        b = f.stack.pop()
        a = f.stack.pop()
        f.stack.append(vm.binop(sym, a, b))
    return h


for _k, _v in OLD_BINOPS.items():
    HANDLERS["BINARY_" + _k] = _binop_handler(_v)
    HANDLERS["INPLACE_" + _k] = _binop_handler(_v + "=")


@handler("BINARY_OP")
def _binary_op(vm, f, ins):
    b = f.stack.pop()
    a = f.stack.pop()
    if ins.arg == 26:  # 3.14 NB_SUBSCR
        f.stack.append(vm.subscr(a, b))
        return
    f.stack.append(vm.binop(NB_OPS[ins.arg], a, b))


@handler("BINARY_SUBSCR")
def _binary_subscr(vm, f, ins):
    k = f.stack.pop()
    o = f.stack.pop()
    f.stack.append(vm.subscr(o, k))


@handler("BINARY_SLICE")
def _binary_slice(vm, f, ins):
    stop = f.stack.pop()
    start = f.stack.pop()
    o = f.stack.pop()
    f.stack.append(vm.subscr(o, slice(start, stop)))


@handler("STORE_SLICE")
def _store_slice(vm, f, ins):
    stop = f.stack.pop()
    start = f.stack.pop()
    o = f.stack.pop()
    v = f.stack.pop()
    vm.store_subscr(o, slice(start, stop), v)


@handler("STORE_SUBSCR")
def _store_subscr(vm, f, ins):
    k = f.stack.pop()
    o = f.stack.pop()
    v = f.stack.pop()
    vm.store_subscr(o, k, v)


@handler("DELETE_SUBSCR")
def _del_subscr(vm, f, ins):
    k = f.stack.pop()
    o = f.stack.pop()
    if isinstance(o, (dict, list, bytearray)):
        try:
            del o[k]
        except Exception as e:
            raise VMRaise(e)
    elif not isinstance(o, Unknown):
        raise VMUnsupported("del on non-container")


@handler("BUILD_SLICE")
def _build_slice(vm, f, ins):
    n = ins.arg
    args = f.stack[-n:]
    del f.stack[-n:]
    f.stack.append(slice(*args))


def _unary(fn):
    def h(vm, f, ins):
        v = f.stack.pop()
        if isinstance(v, Unknown):
            f.stack.append(Unknown("unary on unknown"))
            return
        if isinstance(v, VMObject):
            raise VMUnsupported("unary op on emulated object")
        try:
            f.stack.append(fn(v))
        except Exception as e:
            raise VMRaise(e)
    return h


HANDLERS["UNARY_POSITIVE"] = _unary(operator.pos)
HANDLERS["UNARY_NEGATIVE"] = _unary(operator.neg)
HANDLERS["UNARY_INVERT"] = _unary(operator.invert)


@handler("UNARY_NOT")
def _unary_not(vm, f, ins):
    v = f.stack.pop()
    f.stack.append(not vm.truth(v) if not isinstance(v, Unknown) else Unknown("not unknown"))


@handler("TO_BOOL")
def _to_bool(vm, f, ins):
    v = f.stack.pop()
    f.stack.append(vm.truth(v) if not isinstance(v, Unknown) else v)


@handler("COMPARE_OP")
def _compare(vm, f, ins):
    b = f.stack.pop()
    a = f.stack.pop()
    op = ins.argval
    if isinstance(op, str) and op.startswith("bool("):
        op = op[5:-1]
    if op in ("not-in", "is-not"):  # xdis spelling for <= 3.8
        op = op.replace("-", " ")
    if not isinstance(op, str):
        raise VMUnsupported(f"compare arg {op!r}")
    f.stack.append(vm.compare(op, a, b))


@handler("IS_OP")
def _is_op(vm, f, ins):
    b = f.stack.pop()
    a = f.stack.pop()
    f.stack.append((a is not b) if ins.arg else (a is b))


@handler("CONTAINS_OP")
def _contains_op(vm, f, ins):
    b = f.stack.pop()
    a = f.stack.pop()
    r = vm.contains(b, a)
    f.stack.append(r if isinstance(r, Unknown) else ((not r) if ins.arg else r))


# ---- containers


def _popn(f, n):
    if n == 0:
        return []
    items = f.stack[-n:]
    del f.stack[-n:]
    return items


@handler("BUILD_TUPLE")
def _build_tuple(vm, f, ins):
    f.stack.append(tuple(_popn(f, ins.arg)))


@handler("BUILD_LIST")
def _build_list(vm, f, ins):
    f.stack.append(_popn(f, ins.arg))


@handler("BUILD_SET")
def _build_set(vm, f, ins):
    items = _popn(f, ins.arg)
    try:
        f.stack.append(set(items))
    except VMUnknownError:
        f.stack.append(Unknown("set of unknown"))


@handler("BUILD_MAP")
def _build_map(vm, f, ins):
    items = _popn(f, 2 * ins.arg)
    try:
        f.stack.append({items[i]: items[i + 1] for i in range(0, len(items), 2)})
    except VMUnknownError:
        f.stack.append(Unknown("dict with unknown key"))


@handler("BUILD_CONST_KEY_MAP")
def _build_ckm(vm, f, ins):
    keys = f.stack.pop()
    vals = _popn(f, ins.arg)
    f.stack.append(dict(zip(keys, vals)))


@handler("BUILD_STRING")
def _build_string(vm, f, ins):
    parts = _popn(f, ins.arg)
    if any(isinstance(p, Unknown) for p in parts):
        f.stack.append(Unknown("string with unknown part"))
    else:
        f.stack.append("".join(parts))


@handler("LIST_APPEND")
def _list_append(vm, f, ins):
    v = f.stack.pop()
    f.stack[-ins.arg].append(v)


@handler("SET_ADD")
def _set_add(vm, f, ins):
    v = f.stack.pop()
    f.stack[-ins.arg].add(v)


@handler("MAP_ADD")
def _map_add(vm, f, ins):
    if f.version >= (3, 8):
        v = f.stack.pop()
        k = f.stack.pop()
    else:
        k = f.stack.pop()
        v = f.stack.pop()
    f.stack[-ins.arg][k] = v


def _materialize(vm, seq):
    it = vm.iter_(seq)
    out = []
    while True:
        try:
            out.append(vm.next_(it))
        except StopIteration:
            return out
        if len(out) > SIZE_LIMIT:
            raise VMStop("sequence too long")


@handler("LIST_EXTEND")
def _list_extend(vm, f, ins):
    seq = f.stack.pop()
    f.stack[-ins.arg].extend(_materialize(vm, seq))


@handler("SET_UPDATE")
def _set_update(vm, f, ins):
    seq = f.stack.pop()
    f.stack[-ins.arg].update(_materialize(vm, seq))


@handler("DICT_UPDATE", "DICT_MERGE")
def _dict_update(vm, f, ins):
    d = f.stack.pop()
    if isinstance(d, Unknown):
        raise VMUnknownError("**unknown")
    if isinstance(d, VMObject):
        raise VMUnsupported("** of emulated object")
    f.stack[-ins.arg].update(d)


@handler("LIST_TO_TUPLE")
def _list_to_tuple(vm, f, ins):
    f.stack.append(tuple(f.stack.pop()))


@handler("BUILD_LIST_UNPACK", "BUILD_TUPLE_UNPACK", "BUILD_TUPLE_UNPACK_WITH_CALL", "BUILD_SET_UNPACK")
def _build_unpack(vm, f, ins):
    seqs = _popn(f, ins.arg)
    out = []
    for s in seqs:
        out.extend(_materialize(vm, s))
    if ins.opname == "BUILD_LIST_UNPACK":
        f.stack.append(out)
    elif ins.opname == "BUILD_SET_UNPACK":
        f.stack.append(set(out))
    else:
        f.stack.append(tuple(out))


@handler("BUILD_MAP_UNPACK", "BUILD_MAP_UNPACK_WITH_CALL")
def _build_map_unpack(vm, f, ins):
    out = {}
    for d in _popn(f, ins.arg):
        if not isinstance(d, dict):
            raise VMUnsupported("** of non-dict")
        out.update(d)
    f.stack.append(out)


@handler("UNPACK_SEQUENCE")
def _unpack_seq(vm, f, ins):
    seq = f.stack.pop()
    items = _materialize(vm, seq)
    if len(items) != ins.arg:
        raise VMRaise(ValueError(f"expected {ins.arg} values to unpack, got {len(items)}"))
    f.stack.extend(reversed(items))


@handler("UNPACK_EX")
def _unpack_ex(vm, f, ins):
    before, after = ins.arg & 0xFF, ins.arg >> 8
    items = _materialize(vm, f.stack.pop())
    if len(items) < before + after:
        raise VMRaise(ValueError("not enough values to unpack"))
    mid = items[before:len(items) - after]
    res = items[:before] + [mid] + items[len(items) - after:]
    f.stack.extend(reversed(res))


# ---- formatting


def _fmt(vm, v, conv, spec=""):
    if isinstance(v, Unknown) or isinstance(spec, Unknown):
        return Unknown("format of unknown")
    try:
        if conv == 1:
            v = str(v)
        elif conv == 2:
            v = repr(v)
        elif conv == 3:
            v = ascii(v)
        return format(v, spec)
    except VMUnknownError:
        return Unknown("format of unknown")
    except VMSignal:
        raise
    except Exception as e:
        raise VMRaise(e)


@handler("FORMAT_VALUE")
def _format_value(vm, f, ins):
    spec = f.stack.pop() if ins.arg & 4 else ""
    v = f.stack.pop()
    f.stack.append(_fmt(vm, v, ins.arg & 3, spec))


@handler("CONVERT_VALUE")
def _convert_value(vm, f, ins):
    f.stack.append(_fmt(vm, f.stack.pop(), ins.arg))


@handler("FORMAT_SIMPLE")
def _format_simple(vm, f, ins):
    f.stack.append(_fmt(vm, f.stack.pop(), 0))


@handler("FORMAT_WITH_SPEC")
def _format_with_spec(vm, f, ins):
    spec = f.stack.pop()
    f.stack.append(_fmt(vm, f.stack.pop(), 0, spec))


@handler("CALL_INTRINSIC_1")
def _intrinsic1(vm, f, ins):
    v = f.stack.pop()
    n = ins.arg
    if n == 1:
        vm.event(f"print_expr {describe(v)}")
        f.stack.append(None)
    elif n == 2:
        vm.import_star(v, f.locals if f.locals is not None else f.globals)
        f.stack.append(None)
    elif n == 3:  # INTRINSIC_STOPITERATION_ERROR (generator exception handler)
        if isinstance(v, StopIteration):
            v = RuntimeError("generator raised StopIteration")
        f.stack.append(v)
    elif n == 5:
        f.stack.append(_pos(v))
    elif n == 6:
        f.stack.append(tuple(v))
    else:
        raise VMUnsupported(f"intrinsic {n}")


def _pos(v):
    if isinstance(v, Unknown):
        return Unknown("unary on unknown")
    try:
        return +v
    except Exception as e:
        raise VMRaise(e)


@handler("PRINT_EXPR")
def _print_expr(vm, f, ins):
    f.stack.pop()


@handler("SETUP_ANNOTATIONS")
def _setup_ann(vm, f, ins):
    ns = f.locals if f.locals is not None else f.globals
    ns.setdefault("__annotations__", {})


@handler("STORE_ANNOTATION")
def _store_ann(vm, f, ins):
    ns = f.locals if f.locals is not None else f.globals
    ns.setdefault("__annotations__", {})[ins.argval] = f.stack.pop()


# ---- iteration


@handler("GET_ITER", "GET_YIELD_FROM_ITER")
def _get_iter(vm, f, ins):
    v = f.stack.pop()
    if isinstance(v, Unknown):
        vm.unknown_branch(f, ins, None)
        f.stack.append(iter(()))
    else:
        f.stack.append(vm.iter_(v))
    if ins.opname == "GET_ITER" and f.version >= (3, 15):
        f.stack.append(NULL)  # index slot used by CPython's virtual iterators


@handler("POP_ITER")
def _pop_iter(vm, f, ins):
    f.stack.pop()
    if f.version >= (3, 15):
        f.stack.pop()


# LOAD_COMMON_CONSTANT tables (CPython's dis._common_constants)
_COMMON_CONSTANTS = {
    (3, 14): [AssertionError, NotImplementedError, tuple, all, any],
    (3, 15): [AssertionError, NotImplementedError, tuple, all, any, list, set, None, "", True, False, -1],
}


@handler("LOAD_COMMON_CONSTANT")
def _load_common_constant(vm, f, ins):
    table = _COMMON_CONSTANTS.get(f.version[:2]) or _COMMON_CONSTANTS[max(_COMMON_CONSTANTS)]
    if not 0 <= ins.arg < len(table):
        raise VMUnsupported(f"LOAD_COMMON_CONSTANT {ins.arg} ({bc.vstr(f.version)})")
    f.stack.append(table[ins.arg])


@handler("END_FOR")
def _end_for(vm, f, ins):
    f.stack.pop()
    if f.version < (3, 13):
        f.stack.pop()


# ---- jumps


@handler("JUMP_FORWARD", "JUMP_ABSOLUTE", "JUMP_BACKWARD", "JUMP_BACKWARD_NO_INTERRUPT", "JUMP", "JUMP_NO_INTERRUPT",
         "CONTINUE_LOOP")
def _jump(vm, f, ins):
    return ins.argval


def _cond_jump(jump_if):
    def h(vm, f, ins):
        v = f.stack.pop()
        try:
            t = vm.truth(v)
        except VMUnknownError:
            # enter the "if" body (fall through for POP_JUMP_IF_FALSE)
            return vm.unknown_branch(f, ins, None if not jump_if else None)
        if t == jump_if:
            return ins.argval
    return h


for _p in ("POP_JUMP_IF_", "POP_JUMP_FORWARD_IF_", "POP_JUMP_BACKWARD_IF_"):
    HANDLERS[_p + "TRUE"] = _cond_jump(True)
    HANDLERS[_p + "FALSE"] = _cond_jump(False)


def _none_jump(want_none):
    def h(vm, f, ins):
        v = f.stack.pop()
        if isinstance(v, Unknown):
            return vm.unknown_branch(f, ins, None)
        if (v is None) == want_none:
            return ins.argval
    return h


for _p in ("POP_JUMP_IF_", "POP_JUMP_FORWARD_IF_", "POP_JUMP_BACKWARD_IF_"):
    HANDLERS[_p + "NONE"] = _none_jump(True)
    HANDLERS[_p + "NOT_NONE"] = _none_jump(False)


def _jump_or_pop(jump_if):
    def h(vm, f, ins):
        v = f.stack[-1]
        try:
            t = vm.truth(v)
        except VMUnknownError:
            f.stack.pop()
            return vm.unknown_branch(f, ins, None)
        if t == jump_if:
            return ins.argval
        f.stack.pop()
    return h


HANDLERS["JUMP_IF_TRUE_OR_POP"] = _jump_or_pop(True)
HANDLERS["JUMP_IF_FALSE_OR_POP"] = _jump_or_pop(False)


# ---- blocks / exceptions


@handler("SETUP_LOOP")
def _setup_loop(vm, f, ins):
    f.blocks.append(Block("loop", ins.argval, len(f.stack)))


@handler("SETUP_EXCEPT", "SETUP_FINALLY")
def _setup_except(vm, f, ins):
    f.blocks.append(Block("except", ins.argval, len(f.stack)))


@handler("POP_BLOCK")
def _pop_block(vm, f, ins):
    if f.blocks:
        f.blocks.pop()


@handler("BREAK_LOOP")
def _break_loop(vm, f, ins):
    while f.blocks:
        b = f.blocks.pop()
        if b.kind == "loop":
            del f.stack[b.level:]
            return b.handler
    raise VMUnsupported("break outside loop")


@handler("POP_EXCEPT")
def _pop_except(vm, f, ins):
    if f.version >= (3, 11):
        f.stack.pop()
        if f.handled:
            f.handled.pop()
        return
    while f.blocks:
        b = f.blocks.pop()
        if b.kind == "handler":
            del f.stack[b.level:]
            break
    if f.handled:
        f.handled.pop()


@handler("PUSH_EXC_INFO")
def _push_exc_info(vm, f, ins):
    exc = f.stack.pop()
    f.stack.append(f.handled[-1] if f.handled else None)
    f.stack.append(exc)
    f.handled.append(exc)


@handler("CHECK_EXC_MATCH")
def _check_exc_match(vm, f, ins):
    typ = f.stack.pop()
    f.stack.append(vm.exc_match(f.stack[-1], typ))


@handler("JUMP_IF_NOT_EXC_MATCH")
def _jump_not_exc_match(vm, f, ins):
    typ = f.stack.pop()
    exc = f.stack.pop()
    if not vm.exc_match(exc, typ):
        return ins.argval


@handler("RERAISE")
def _reraise(vm, f, ins):
    if f.version >= (3, 11):
        exc = f.stack.pop()
        if ins.arg:
            f.stack.pop()
    else:
        f.stack.pop()
        exc = f.stack.pop()
        f.stack.pop()
    raise VMRaise(exc)


@handler("END_FINALLY")
def _end_finally(vm, f, ins):
    top = f.stack.pop()
    if top is None or top is NULL:
        return
    if isinstance(top, type) and issubclass(top, BaseException):
        val = f.stack.pop()
        f.stack.pop()
        raise VMRaise(val)
    raise VMUnsupported("END_FINALLY with pending return/break")


@handler("BEGIN_FINALLY")
def _begin_finally(vm, f, ins):
    f.stack.append(NULL)


@handler("RAISE_VARARGS")
def _raise(vm, f, ins):
    if ins.arg == 0:
        if f.handled:
            raise VMRaise(f.handled[-1])
        raise VMRaise(RuntimeError("No active exception to reraise"))
    if ins.arg == 2:
        f.stack.pop()
    exc = f.stack.pop()
    if isinstance(exc, type) and issubclass(exc, BaseException):
        exc = exc()
    if isinstance(exc, Unknown):
        raise VMUnknownError("raise of unknown")
    if not isinstance(exc, BaseException):
        raise VMUnsupported("raising non-host exception")
    raise VMRaise(exc)


# ---- calls


def _do_call(vm, f, fn, args, kwnames=()):
    kwargs = {}
    if kwnames:
        n = len(kwnames)
        kwargs = dict(zip(kwnames, args[-n:]))
        args = args[:-n]
    return vm.call(fn, tuple(args), kwargs)


@handler("CALL_FUNCTION")
def _call_function(vm, f, ins):
    args = _popn(f, ins.arg)
    fn = f.stack.pop()
    f.stack.append(_do_call(vm, f, fn, args))


@handler("CALL_FUNCTION_KW")
def _call_function_kw(vm, f, ins):
    kwnames = f.stack.pop()
    args = _popn(f, ins.arg)
    fn = f.stack.pop()
    f.stack.append(_do_call(vm, f, fn, args, kwnames))


@handler("CALL_METHOD")
def _call_method(vm, f, ins):
    args = _popn(f, ins.arg)
    y = f.stack.pop()
    x = f.stack.pop()
    if x is NULL:
        f.stack.append(_do_call(vm, f, y, args))
    else:
        f.stack.append(_do_call(vm, f, x, [y] + args))


@handler("KW_NAMES")
def _kw_names(vm, f, ins):
    f.kwnames = f.code.co_consts[ins.arg]


@handler("CALL")
def _call(vm, f, ins):
    args = _popn(f, ins.arg)
    kwnames, f.kwnames = f.kwnames, ()
    if f.version >= (3, 13):
        self_or_null = f.stack.pop()
        fn = f.stack.pop()
        if self_or_null is not NULL:
            args = [self_or_null] + args
    else:
        y = f.stack.pop()
        x = f.stack.pop()
        if x is NULL:
            fn = y
        else:
            fn, args = x, [y] + args
    f.stack.append(_do_call(vm, f, fn, args, kwnames))


@handler("CALL_KW")
def _call_kw(vm, f, ins):
    kwnames = f.stack.pop()
    args = _popn(f, ins.arg)
    self_or_null = f.stack.pop()
    fn = f.stack.pop()
    if self_or_null is not NULL:
        args = [self_or_null] + args
    f.stack.append(_do_call(vm, f, fn, args, kwnames))


@handler("CALL_FUNCTION_EX")
def _call_ex(vm, f, ins):
    kwargs = f.stack.pop() if (ins.arg & 1) else {}
    args = f.stack.pop()
    if f.version >= (3, 13):
        f.stack.pop()  # NULL
        fn = f.stack.pop()
    elif f.version >= (3, 11):
        fn = f.stack.pop()
        f.stack.pop()  # NULL
    else:
        fn = f.stack.pop()
    if isinstance(args, Unknown) or isinstance(kwargs, Unknown):
        f.stack.append(Unknown("call with unknown *args"))
        return
    args = _materialize(vm, args)
    if not isinstance(kwargs, dict):
        raise VMUnsupported("** of non-dict")
    f.stack.append(vm.call(fn, tuple(args), dict(kwargs)))


@handler("MAKE_FUNCTION")
def _make_function(vm, f, ins):
    if f.version < (3, 11):
        qualname = f.stack.pop()
        code = f.stack.pop()
    else:
        code = f.stack.pop()
        qualname = getattr(code, "co_qualname", None) or code.co_name
    flags = ins.arg or 0
    closure = f.stack.pop() if flags & 8 else None
    annotations = f.stack.pop() if flags & 4 else None
    kwdefaults = f.stack.pop() if flags & 2 else None
    defaults = f.stack.pop() if flags & 1 else None
    if isinstance(code, CodeRef):
        ver, code = code.blob.version, code.code
    else:
        ver = f.version
    fn = VMFunction(vm, code, ver, f.globals, name=code.co_name, qualname=str(qualname), defaults=defaults,
                    kwdefaults=kwdefaults, closure=closure, annotations=annotations)
    f.stack.append(fn)


@handler("SET_FUNCTION_ATTRIBUTE")
def _set_fn_attr(vm, f, ins):
    fn = f.stack.pop()
    attr = f.stack.pop()
    flag = ins.arg
    if flag == 1:
        fn.defaults = tuple(attr)
    elif flag == 2:
        fn.kwdefaults = dict(attr)
    elif flag == 4:
        fn.annotations = attr
    elif flag == 8:
        fn.closure = tuple(attr)
    f.stack.append(fn)


@handler("LOAD_BUILD_CLASS")
def _load_build_class(vm, f, ins):
    f.stack.append(vm.builtins["__build_class__"])


# ---- imports


@handler("IMPORT_NAME")
def _import_name(vm, f, ins):
    fromlist = f.stack.pop()
    level = f.stack.pop()
    f.stack.append(vm.import_(ins.argval, fromlist, level or 0))


@handler("IMPORT_FROM")
def _import_from(vm, f, ins):
    mod = f.stack[-1]
    if isinstance(mod, SafeModule):
        sub = safelib.get_module(vm, f"{mod.__name__}.{ins.argval}")
        f.stack.append(sub if sub is not None else mod.get(ins.argval))
    else:
        f.stack.append(vm.getattr(mod, ins.argval))


@handler("IMPORT_STAR")
def _import_star(vm, f, ins):
    vm.import_star(f.stack.pop(), f.locals if f.locals is not None else f.globals)

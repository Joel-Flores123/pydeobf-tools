"""Restricted static unpickler for pickle / dill / cloudpickle payloads.

Packers hide code as ``exec(dill.loads(...))``: the pickle builds a code object
with ``dill._dill._create_code``.  Real unpickling calls arbitrary functions, so
the opcodes are interpreted here instead and the only callables reachable are
the allow-listed constructors below (code objects go through the emulator's
``types.CodeType``).  Any other global raises VMUnsupported; nothing runs.
"""
import codecs
import pickletools

from . import vm as _vm

_MAX_OPS = 2_000_000
_MARK = object()


def _code(vm):
    def create_code(*args):
        # dill < 0.3.6 pickles the CodeType arguments as-is; newer dill puts co_lnotab/co_linetable first
        if args and not isinstance(args[0], int):
            args = args[1:]
        return vm.make_code_type(*args)
    return create_code


def _globals(vm):
    safe_types = {"bytes": bytes, "bytearray": bytearray, "set": set, "frozenset": frozenset, "complex": complex,
                  "list": list, "tuple": tuple, "dict": dict, "str": str, "int": int, "float": float}
    load_types = {"CodeType": vm.make_code_type, **safe_types}
    G = {("dill._dill", "_create_code"): _code(vm),
         ("dill.dill", "_create_code"): _code(vm),
         ("dill._dill", "_load_type"): lambda n: load_types[n] if n in load_types else _refuse(f"_load_type({n!r})"),
         ("dill.dill", "_load_type"): lambda n: load_types[n] if n in load_types else _refuse(f"_load_type({n!r})"),
         ("_codecs", "encode"): lambda s, enc="utf-8", *a: codecs.encode(s, enc),  # protocol <= 2 bytes
         ("types", "CodeType"): vm.make_code_type}
    for mod in ("builtins", "__builtin__"):
        G.update({(mod, n): t for n, t in safe_types.items()})
    return G


def _refuse(what):
    raise _vm.VMUnsupported(f"unpickling {what}")


def loads(vm, data, via="pickle"):
    if isinstance(data, _vm.Unknown):
        return _vm.Unknown(f"{via}.loads of an unknown value")
    if isinstance(data, (bytearray, memoryview)):
        data = bytes(data)
    if not isinstance(data, bytes):
        raise _vm.VMRaise(TypeError(f"a bytes-like object is required, not '{type(data).__name__}'"))
    G = _globals(vm)
    stack, memo, marks = [], {}, []

    def pop_mark():
        items = stack[marks[-1] + 1:] if marks else []
        k = marks.pop()
        del stack[k:]
        return items

    try:
        ops = pickletools.genops(data)
        for n, (op, arg, _pos) in enumerate(ops):
            if n > _MAX_OPS:
                raise _vm.VMStop("refusing huge pickle")
            name = op.name
            if name in ("PROTO", "FRAME"):
                continue
            if name == "STOP":
                vm.event(f"{via}.loads -> {_vm.describe(stack[-1] if stack else None)} (static unpickle)")
                return stack.pop()
            if name in ("NONE", "NEWTRUE", "NEWFALSE"):
                stack.append({"NONE": None, "NEWTRUE": True, "NEWFALSE": False}[name])
            elif name in ("INT", "BININT", "BININT1", "BININT2", "LONG", "LONG1", "LONG4", "FLOAT", "BINFLOAT",
                          "BINUNICODE", "SHORT_BINUNICODE", "BINUNICODE8", "UNICODE",
                          "BINBYTES", "SHORT_BINBYTES", "BINBYTES8"):
                stack.append(arg)
            elif name in ("STRING", "BINSTRING", "SHORT_BINSTRING"):  # py2 str: latin-1, as pickle's default
                stack.append(arg.decode("latin-1") if isinstance(arg, bytes) else arg)
            elif name == "BYTEARRAY8":
                stack.append(bytearray(arg))
            elif name == "MARK":
                marks.append(len(stack))
                stack.append(_MARK)
            elif name == "POP":
                stack.pop()
            elif name == "POP_MARK":
                pop_mark()
            elif name == "DUP":
                stack.append(stack[-1])
            elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
                memo[arg] = stack[-1]
            elif name == "MEMOIZE":
                memo[len(memo)] = stack[-1]
            elif name in ("GET", "BINGET", "LONG_BINGET"):
                stack.append(memo[arg])
            elif name == "EMPTY_TUPLE":
                stack.append(())
            elif name in ("TUPLE1", "TUPLE2", "TUPLE3"):
                k = int(name[-1])
                t = tuple(stack[-k:])
                del stack[-k:]
                stack.append(t)
            elif name == "TUPLE":
                stack.append(tuple(pop_mark()))
            elif name == "EMPTY_LIST":
                stack.append([])
            elif name == "LIST":
                stack.append(list(pop_mark()))
            elif name == "APPEND":
                v = stack.pop()
                stack[-1].append(v)
            elif name == "APPENDS":
                items = pop_mark()
                stack[-1].extend(items)
            elif name == "EMPTY_DICT":
                stack.append({})
            elif name == "DICT":
                items = pop_mark()
                stack.append(dict(zip(items[::2], items[1::2])))
            elif name == "SETITEM":
                v, k = stack.pop(), stack.pop()
                stack[-1][k] = v
            elif name == "SETITEMS":
                items = pop_mark()
                stack[-1].update(zip(items[::2], items[1::2]))
            elif name == "EMPTY_SET":
                stack.append(set())
            elif name == "ADDITEMS":
                items = pop_mark()
                stack[-1].update(items)
            elif name == "FROZENSET":
                stack.append(frozenset(pop_mark()))
            elif name in ("GLOBAL", "STACK_GLOBAL"):
                if name == "GLOBAL":
                    mod, _, qual = arg.partition(" ")
                else:
                    qual, mod = stack.pop(), stack.pop()
                fn = G.get((mod, qual))
                if fn is None:
                    _refuse(f"global {mod}.{qual}")
                stack.append(fn)
            elif name == "REDUCE":
                args, fn = stack.pop(), stack.pop()
                if not isinstance(args, tuple):
                    _refuse("REDUCE with non-tuple arguments")
                stack.append(fn(*args))
            else:
                _refuse(f"opcode {name}")
    except (IndexError, KeyError, ValueError, TypeError) as e:
        raise _vm.VMRaise(type(e)(f"unpickling failed: {e}"))
    raise _vm.VMRaise(EOFError("Ran out of input"))

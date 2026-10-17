"""Source-level clean-up of recovered payloads.

* Strip known anti-analysis boilerplate (Specter's "anti skid" block).
* Inline pyminifier-style aliases (``ﺭ=print`` / ``ﻞ=os.path.join``).
* Rename non-Latin identifiers (pyminifier --nonlatin) to readable names.
* Fold encoded constant expressions (``''.join(chr(x) for x in [...])``,
  ``base64.b64decode('..').decode()`` ...).  Folding is done by the sandboxed
  emulator, never by ``eval``.
"""

from __future__ import annotations

import ast
import builtins
import re
import time

from . import bytecode as bc
from . import safelib
from . import vm as V

SPECTER_ANTISKID = re.compile(
    r"\A\s*# GG! You just deobfuscated Specter.*?__import__\('sys'\)\.exit\(\)[ \t]*\n+", re.S)

PURE_NAMES = {
    "chr", "ord", "str", "bytes", "bytearray", "int", "float", "bool", "len", "list", "tuple", "sorted",
    "reversed", "map", "filter", "zip", "range", "sum", "min", "max", "abs", "hex", "oct", "bin", "repr",
    "round", "divmod", "pow", "all", "any", "enumerate", "format", "ascii", "slice", "set", "frozenset", "dict",
}
FOLD_MODULES = {"base64", "zlib", "bz2", "lzma", "gzip", "binascii", "codecs", "string", "math", "operator",
                "functools", "itertools", "struct", "hashlib"}


def strip_boilerplate(src: str) -> tuple[str, list[str]]:
    notes = []
    new = SPECTER_ANTISKID.sub("", src, count=1)
    if new != src:
        notes.append("removed Specter anti-skid preamble")
    return new, notes


# ------------------------------------------------------------------ helpers


def _stored_names(tree):
    stored = {}

    def add(n):
        stored[n] = stored.get(n, 0) + 1

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            add(node.name)
        elif isinstance(node, ast.arg):
            add(node.arg)
        elif isinstance(node, ast.alias):
            add((node.asname or node.name).split(".")[0])
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for n in node.names:
                add(n)
                add(n)  # treat as "assigned elsewhere"
        elif isinstance(node, ast.ExceptHandler) and node.name:
            add(node.name)
        elif isinstance(node, (ast.MatchAs, ast.MatchStar)) and getattr(node, "name", None):
            add(node.name)
    return stored


def _module_imports(tree):
    """Top-level imports of whitelisted modules: name -> (module, attr|None)."""
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FOLD_MODULES:
                    out[a.asname or a.name.split(".")[0]] = (a.asname and a.name or a.name.split(".")[0], None)
        elif isinstance(node, ast.ImportFrom) and node.module in FOLD_MODULES and not node.level:
            for a in node.names:
                if a.name != "*":
                    out[a.asname or a.name] = (node.module, a.name)
    return out


# ------------------------------------------------------------ alias inline


_SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef, ast.ListComp, ast.SetComp,
           ast.DictComp, ast.GeneratorExp)


def _scope_bindings(scope):
    """Names bound locally in ``scope`` (not descending into nested scopes)."""
    bound, globs = set(), set()
    if isinstance(scope, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        a = scope.args
        for arg in a.posonlyargs + a.args + a.kwonlyargs + [x for x in (a.vararg, a.kwarg) if x]:
            bound.add(arg.arg)
        body = scope.body if isinstance(scope.body, list) else [scope.body]
    elif isinstance(scope, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
        for g in scope.generators:
            for n in ast.walk(g.target):
                if isinstance(n, ast.Name):
                    bound.add(n.id)
        return bound
    else:
        body = scope.body
    stack = list(body)
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(n.name)
            stack.extend(n.decorator_list)
            continue
        if isinstance(n, _SCOPES):
            continue
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            bound.add(n.id)
        elif isinstance(n, ast.alias):
            bound.add((n.asname or n.name).split(".")[0])
        elif isinstance(n, ast.ExceptHandler) and n.name:
            bound.add(n.name)
        elif isinstance(n, ast.Global):
            globs.update(n.names)
        stack.extend(ast.iter_child_nodes(n))
    return bound - globs


def inline_aliases(tree, notes):
    """Inline module-level ``X = builtin`` / ``X = module.attr`` aliases (pyminifier),
    respecting scopes so a local variable that reuses the name is left alone."""
    module_bound = _scope_bindings(tree)
    imported = {(a.asname or a.name).split(".")[0] for n in tree.body
                if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    declared_global = {name for n in ast.walk(tree) if isinstance(n, ast.Global) for name in n.names}
    counts = {}
    for node in tree.body:
        for n in ast.walk(node) if not isinstance(node, _SCOPES) else []:
            if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
                counts[n.id] = counts.get(n.id, 0) + 1
    aliases = {}
    for node in tree.body:
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)):
            continue
        tgt = node.targets[0].id
        if counts.get(tgt) != 1 or tgt in declared_global or tgt in imported:
            continue
        val = node.value
        if isinstance(val, ast.Name) and hasattr(builtins, val.id) and val.id not in module_bound:
            aliases[tgt] = val
        elif isinstance(val, ast.Attribute):
            root = val
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name) and root.id in imported and counts.get(root.id, 0) == 0:
                aliases[tgt] = val
    if not aliases:
        return tree

    class Inline(ast.NodeTransformer):
        def __init__(self):
            self.shadow = [set()]  # names bound by enclosing function-like scopes

        def _scoped(self, node, bound):
            self.shadow.append(self.shadow[-1] | bound)
            self.generic_visit(node)
            self.shadow.pop()
            return node

        def visit_FunctionDef(self, node):
            for d in node.decorator_list + [x for x in node.args.defaults + node.args.kw_defaults if x]:
                self.visit(d)
            self.shadow.append(self.shadow[-1] | _scope_bindings(node))
            node.body = [self.visit(s) for s in node.body]
            self.shadow.pop()
            return node

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_Lambda(self, node):
            return self._scoped(node, _scope_bindings(node))

        def visit_ClassDef(self, node):
            # class-body bindings don't leak into methods, but do shadow in the body
            return self._scoped(node, set())

        def _comp(self, node):
            return self._scoped(node, _scope_bindings(node))

        visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comp

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in aliases and node.id not in self.shadow[-1]:
                return ast.copy_location(_clone(aliases[node.id]), node)
            return node

    tree.body = [n for n in tree.body if not (isinstance(n, ast.Assign) and len(n.targets) == 1 and
                                              isinstance(n.targets[0], ast.Name) and n.targets[0].id in aliases)]
    Inline().visit(tree)
    notes.append(f"inlined {len(aliases)} builtin/import aliases")
    return tree


def _clone(node):
    return ast.parse(ast.unparse(node), mode="eval").body


# ---------------------------------------------------------- non-latin names


def rename_nonlatin(tree, notes):
    mapping = {}

    def new(name):
        if name.isascii():
            return name
        if name not in mapping:
            mapping[name] = f"_obf{len(mapping)}"
        return mapping[name]

    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            node.id = new(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.name = new(node.name)
        elif isinstance(node, ast.arg):
            node.arg = new(node.arg)
        elif isinstance(node, ast.Attribute):
            node.attr = new(node.attr)
        elif isinstance(node, ast.keyword) and node.arg:
            node.arg = new(node.arg)
        elif isinstance(node, ast.alias) and node.asname:
            node.asname = new(node.asname)
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            node.names = [new(n) for n in node.names]
        elif isinstance(node, ast.ExceptHandler) and node.name:
            node.name = new(node.name)
    if mapping:
        notes.append(f"renamed {len(mapping)} non-Latin identifiers to _obfN")
    return tree


# ------------------------------------------------------------ const folding


class _Folder(ast.NodeTransformer):
    def __init__(self, allowed, env_factory, notes):
        self.allowed = allowed
        self.env_factory = env_factory
        self.count = 0
        self.budget = time.monotonic() + 30

    def _free_names(self, node):
        loads, bound = set(), set()
        for n in ast.walk(node):
            if isinstance(n, ast.Name):
                (loads if isinstance(n.ctx, ast.Load) else bound).add(n.id)
            elif isinstance(n, ast.arg):
                bound.add(n.arg)
        return loads - bound

    def _foldable(self, node):
        if isinstance(node, (ast.Constant, ast.Name, ast.Starred, ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom,
                             ast.NamedExpr)):
            return False
        if getattr(node, "ctx", None) is not None and not isinstance(node.ctx, ast.Load):
            return False
        if not any(isinstance(n, (ast.Call, ast.Subscript, ast.BinOp, ast.JoinedStr)) for n in ast.walk(node)):
            return False
        if any(isinstance(n, (ast.Await, ast.Yield, ast.YieldFrom, ast.NamedExpr)) for n in ast.walk(node)):
            return False
        if not self._free_names(node) <= self.allowed:
            return False
        # plain arithmetic between literals is left alone
        if not any(isinstance(n, (ast.Call, ast.Subscript)) for n in ast.walk(node)) and \
                all(isinstance(n.value, (int, float)) for n in ast.walk(node) if isinstance(n, ast.Constant)):
            return False
        return True

    def _evaluate(self, node):
        expr = ast.Expression(body=node)
        ast.fix_missing_locations(expr)
        try:
            code = compile(expr, "<fold>", "eval")  # compile only; emulator evaluates
        except (SyntaxError, ValueError, TypeError):
            return V.EMPTY
        emu, g = self.env_factory()
        emu.deadline = time.monotonic() + 3
        try:
            val = emu.run_code(code, bc.HOST_VERSION, g, None, module=True)
        except (V.VMSignal, RecursionError, MemoryError):
            return V.EMPTY
        if emu.payloads or any(e.startswith(("call ", "import ", "open(", "exec")) for e in emu.events):
            return V.EMPTY
        return val

    def _acceptable(self, val, node):
        # Only fold expressions that *decode* text/bytes; ordinary computations
        # (list comprehensions over range(), arithmetic) are left untouched.
        if isinstance(val, (tuple, list)):
            if not (0 < len(val) <= 64 and all(isinstance(x, (str, bytes)) for x in val)):
                return False
        elif not isinstance(val, (str, bytes)):
            return False
        try:
            seg = len(ast.unparse(node))
        except Exception:
            return False
        return len(repr(val)) <= seg * 1.5 + 40

    def generic_visit(self, node):
        return super().generic_visit(node)

    def visit(self, node):
        if isinstance(node, ast.expr) and time.monotonic() < self.budget and self._foldable(node):
            val = self._evaluate(node)
            if val is not V.EMPTY and self._acceptable(val, node):
                self.count += 1
                new = ast.parse(repr(val), mode="eval").body
                return ast.copy_location(new, node)
        return super().visit(node)


def fold_constants(tree, notes):
    stored = _stored_names(tree)
    imports = _module_imports(tree)
    # eval/__import__ are safe here: folding runs in the emulator, which
    # emulates eval'd code and only exposes allow-listed modules.
    allowed = {n for n in PURE_NAMES | {"eval", "__import__"} if stored.get(n, 0) == 0} | {
        n for n in imports if stored.get(n, 0) == 1}

    def env_factory():
        emu = V.Emulator(max_steps=300_000, max_seconds=3)
        g = emu.new_globals()
        for name, (mod, attr) in imports.items():
            m = safelib.get_module(emu, mod)
            if m is None:
                continue
            g[name] = m if attr is None else m.get(attr)
        return emu, g

    folder = _Folder(allowed, env_factory, notes)
    tree = folder.visit(tree)
    if folder.count:
        notes.append(f"folded {folder.count} constant expressions")
    return tree


# ------------------------------------------------- globals propagation


def _value_to_ast(v, tree_imports, depth=0):
    """AST for a value the emulator computed, or None if it has no literal form."""
    if isinstance(v, bool) or v is None or isinstance(v, (int, float, complex, str, bytes)):
        r = repr(v)
        return ast.parse(r, mode="eval").body if len(r) <= 400 else None
    if isinstance(v, (tuple, list)) and depth == 0 and len(v) <= 32:
        parts = [_value_to_ast(x, tree_imports, 1) for x in v]
        if all(p is not None for p in parts):
            return (ast.Tuple if isinstance(v, tuple) else ast.List)(elts=parts, ctx=ast.Load())
        return None
    if isinstance(v, V.Special):
        return ast.Name(v.__name__, ast.Load()) if getattr(builtins, v.__name__, None) is not None else None
    if isinstance(v, (type, types_builtin)) and getattr(builtins, getattr(v, "__name__", ""), None) is v:
        return ast.Name(v.__name__, ast.Load())
    if isinstance(v, types_builtin) and isinstance(getattr(v, "__self__", None), (str, bytes)) and len(v.__self__) < 40:
        return ast.Attribute(ast.Constant(v.__self__), v.__name__, ast.Load())
    path = _module_function_path(v)
    if path:
        mod, attr = path
        base = ast.Name(mod, ast.Load()) if mod in tree_imports else \
            ast.Call(ast.Name("__import__", ast.Load()), [ast.Constant(mod)], [])
        return ast.Attribute(base, attr, ast.Load())
    if isinstance(v, V.SafeModule) or (isinstance(v, V.Opaque) and _DOTTED.fullmatch(v.reason)):
        path = v.__name__ if isinstance(v, V.SafeModule) else v.reason
        root = path.split(".")[0]
        if root in tree_imports or root == "builtins":
            node = ast.Name(root, ast.Load())
            for part in path.split(".")[1:]:
                node = ast.Attribute(node, part, ast.Load())
            return node
    return None


_MODFUNC_CACHE = {}


def _module_function_path(v):
    """(module, attr) for a function exposed by one of the emulator's safe modules."""
    if not _MODFUNC_CACHE:
        emu = V.Emulator()
        for mod in FOLD_MODULES | {"os.path"}:
            m = safelib.get_module(emu, mod)
            if m is not None:
                for k, fn in m.attrs.items():
                    if callable(fn) and not isinstance(fn, type):
                        _MODFUNC_CACHE.setdefault(id(fn), (mod, k, fn))
    hit = _MODFUNC_CACHE.get(id(v))
    return (hit[0], hit[1]) if hit and hit[2] is v else None


_DOTTED = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
types_builtin = type(len)


def _is_junk_function(fn):
    """lambda x: globals() / locals() / vars() - Hyperion-style no-op filler."""
    if not isinstance(fn, V.VMFunction) or fn.code.co_name != "<lambda>":
        return False
    if any(bc.is_code(c) for c in fn.code.co_consts):
        return False
    for n in fn.code.co_names:
        target = fn.globals.get(n, fn.vm.builtins.get(n))
        if not (isinstance(target, V.Special) and target.__name__ in ("globals", "locals", "vars", "dir")):
            return False
    return bool(fn.code.co_names)


def _all_stored(tree):
    out = set()
    for scope in [tree] + [n for n in ast.walk(tree) if isinstance(n, _SCOPES)]:
        out |= _scope_bindings(scope)
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
    return out


def propagate_globals(tree, env, notes):
    """Replace names whose value the emulator knows (constants, builtins,
    modules) and drop calls to no-op filler lambdas."""
    stored = _all_stored(tree)
    imports = {(a.asname or a.name).split(".")[0] for n in ast.walk(tree)
               if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names}
    mapping, junk = {}, set()
    loads = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    for name in loads - stored:
        if name not in env or name.startswith("__"):
            continue
        v = env[name]
        if _is_junk_function(v):
            junk.add(name)
            continue
        node = _value_to_ast(v, imports)
        if node is not None and not (isinstance(node, ast.Name) and node.id == name):
            mapping[name] = node
    if not mapping and not junk:
        return tree

    class Drop(ast.NodeTransformer):
        def visit_Expr(self, node):
            c = node.value
            if isinstance(c, ast.Call) and isinstance(c.func, ast.Name) and c.func.id in junk:
                return None
            return self.generic_visit(node)

    tree = Drop().visit(tree)
    _substitute(tree, mapping)
    notes.append(f"propagated {len(mapping)} values from the emulated namespace"
                 + (f", removed calls to {len(junk)} filler lambdas" if junk else ""))
    return tree


def _substitute(tree, mapping):
    class Sub(ast.NodeTransformer):
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load) and node.id in mapping:
                return ast.copy_location(_clone(mapping[node.id]), node)
            return node
    Sub().visit(tree)


class _DeadCode(ast.NodeTransformer):
    """Resolve ``if <constant>:`` and drop no-op expression statements."""

    def __init__(self):
        self.count = 0

    def visit_If(self, node):
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            self.count += 1
            return (node.body if node.test.value else node.orelse) or None
        if isinstance(node.test, ast.Name) and not node.orelse and \
                (not node.body or all(isinstance(s, ast.Pass) for s in node.body)):
            self.count += 1  # `if name: pass` has no effect
            return None
        return node

    def visit_Expr(self, node):
        v = node.value
        if isinstance(v, ast.Constant) and not isinstance(v.value, str):
            self.count += 1
            return None
        if isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id in ("exec", "eval") \
                and len(v.args) == 1 and isinstance(v.args[0], ast.Constant) and not str(v.args[0].value).strip():
            self.count += 1
            return None
        return self.generic_visit(node)


def _refill_bodies(tree):
    for n in ast.walk(tree):
        if isinstance(getattr(n, "body", None), list) and not n.body and not isinstance(n, ast.Module):
            n.body = [ast.Pass()]
        if isinstance(n, ast.Try) and not n.handlers and not n.finalbody:
            n.finalbody = [ast.Pass()]


_TAMPER_NAMES = {"__author__", "__authors__", "__obfuscator__", "__github__", "__discord__", "__license__", "__code__"}


def remove_anti_tamper(tree, notes):
    """Drop ``try: if __author__ != ...: crash  except: input(); exit()`` blocks."""
    def is_tamper(n):
        if not isinstance(n, ast.Try) or len(n.body) != 1 or not isinstance(n.body[0], ast.If):
            return False
        names = {x.id for x in ast.walk(n.body[0].test) if isinstance(x, ast.Name)}
        return len(names & _TAMPER_NAMES) >= 2

    before = len(tree.body)
    tree.body = [n for n in tree.body if not is_tamper(n)]
    if len(tree.body) != before:
        notes.append("removed anti-tamper check")
    return tree


_NS_FUNCS = {"globals", "locals", "vars"}


def _ns_write(stmt):
    """``globals()['name'] = value`` -> 'name' (or None)."""
    if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        t = stmt.targets[0]
        if isinstance(t, ast.Subscript) and isinstance(t.value, ast.Call) and not t.value.args \
                and isinstance(t.value.func, ast.Name) and t.value.func.id in _NS_FUNCS \
                and isinstance(t.slice, ast.Constant) and isinstance(t.slice.value, str) \
                and t.slice.value.isidentifier():
            return t.slice.value
    return None


def simplify_namespace_writes(tree, notes):
    """Module-level ``globals()['x'] = v``: drop if 'x' is unused, else ``x = v``."""
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    dropped = renamed = 0
    body = []
    for stmt in tree.body:
        name = _ns_write(stmt)
        if name is None:
            body.append(stmt)
        elif name not in used:
            dropped += 1
        else:
            body.append(ast.copy_location(ast.Assign([ast.Name(name, ast.Store())], stmt.value), stmt))
            renamed += 1
    tree.body = body
    if dropped or renamed:
        notes.append(f"namespace writes: dropped {dropped} unused, rewrote {renamed} as assignments")
    return tree


class _PyArmorArtifacts(ast.NodeTransformer):
    """Remove residue left after PyArmor marker stripping: ``try: BODY finally:
    pass`` wrappers (semantically just BODY), ``if False:`` blocks, and calls to
    leftover ``__pyarmor_*`` names."""

    def __init__(self):
        self.count = 0

    def _clean(self, stmts, keep_pass=True):
        out = []
        for s in stmts:
            r = self.visit(s)
            if r is None:
                continue
            out.extend(r if isinstance(r, list) else [r])
        if not keep_pass:
            out = [s for s in out if not isinstance(s, ast.Pass)]
        return out

    def visit_Try(self, node):
        node.body = self._clean(node.body) or [ast.Pass()]
        node.orelse = self._clean(node.orelse, keep_pass=False)
        node.finalbody = self._clean(node.finalbody, keep_pass=False)
        node.handlers = [self.visit(h) for h in node.handlers]
        handlers_trivial = all(not self._clean(hd.body, keep_pass=False) for hd in node.handlers)
        if not node.finalbody and not node.orelse and (not node.handlers or handlers_trivial):
            self.count += 1
            return node.body  # try/finally/except reduced to just the body
        return node

    def visit_If(self, node):
        node.body = self._clean(node.body) or [ast.Pass()]
        node.orelse = self._clean(node.orelse)
        if isinstance(node.test, ast.Constant) and not node.test.value:
            self.count += 1
            return node.orelse or None
        return node

    def visit_Expr(self, node):
        c = node.value
        if isinstance(c, ast.Call):
            f = c.func
            name = f.id if isinstance(f, ast.Name) else (f.value.value if isinstance(f, ast.Constant) else None)
            if isinstance(name, str) and name.startswith("__pyarmor_"):
                self.count += 1
                return None
        return node


def remove_pyarmor_artifacts(tree, notes):
    t = _PyArmorArtifacts()
    tree = t.visit(tree)
    if t.count:
        notes.append(f"removed {t.count} PyArmor wrapper artifacts")
    return tree


def drop_redundant_pass(tree):
    """Remove ``pass`` from any block that also contains real statements."""
    for node in ast.walk(tree):
        for fieldname in ("body", "orelse", "finalbody"):
            block = getattr(node, fieldname, None)
            if isinstance(block, list) and len(block) > 1:
                pruned = [s for s in block if not isinstance(s, ast.Pass)]
                setattr(node, fieldname, pruned or [ast.Pass()])
    return tree


def remove_dead_code(tree, notes):
    dc = _DeadCode()
    tree = dc.visit(tree)
    _refill_bodies(tree)
    if dc.count:
        notes.append(f"removed {dc.count} dead/filler statements")
    return tree


# ---------------------------------------------------- confusable names

_CONFUSABLE_MIN = 12


def _is_confusable(name):
    core = name.strip("_")
    return len(core) >= _CONFUSABLE_MIN and len(set(core)) <= 3 and not name.startswith("__")


def rename_confusables(tree, notes):
    """Rename IlIlIl / O0O0O / SS2S2-style identifiers to readable names."""
    kinds = {}
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_confusable(n.name):
            kinds.setdefault(n.name, "func")
        elif isinstance(n, ast.ClassDef) and _is_confusable(n.name):
            kinds.setdefault(n.name, "Class")
        elif isinstance(n, ast.arg) and _is_confusable(n.arg):
            kinds.setdefault(n.arg, "arg")
    for n in ast.walk(tree):
        for name in ([n.id] if isinstance(n, ast.Name) else [n.attr] if isinstance(n, ast.Attribute) else []):
            if _is_confusable(name):
                kinds.setdefault(name, "var")
    if not kinds:
        return tree
    counters, mapping = {}, {}
    for name, kind in kinds.items():
        counters[kind] = counters.get(kind, 0) + 1
        mapping[name] = f"{kind}_{counters[kind]}"
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in mapping:
            n.id = mapping[n.id]
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n.name in mapping:
            n.name = mapping[n.name]
        elif isinstance(n, ast.arg) and n.arg in mapping:
            n.arg = mapping[n.arg]
        elif isinstance(n, ast.Attribute) and n.attr in mapping:
            n.attr = mapping[n.attr]
        elif isinstance(n, ast.keyword) and n.arg in mapping:
            n.arg = mapping[n.arg]
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            n.names = [mapping.get(x, x) for x in n.names]
    notes.append(f"renamed {len(mapping)} confusable identifiers (func_N / Class_N / var_N)")
    return tree


# ------------------------------------------------------------------ driver


def clean(src: str, *, fold=True, rename=True, inline=True, env=None) -> tuple[str, list[str]]:
    notes = []
    src, n = strip_boilerplate(src)
    notes += n
    try:
        tree = ast.parse(src)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return src, notes + ["source does not parse; AST clean-up skipped"]
    before = len(notes)
    if env:
        tree = propagate_globals(tree, env, notes)
    if inline:
        tree = inline_aliases(tree, notes)
    if rename:
        tree = rename_nonlatin(tree, notes)
    if fold:
        tree = fold_constants(tree, notes)
    tree = remove_anti_tamper(tree, notes)
    tree = remove_pyarmor_artifacts(tree, notes)
    tree = remove_dead_code(tree, notes)
    for _ in range(3):  # writes can feed other writes; iterate to a fixed point
        n = len(notes)
        tree = simplify_namespace_writes(tree, notes)
        if len(notes) == n:
            break
    if rename:
        tree = rename_confusables(tree, notes)
    if len(notes) == before:
        return src, notes
    tree = drop_redundant_pass(tree)
    ast.fix_missing_locations(tree)
    try:
        return ast.unparse(tree) + "\n", notes
    except (RecursionError, ValueError):
        return src, notes + ["unparse failed; AST clean-up skipped"]

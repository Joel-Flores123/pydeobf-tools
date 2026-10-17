"""Layer-peeling engine.

Input (.pyc or possibly-broken .py)
  -> Layer 0
  -> emulate (sandboxed, see vm.py) and collect exec/eval/marshal payloads
  -> each payload becomes a child layer, recursively
  -> fallbacks: broken-source salvage, blob brute-force, PyLingual decompile
     of code the emulator cannot handle
  -> leaves (and substantial non-wrapper code layers) are decompiled with
     PyLingual when they are bytecode, then cleaned up.
"""

from __future__ import annotations

import ast
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import blobs
from . import bytecode as bc
from . import cleanup
from . import detect
from . import pylingual_bridge
from . import repair
from . import vm as V


@dataclass
class Options:
    version_hint: tuple | None = None
    use_pylingual: bool = True
    decompile_all: bool = False
    pylingual_timeout: int = 600
    max_steps: int = 30_000_000
    max_seconds: float = 120.0
    max_layers: int = 400
    platform: str = "windows"
    pyarmor_runtime: str | None = None
    pyarmor: bool = True
    fold: bool = True
    rename: bool = True
    blob_search: bool = True
    quiet: bool = False


@dataclass
class Layer:
    index: int
    parent: int | None
    kind: str
    origin: str
    source: str | None = None
    blob: bc.CodeBlob | None = None
    globals: dict | None = None
    detections: list = field(default_factory=list)
    events: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    children: list = field(default_factory=list)
    emu: dict = field(default_factory=dict)
    wrapper: bool = False
    decompiled: str | None = None
    decompile_info: dict | None = None
    final: str | None = None
    salvaged: str | None = None
    prerun: object = None  # vm.Payload from inline emulation inside the parent
    _bcc: list = field(default_factory=list)  # PyArmor BCC native segments
    files: dict = field(default_factory=dict)

    @property
    def version(self):
        return self.blob.version if self.blob else None

    def summary(self):
        d = {
            "index": self.index, "parent": self.parent, "kind": self.kind, "origin": self.origin,
            "python_version": bc.vstr(self.version) if self.version else None,
            "size": len(self.source) if self.source is not None else (len(self.blob.raw) if self.blob and self.blob.raw else None),
            "detections": self.detections, "children": self.children, "wrapper": self.wrapper,
            "emulation": self.emu, "notes": self.notes, "events": self.events[:300],
            "decompile": self.decompile_info, "files": self.files,
        }
        return d


def _last_line(text):
    lines = [ln.strip() for ln in str(text).splitlines() if ln.strip()]
    return lines[-1] if lines else str(text)


def _comment_header(text):
    out = []
    for line in text.splitlines(True):
        if not line.startswith("#"):
            break
        out.append(line)
    return "".join(out)


class Engine:
    KNOWN_WRAPPERS = {"Specter", "Berserker", "Kramer", "PyObfuscate", "pyminifier", "Hyperion"}

    def __init__(self, options: Options | None = None):
        self.o = options or Options()
        self.layers: list[Layer] = []
        self.seen = set()
        self.version_hint = self.o.version_hint
        self.emu: V.Emulator | None = None
        self.pyarmor: dict | None = None

    # ------------------------------------------------------------ utils

    def log(self, msg):
        if not self.o.quiet:
            print(f"[pydeobf] {msg}", flush=True)

    def _key(self, kind, value):
        if kind == "source":
            data = value.encode("utf-8", "surrogatepass")
        else:
            data = value.raw if value.raw is not None else repr(value.code.co_code).encode()
        return hashlib.sha256(kind.encode() + data).hexdigest()

    def add_layer(self, parent, kind, value, origin, globals_=None):
        key = self._key(kind, value)
        if key in self.seen:
            return None
        if len(self.layers) >= self.o.max_layers:
            if parent is not None:
                self.layers[parent].notes.append("layer limit reached; payload not processed")
            return None
        self.seen.add(key)
        lay = Layer(index=len(self.layers), parent=parent, kind=kind, origin=origin, globals=globals_)
        if kind == "source":
            lay.source = value
        else:
            lay.blob = value
        self.layers.append(lay)
        if parent is not None:
            self.layers[parent].children.append(lay.index)
        return lay

    # ------------------------------------------------------------- input

    def load(self, path: Path):
        data = path.read_bytes()
        self.input_path = path
        self.input_data = data
        self.emu = V.Emulator(version_hint=self.version_hint, max_steps=self.o.max_steps,
                              max_seconds=self.o.max_seconds, self_file=str(path), self_file_data=data,
                              platform=self.o.platform)
        if bc.looks_like_pyc(data):
            blob = bc.read_pyc(data)
            self.version_hint = self.version_hint or blob.version
            self.add_layer(None, "code", blob, "input (.pyc)")
        else:
            self.add_layer(None, "source", repair.decode_source(data), "input (.py)")

    # ----------------------------------------------------------- process

    def run(self, path: Path):
        self.load(Path(path))
        i = 0
        while i < len(self.layers):
            lay = self.layers[i]
            self.log(f"layer {lay.index}: {lay.kind} via {lay.origin}"
                     + (f" (Python {bc.vstr(lay.version)})" if lay.version else ""))
            try:
                self.process(lay)
            except (V.VMSignal, RecursionError) as e:
                lay.notes.append(f"emulation aborted: {type(e).__name__}: {e}")
            i += 1
        self.finish()
        return self

    def _detect(self, lay):
        if lay.kind == "source":
            lay.detections = detect.detect_text(lay.source)
            raw = lay.source.encode("utf-8", "surrogatepass")
            if "PyArmor" in lay.detections:
                raw += b"\n" + b"\n".join(c if isinstance(c, bytes) else c.encode("utf-8", "surrogatepass")
                                          for c in self._constants(lay))
        else:
            strs = bc.string_constants(lay.blob.code, 4)
            names = [n for c in bc.iter_code_objects(lay.blob.code) for n in c.co_names]
            lay.detections = detect.detect_code_strings(strs, names)
            raw = b"\n".join(s if isinstance(s, bytes) else s.encode("utf-8", "surrogatepass") for s in strs) \
                + b"\n" + " ".join(names).encode()
        if "PyArmor" in lay.detections:
            info = detect.pyarmor_info(raw)
            self.pyarmor = {**(self.pyarmor or {}), **info, "layer": lay.index}

    def _resolve_runtime(self):
        if self.o.pyarmor_runtime:
            return self.o.pyarmor_runtime
        exts = (".so", ".pyd", ".dylib")
        base = Path(self.input_path).resolve().parent
        cands = []
        for d in [base] + sorted(base.glob("pyarmor_runtime_*")) + sorted(base.glob("*/pyarmor_runtime_*")):
            if d.is_dir():
                for e in exts:
                    cands += sorted(d.glob(f"pyarmor_runtime*{e}"))
        return str(cands[0]) if cands else None

    def _try_pyarmor(self, lay):
        """Statically decrypt a PyArmor-obfuscated source layer into a standard
        .pyc child layer. Never loads the runtime or runs the sample."""
        if lay.kind != "source" or lay.children:
            return
        from . import pyarmor as pa
        if pa.find_pyarmor_blob(lay.source) is None:
            return
        runtime = self._resolve_runtime()
        try:
            up = pa.unpack(lay.source, runtime_path=runtime)
        except Exception as e:
            lay.notes.append(f"PyArmor unpack failed: {type(e).__name__}: {e}")
            return
        if up is None:
            return
        self.version_hint = up.version
        self.pyarmor = {**(self.pyarmor or {}), "unpacked": True, "key_source": up.key.source,
                        "target_python": bc.vstr(up.version), "layer": lay.index}
        lay.notes.append(f"PyArmor decrypted statically ({up.key.source}); recovered standard "
                         f"Python {bc.vstr(up.version)} bytecode")
        if up.bcc_segments:
            self.pyarmor["bcc_segments"] = [a for a, _ in up.bcc_segments]
            lay.notes.append(f"BCC mode: {len(up.bcc_segments)} native segment(s) extracted "
                             "(function bodies compiled to machine code are not in the bytecode)")
            lay._bcc = up.bcc_segments
        blob = bc.read_pyc(up.pyc)
        child = self.add_layer(lay.index, "code", blob, "pyarmor-unpack")
        if child is None:
            lay.notes.append("PyArmor payload already seen")

    def _emulate(self, lay, code, version):
        emu = self.emu
        emu.payloads = []
        emu.events = []
        emu.ctx = []
        emu.steps = 0
        emu.assumptions = 0
        emu.version_hint = self.version_hint
        g = lay.globals if lay.globals is not None else emu.new_globals()
        t0 = time.monotonic()
        res = emu.run(code, version, g)
        lay.emu = {"error": res.error, "stopped": res.stop_reason, "steps": res.steps,
                   "unknown_branches": res.assumptions, "seconds": round(time.monotonic() - t0, 3),
                   "payloads": len(res.payloads)}
        lay.events.extend(res.events)
        if emu.version_hint:
            self.version_hint = emu.version_hint
        return res

    def _compile_source(self, lay):
        code, err = repair.try_compile(lay.source, f"<layer{lay.index}>")
        if code is not None:
            return code
        lay.notes.append(f"source does not compile: {type(err).__name__}: {err}")
        fixed = repair.py2_fixes(lay.source)
        if fixed != lay.source:
            code, _ = repair.try_compile(fixed)
            if code is not None:
                lay.notes.append("applied Python 2 -> 3 syntax fixes for emulation")
                return code
        salvaged, notes = repair.salvage(lay.source)
        lay.notes.extend(notes[:50])
        if salvaged.strip():
            code, _ = repair.try_compile(salvaged)
            if code is not None:
                lay.notes.append("emulating the compilable subset of the source")
                dropped = [n.split(": ", 1)[1] for n in notes if n.startswith("dropped")]
                lay.salvaged = ("# pydeobf: uncompilable statements were dropped:\n"
                                + "".join(f"#   {d}\n" for d in dropped[:50]) + salvaged)
                return code
        return None

    def _use_prerun(self, lay):
        """Reuse the inline emulation the parent already did for this payload."""
        pre = lay.prerun
        if pre is None:
            return False
        if not (pre.emulated or pre.stop or (pre.raised and not pre.error)):
            if pre.error:
                lay.notes.append(f"inline emulation incomplete ({pre.error}); re-processing")
            return False
        lay.emu = {"inline": True, "error": pre.error, "raised": pre.raised, "stopped": pre.stop,
                   "payloads": len(pre.children)}
        lay.events.extend(pre.events)
        for c in pre.children:
            self._add_payload(lay, c)
        return True

    def process(self, lay: Layer):
        self._detect(lay)
        if self.o.pyarmor and "PyArmor" in lay.detections:
            self._try_pyarmor(lay)
            if lay.children:
                return
            lay.notes.append(detect.PYARMOR_NOTE)
        res = None
        if self._use_prerun(lay):
            pass
        elif lay.kind == "source":
            code = self._compile_source(lay)
            if code is not None:
                res = self._emulate(lay, code, bc.HOST_VERSION)
        else:
            if lay.blob.version in bc.EMULATED_VERSIONS:
                res = self._emulate(lay, lay.blob.code, lay.blob.version)
            else:
                lay.notes.append(f"Python {bc.vstr(lay.blob.version)} bytecode is not emulated; "
                                 "decompiling first and emulating the decompiled source")
        if res is not None:
            for p in res.payloads:
                self._add_payload(lay, p)
        # Fallback 1: bytecode the emulator could not follow -> decompile and retry at source level
        if lay.kind == "code" and not lay.children and (res is None or res.error) and self.o.use_pylingual:
            self.decompile_layer(lay)
            if lay.decompiled:
                code = self._compile_decompiled(lay)
                if code is not None:
                    res2 = self._emulate(lay, code, bc.HOST_VERSION)
                    for p in res2.payloads:
                        self._add_payload(lay, p, origin_prefix="pylingual+")
        # Fallback 2: brute-force decode long constants
        if not lay.children and self.o.blob_search:
            consts = self._constants(lay)
            for cand in blobs.search_many(consts, self.version_hint):
                child = self.add_layer(lay.index, cand.kind, cand.value, "blob:" + "+".join(cand.chain),
                                       lay.globals)
                if child is not None:
                    lay.notes.append(f"decoded embedded blob via {' -> '.join(cand.chain)}")
        if lay.kind == "code" and lay.children:
            lay.wrapper = self._is_wrapper(lay)

    def _constants(self, lay):
        if lay.kind == "code":
            return bc.string_constants(lay.blob.code, blobs.MIN_BLOB)
        try:
            tree = ast.parse(lay.source)
            return [n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                    and isinstance(n.value, (str, bytes)) and len(n.value) >= blobs.MIN_BLOB]
        except (SyntaxError, ValueError, RecursionError, MemoryError):
            return repair.literal_strings(lay.source, blobs.MIN_BLOB)

    def _compile_decompiled(self, lay):
        code, _ = repair.try_compile(lay.decompiled)
        if code is None:
            salvaged, _ = repair.salvage(lay.decompiled)
            code, _ = repair.try_compile(salvaged)
        return code

    def _add_payload(self, lay, p: V.Payload, origin_prefix=""):
        if p.kind == "source":
            if self._trivial_source(p.value):
                return
            if p.value.lstrip().startswith(("#", "\x01")) and self._trivial_source(p.value.replace("\x01", "")):
                return
        else:
            if self._trivial_code(p.value.code):
                return
            if p.value.version:
                self.version_hint = p.value.version
        child = self.add_layer(lay.index, p.kind, p.value, origin_prefix + p.via, p.globals)
        if child is None:
            lay.notes.append(f"{p.via} payload already seen (loop or duplicate)")
        else:
            child.prerun = p

    @staticmethod
    def _trivial_source(src):
        try:
            tree = ast.parse(src)
        except (SyntaxError, ValueError):
            return len(src.strip()) < 3
        except (RecursionError, MemoryError):
            return False
        def literal(node):
            if isinstance(node, ast.Name):
                return True
            try:
                ast.literal_eval(node)  # literal parsing only
                return True
            except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
                return False
        return all(isinstance(n, ast.Expr) and literal(n.value) for n in tree.body)

    @staticmethod
    def _trivial_code(code):
        try:
            return not code.co_names and not any(bc.is_code(c) for c in code.co_consts) and len(code.co_code) <= 12
        except Exception:
            return False

    BEHAVIOUR_PREFIXES = ("call ", "import ", "open(", "print:", "input(", "setattr", "time.sleep")

    def _is_wrapper(self, lay):
        """A bytecode layer that produced payloads but showed no behaviour of
        its own (no calls/imports/IO besides exec/eval/marshal) is a wrapper
        and is not worth decompiling."""
        if any(d in self.KNOWN_WRAPPERS for d in lay.detections):
            return True
        return not any(e.startswith(self.BEHAVIOUR_PREFIXES) for e in lay.events)

    # ------------------------------------------------------------ finish

    def decompile_layer(self, lay):
        if lay.decompiled is not None or lay.decompile_info is not None:
            return
        if not self.o.use_pylingual:
            lay.decompile_info = {"ok": False, "error": "PyLingual disabled (--no-pylingual)"}
            return
        if self.workdir is None:
            return
        self.log(f"layer {lay.index}: decompiling Python {bc.vstr(lay.blob.version)} bytecode with PyLingual")
        d = pylingual_bridge.decompile(lay.blob, self.workdir / "layers", f"L{lay.index:02d}",
                                       timeout=self.o.pylingual_timeout)
        lay.decompiled = d.source
        lay.decompile_info = d.info
        if d.source is None:
            lay.notes.append(f"PyLingual failed: {_last_line(d.info.get('error', '?'))[:300]}")

    workdir: Path | None = None

    def finish(self):
        for lay in self.layers:
            leaf = not lay.children
            if lay.kind == "code" and (leaf or self.o.decompile_all or not lay.wrapper):
                size = sum(len(c.co_code) for c in bc.iter_code_objects(lay.blob.code))
                if not leaf and not self.o.decompile_all and size > 200_000:
                    lay.notes.append(f"not decompiled: {size} bytes of bytecode in a non-final layer "
                                     "(use --decompile-all)")
                    continue
                self.decompile_layer(lay)
        self.groups = []
        for group in self._leaf_groups():
            texts = [(l.salvaged or l.source) if l.kind == "source" else l.decompiled for l in group]
            texts = [t for t in texts if t]
            if not texts:
                continue
            merged = "\n".join(t.rstrip("\n") + "\n" for t in texts)
            env = self._env_for(group[0])
            final, notes = cleanup.clean(merged, fold=self.o.fold, rename=self.o.rename, env=env)
            banner = "".join(_comment_header(l.salvaged)
                             for l in group if l.salvaged)
            if banner and not final.startswith(banner):
                final = banner + final
            head = group[0]
            di = head.decompile_info or {}
            rate = di.get("success_rate")
            if head.kind == "code" and di.get("ok") and rate is not None and rate < 0.5:
                final = (
                    f"# pydeobf: PyLingual decompilation was UNRELIABLE for this layer "
                    f"(equivalence {rate:.0%}) - the source below is likely wrong.\n"
                    f"# Authoritative output: the bytecode disassembly layers/L{head.index:02d}.das.txt\n"
                    f"# and the recovered standard .pyc layers/L{head.index:02d}_py"
                    f"{head.blob.version[0]}{head.blob.version[1]}.pyc (run a {bc.vstr(head.blob.version)}"
                    f"-capable decompiler on it).\n\n"
                ) + final
                head.notes.append(f"PyLingual equivalence {rate:.0%}; see the disassembly (authoritative)")
            head.final = final
            head.notes.extend(notes)
            if len(group) > 1:
                head.notes.append(f"merged {len(group)} sibling exec chunks (layers {group[0].index}-{group[-1].index})")
                for other in group[1:]:
                    other.notes.append(f"merged into layer {head.index}")
            self.groups.append(group)

    def _leaf_groups(self):
        """Consecutive leaf siblings (same parent) form one program, e.g. code
        that was exec'd statement-by-statement (Hyperion's shell layer)."""
        groups = []
        for parent in [None] + list(range(len(self.layers))):
            kids = [self.layers[0]] if parent is None else [self.layers[i] for i in self.layers[parent].children]
            run = []
            for k in kids:
                if not k.children:
                    run.append(k)
                elif run:
                    groups.append(run)
                    run = []
            if run:
                groups.append(run)
            if parent is None and kids and kids[0].children:
                groups = [g for g in groups if g[0] is not kids[0]]
        return groups

    def _env_for(self, lay):
        g = lay.prerun.globals if lay.prerun is not None and lay.prerun.globals is not None else lay.globals
        return g if isinstance(g, dict) else None

    # ----------------------------------------------------------- output

    def leaves(self):
        return [l for l in self.layers if not l.children]

    def write(self, outdir: Path):
        outdir.mkdir(parents=True, exist_ok=True)
        ld = outdir / "layers"
        ld.mkdir(exist_ok=True)
        for lay in self.layers:
            stem = f"L{lay.index:02d}"
            if lay.kind == "source":
                p = ld / f"{stem}.py"
                p.write_text(lay.source, encoding="utf-8", errors="surrogateescape")
                lay.files["source"] = str(p)
            elif lay.blob.raw is not None:
                p = ld / f"{stem}_py{lay.blob.version[0]}{lay.blob.version[1]}.pyc"
                p.write_bytes(bc.make_pyc(lay.blob.raw, lay.blob.version))
                lay.files["pyc"] = str(p)
            if lay.decompiled:
                p = ld / f"{stem}.decompiled.py"
                p.write_text(lay.decompiled, encoding="utf-8", errors="surrogateescape")
                lay.files["decompiled"] = str(p)
            if lay.final:
                p = ld / f"{stem}.final.py"
                p.write_text(lay.final, encoding="utf-8", errors="surrogateescape")
                lay.files["final"] = str(p)
            for i, (arch, elf) in enumerate(lay._bcc):
                p = ld / f"{stem}.bcc.{arch}.{i}.elf"
                p.write_bytes(elf)
                lay.files.setdefault("bcc", []).append(str(p))
            # always-correct disassembly for code layers (reliable when PyLingual is noisy)
            if lay.kind == "code":
                try:
                    das = bc.disassemble(lay.blob.code, lay.blob.version)
                    p = ld / f"{stem}.das.txt"
                    p.write_text(das, encoding="utf-8", errors="surrogateescape")
                    lay.files["disassembly"] = str(p)
                except Exception:
                    pass
        finals = [l for l in self.layers if l.final]
        if finals:
            parts = []
            for l in finals:
                if len(finals) > 1:
                    parts.append(f"# ===== pydeobf: layer {l.index} ({l.origin}) =====\n")
                parts.append(l.final)
            (outdir / "final.py").write_text("\n".join(parts), encoding="utf-8", errors="surrogateescape")
        report = self.report()
        (outdir / "report.json").write_text(json.dumps(report, indent=2, default=str))
        (outdir / "report.txt").write_text(self.render(report))
        return report

    def report(self):
        events = []
        for l in self.layers:
            for e in l.events:
                if e.startswith(("call ", "import ", "open(", "print:", "input(", "time.sleep", "setattr")):
                    events.append(f"L{l.index}: {e}")
        return {
            "input": str(self.input_path),
            "layers": [l.summary() for l in self.layers],
            "detected_obfuscators": sorted({d for l in self.layers for d in l.detections}),
            "final_layers": [l.index for l in self.layers if l.final] or [l.index for l in self.leaves()],
            "pyarmor": self.pyarmor,
            "recovered_strings": self._recovered_strings(),
            "interesting_events": events[:1000],
        }

    _IOC_RE = __import__("re").compile(r"https?://|\\\\|%[A-Za-z]+%|\.(?:php|json|exe|dll|bin|ps1|bat)\b"
                                       r"|discord|webhook|token|api|AppData|[A-Za-z]:\\")

    def _recovered_strings(self, limit=200):
        """Notable string constants recovered from every code layer: URLs, paths,
        and other IOC-ish or long strings. Pure data pulled from the bytecode."""
        import re
        seen, urls, iocs, other = set(), [], [], []
        for lay in self.layers:
            if lay.kind != "code" or lay.blob is None:
                continue
            for s in bc.string_constants(lay.blob.code, 4):
                if not isinstance(s, str) or s in seen or "pyarmor" in s:
                    continue
                if s.endswith((".<locals>.<lambda>", ".<locals>.<genexpr>", ".<locals>.<listcomp>")):
                    continue
                seen.add(s)
                if re.match(r"https?://", s):
                    urls.append(s)
                elif self._IOC_RE.search(s):
                    iocs.append(s)
                elif len(s) >= 8:
                    other.append(s)
        other.sort(key=len, reverse=True)
        return {"urls": urls[:limit], "iocs": iocs[:limit], "other": other[:limit]}

    @staticmethod
    def render(report):
        out = [f"input: {report['input']}",
               f"detected: {', '.join(report['detected_obfuscators']) or 'nothing specific'}", ""]
        out.append(f"{'#':>3} {'parent':>6} {'kind':6} {'py':5} {'size':>9}  origin / notes")
        for l in report["layers"]:
            par = "" if l["parent"] is None else str(l["parent"])
            out.append(f"{l['index']:>3} {par:>6} {l['kind']:6} {l['python_version'] or '':5} {l['size'] or 0:>9}  "
                       f"{l['origin']}{'  [wrapper]' if l['wrapper'] else ''}"
                       f"{'  [FINAL]' if l['index'] in report['final_layers'] else ''}")
            emu = l["emulation"] or {}
            if emu.get("error") or emu.get("stopped"):
                out.append(f"{'':28}emulation: {emu.get('error') or emu.get('stopped')}")
            for n in l["notes"][:8]:
                out.append(f"{'':28}- {n[:160]}")
            if l["decompile"]:
                d = l["decompile"]
                out.append(f"{'':28}pylingual: " + (f"ok, equivalence {d.get('success_rate', 0):.0%}" if d.get("ok")
                                                    else f"failed: {_last_line(d.get('error'))[:160]}"))
        if report["pyarmor"]:
            out += ["", f"PyArmor: {report['pyarmor']}"]
        rs = report.get("recovered_strings") or {}
        if rs.get("urls") or rs.get("iocs") or rs.get("other"):
            out += ["", "recovered strings (IOC candidates, from the decrypted/decoded bytecode):"]
            for u in rs.get("urls", []):
                out.append(f"  URL  {u[:200]}")
            for s in rs.get("iocs", []):
                out.append(f"  IOC  {s[:200]!r}")
            for s in rs.get("other", [])[:40]:
                out.append(f"       {s[:200]!r}")
        if report["interesting_events"]:
            out += ["", "behaviour observed during emulation (nothing was executed):"]
            out += [f"  {e[:200]}" for e in report["interesting_events"][:200]]
        return "\n".join(out) + "\n"

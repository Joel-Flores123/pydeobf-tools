"""Make broken / partial Python source usable for (static) emulation."""

from __future__ import annotations

import io
import re
import tokenize


def decode_source(data: bytes) -> str:
    data = data.lstrip(b"\xef\xbb\xbf")
    enc = "utf-8"
    for line in data.split(b"\n", 2)[:2]:
        m = re.match(rb"^[ \t\f]*#.*?coding[:=][ \t]*([-\w.]+)", line)
        if m:
            enc = m.group(1).decode()
            break
    for e in (enc, "utf-8", "latin-1"):
        try:
            return data.decode(e)
        except (UnicodeDecodeError, LookupError):
            continue
    return data.decode("latin-1")


def try_compile(src: str, name="<layer>"):
    """compile() parses and generates bytecode; it never executes anything."""
    try:
        return compile(src, name, "exec", dont_inherit=True), None
    except (SyntaxError, ValueError) as e:
        return None, e
    except (MemoryError, RecursionError) as e:
        return None, e


_PRINT_STMT = re.compile(r"^(\s*)print\s+(?![\s(=])(.+)$", re.M)
_EXEC_STMT = re.compile(r"^(\s*)exec\s+(?![\s(=])(.+?)(?:\s+in\s+(.+))?$", re.M)
_EXCEPT_COMMA = re.compile(r"^(\s*except\s+[\w.]+)\s*,\s*(\w+)\s*:", re.M)
_UR = re.compile(r"\bur(['\"])")


def py2_fixes(src: str) -> str:
    src = _PRINT_STMT.sub(lambda m: f"{m.group(1)}print({m.group(2).rstrip().rstrip(',')})", src)
    src = _EXEC_STMT.sub(lambda m: f"{m.group(1)}exec({m.group(2)}{', ' + m.group(3) if m.group(3) else ''})", src)
    src = _EXCEPT_COMMA.sub(r"\1 as \2:", src)
    src = _UR.sub(r"r\1", src)
    return src


_CONT = re.compile(r"^(?:else|elif|except|finally|\)|\]|\}|#|@)")


def top_level_chunks(src: str):
    """Split source into top-level statements by indentation."""
    lines = src.splitlines(True)
    chunks, cur = [], []
    for ln in lines:
        starts_new = ln[:1] not in (" ", "\t", "\n", "\r", "") and not _CONT.match(ln)
        if starts_new and cur:
            chunks.append("".join(cur))
            cur = []
        cur.append(ln)
    if cur:
        chunks.append("".join(cur))
    return chunks


def salvage(src: str, max_merge=40):
    """Return (compilable_source, notes) keeping every top-level statement that
    compiles (merging neighbours to repair multi-chunk constructs)."""
    chunks = top_level_chunks(src)
    kept, notes = [], []
    i = 0
    while i < len(chunks):
        ok = False
        for j in range(i + 1, min(len(chunks), i + max_merge) + 1):
            piece = "".join(chunks[i:j])
            code, err = try_compile(piece)
            if code is None:
                fixed = py2_fixes(piece)
                if fixed != piece:
                    code, err = try_compile(fixed)
                    piece = fixed
            if code is not None:
                kept.append(piece)
                i = j
                ok = True
                break
        if not ok:
            first = chunks[i].strip().splitlines()[0][:80] if chunks[i].strip() else ""
            notes.append(f"dropped uncompilable statement: {first!r}")
            i += 1
    return "".join(kept), notes


def literal_strings(src: str, min_len=48):
    """Long string literals found in (possibly broken) source."""
    out = []
    try:
        for tok in tokenize.generate_tokens(io.StringIO(src).readline):
            if tok.type == tokenize.STRING and len(tok.string) >= min_len:
                out.append(tok.string)
    except (tokenize.TokenError, IndentationError, SyntaxError):
        pass
    if not out:
        out = [m.group(0) for m in re.finditer(r"""(?:[bBrRuU]{0,2})(?:'''[\s\S]*?'''|\"\"\"[\s\S]*?\"\"\"|'(?:[^'\\\n]|\\.)*'|"(?:[^"\\\n]|\\.)*")""", src)
               if len(m.group(0)) >= min_len]
    values = []
    import ast
    for lit in out:
        try:
            values.append(ast.literal_eval(lit))  # literal parsing only
        except Exception:
            body = lit.strip("bBrRuU")[1:-1]
            values.append(body)
    return values

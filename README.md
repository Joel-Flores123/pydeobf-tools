# pydeobf

Static deobfuscator for Python malware and protected scripts. It peels
obfuscation layers **without ever executing the sample**, then (optionally)
decompiles the recovered bytecode back to source.

## Features

- Sandboxed bytecode emulator for CPython 3.6–3.15. It follows `exec`/`eval`/`marshal.loads` chains and records each payload as a layer.
- Static PyArmor 8/9 unpacking. The runtime library is only read as bytes, never loaded.
- Blob search for base64, base32, hex, zlib, bz2, lzma, marshal, rot13 and shift encodings.
- Source clean-up: folds encoded strings, inlines aliases, removes dead code, renames confusable identifiers.
- IOC report: URLs, paths and other strings, plus every file, OS and network call the sample attempted.

Supported: Specter, Berserker, Kramer, Kyrie-Eleison, PyObfuscate, pyminifier,
Hyperion, PyArmor 8/9. Names renamed by pyminifier or Hyperion can't be
recovered. PyArmor 7 is only detected.

## Install

```
uv sync                       # core
uv sync --extra decompile     # + PyLingual, for bytecode -> source
```

Requires Python 3.12+. PyLingual pulls in torch and uses the GPU if one is
visible. Set `CUDA_VISIBLE_DEVICES=""` to force CPU.

## Usage

```
pydeobf sample.py                  # -> sample.py.deobf/{final.py, report.txt, report.json, layers/}
pydeobf samples/ -o out/           # whole directory, recursive
pydeobf armored.py --pyarmor-runtime pyarmor_runtime.so
```

Input is a `.pyc` (even if named `.py`) or a possibly broken `.py`. Run
`pydeobf --help` for all options.

Output:

| file | contents |
|---|---|
| `final.py` | recovered source |
| `report.txt` / `report.json` | layers found, IOCs, logged behaviour |
| `layers/` | every intermediate layer, as `.py`, `.pyc` and disassembly (`.das.txt`) |

## How it works

1. **Emulator** (`vm.py`, `safelib.py`): a CPython bytecode interpreter written
   in Python. It calls host functions only from an allow-list of side-effect-free
   ones. Anything else (files, network, `ctypes`, introspection escapes) returns
   a placeholder and is logged. Instruction, time and memory budgets stop runaway samples.
2. **Blob search** (`blobs.py`): decodes long constants the emulator couldn't resolve.
3. **PyArmor** (`pyarmor.py`): decrypts the outer blob and each code object using
   the runtime's key, or the public trial key, and rebuilds a standard `.pyc`.
4. **PyLingual** (`pylingual_bridge.py`, optional): the only bytecode-to-source step,
   run in a subprocess. Source-level obfuscators and PyArmor unpacking don't need it.
   Without it (or with `--no-pylingual`), bytecode layers are written as `.pyc` plus a disassembly.
5. **Clean-up** (`cleanup.py`): the source passes listed above.

PyArmor decryption is exact, but PyLingual is weak on PyArmor output for Python
3.8–3.10. In that case `final.py` says so and points to the disassembly in
`layers/*.das.txt`, which is always correct.

## Safety

Samples are never run. Don't change that: extend the emulator's allow-list
instead of calling `exec`, `eval` or `marshal.loads` on sample data.

## Credits

PyArmor unpacking follows the algorithm of
[Pyarmor-Static-Unpack-1shot](https://github.com/Lil-House/Pyarmor-Static-Unpack-1shot)
and cyber.wtf's "Unpacking Pyarmor v8+ scripts". Bytecode is decoded with
[xdis](https://github.com/rocky/python-xdis) and decompiled with
[PyLingual](https://github.com/syssec-utd/pylingual).

## License

GPL-3.0-or-later. See [LICENSE](LICENSE).

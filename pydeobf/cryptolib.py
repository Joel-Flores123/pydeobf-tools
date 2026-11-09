"""Side-effect-free stand-ins for the crypto libraries that loaders decrypt
their payload with: pycryptodome (``Crypto`` / ``Cryptodome``) and the common
parts of ``cryptography`` (Fernet, AESGCM, Cipher/algorithms/modes, PKCS7,
PBKDF2HMAC), all computed with pycryptodome.

Ciphers and keys are handed to emulated code as HelperObjects exposing only
their pure methods. Random sources are deterministic, nothing touches the host.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import random

from Crypto.Cipher import AES, ARC2, ARC4, CAST, DES, DES3, PKCS1_OAEP, PKCS1_v1_5, Blowfish, ChaCha20, \
    ChaCha20_Poly1305, Salsa20
from Crypto.Hash import MD5, SHA1, SHA224, SHA256, SHA384, SHA512, HMAC as _HMAC
from Crypto.Protocol import KDF
from Crypto.PublicKey import RSA
from Crypto.Util import Counter, Padding, number

from . import vm as _vm

_rng = random.Random(1337)


def _rand(n):
    return _rng.randbytes(n)


def _bytes(v):
    if isinstance(v, str):
        return v.encode("latin-1")
    if isinstance(v, (bytearray, memoryview)):
        return bytes(v)
    return v


class Cipher(_vm.HelperObject):
    """A pycryptodome cipher (or anything with the same encrypt/decrypt surface)."""

    ALLOWED = frozenset({"decrypt", "encrypt", "decrypt_and_verify", "encrypt_and_digest", "verify", "digest",
                         "hexdigest", "update", "nonce", "iv", "IV", "block_size", "key_size", "mode",
                         "can_decrypt"})

    def __init__(self, c, iv=None):
        self._c = c
        self.nonce = getattr(c, "nonce", None)
        self.iv = self.IV = getattr(c, "iv", iv)
        self.block_size = getattr(c, "block_size", None)
        self.mode = getattr(c, "mode", None)

    def decrypt(self, data, *a, **k):
        return self._c.decrypt(_bytes(data), *a, **k)

    def encrypt(self, data, *a, **k):
        return self._c.encrypt(_bytes(data), *a, **k)

    def decrypt_and_verify(self, data, tag, *a, **k):
        # the payload is what matters; a bad tag (patched sample) shouldn't hide it
        try:
            return self._c.decrypt_and_verify(_bytes(data), _bytes(tag), *a, **k)
        except ValueError:
            return self._c.decrypt(_bytes(data))

    def encrypt_and_digest(self, data, *a, **k):
        return self._c.encrypt_and_digest(_bytes(data), *a, **k)

    def verify(self, *a, **k):
        return None

    def digest(self):
        return self._c.digest()

    def hexdigest(self):
        return self._c.hexdigest()

    def update(self, data):
        self._c.update(_bytes(data))
        return self

    def can_decrypt(self):
        return True


class HashObj(_vm.HelperObject):
    ALLOWED = frozenset({"update", "digest", "hexdigest", "copy", "digest_size", "block_size", "name", "new"})

    def __init__(self, h):
        self._h = h
        self.digest_size = h.digest_size
        self.block_size = getattr(h, "block_size", None)
        self.name = getattr(h, "name", None)

    def update(self, data):
        self._h.update(_bytes(data))
        return self

    def digest(self):
        return self._h.digest()

    def hexdigest(self):
        return self._h.hexdigest()

    def copy(self):
        return HashObj(self._h.copy())

    def new(self, data=None):
        return HashObj(self._h.new(_bytes(data)) if data is not None else self._h.new())


class RSAKey(_vm.HelperObject):
    ALLOWED = frozenset({"n", "e", "d", "p", "q", "size_in_bits", "size_in_bytes", "has_private", "publickey",
                         "public_key", "export_key", "exportKey", "can_encrypt", "can_sign"})

    def __init__(self, k):
        self._k = k
        self.n, self.e = k.n, k.e
        self.d = getattr(k, "d", None) if k.has_private() else None
        self.p = getattr(k, "p", None) if k.has_private() else None
        self.q = getattr(k, "q", None) if k.has_private() else None

    def size_in_bits(self):
        return self._k.size_in_bits()

    def size_in_bytes(self):
        return self._k.size_in_bytes()

    def has_private(self):
        return self._k.has_private()

    def publickey(self):
        return RSAKey(self._k.public_key())

    public_key = publickey

    def export_key(self, format="PEM", *a, **k):
        return self._k.export_key(format)

    exportKey = export_key

    def can_encrypt(self):
        return True

    def can_sign(self):
        return True


def _unwrap(v):
    return v._k if isinstance(v, RSAKey) else v


def _hash_arg(v):
    """The real Crypto.Hash module behind ``SHA256`` (module), ``SHA256.new`` or hashes.SHA256()."""
    if isinstance(v, _HashMod):
        return v.mod
    if isinstance(v, _vm.SafeModule):
        return _HASHES.get(v.__name__.rsplit(".", 1)[-1], v)
    return v


class _HashMod:
    """Crypto.Hash.<NAME> / cryptography hashes.<NAME>(): stands for an algorithm."""

    def __init__(self, name, mod):
        self.name, self.mod = name, mod

    def new(self, data=None):
        return HashObj(self.mod.new(_bytes(data)) if data is not None else self.mod.new())


# ------------------------------------------------------------ pycryptodome

def _cipher_new(mod):
    def new(key, *args, **kwargs):
        args = tuple(_bytes(a) for a in args)
        kwargs = {k: (_bytes(v) if isinstance(v, (str, bytearray, memoryview)) else v) for k, v in kwargs.items()}
        return Cipher(mod.new(_bytes(key), *args, **kwargs))
    new.__name__ = f"{mod.__name__.rsplit('.', 1)[-1]}.new"
    return new


def _block_mod(mod):
    attrs = {k: getattr(mod, k) for k in dir(mod) if k.startswith("MODE_") or k in ("block_size", "key_size")}
    attrs["new"] = _cipher_new(mod)
    if hasattr(mod, "adjust_key_parity"):
        attrs["adjust_key_parity"] = mod.adjust_key_parity
    return attrs


def _pk_new(mod):
    def new(key, *args, **kwargs):
        if "hashAlgo" in kwargs:
            kwargs["hashAlgo"] = _hash_arg(kwargs["hashAlgo"])
        return Cipher(mod.new(_unwrap(key), *args, **kwargs))
    return new


def _pkcs15_decrypt_wrapper(c):
    class _C(Cipher):
        def decrypt(self, data, sentinel=None, *a, **k):
            return self._c.decrypt(_bytes(data), sentinel, *a, **k)
    return _C(c)


def _import_key(data, passphrase=None):
    return RSAKey(RSA.import_key(_bytes(data), passphrase))


def _construct(tup, consistency_check=True):
    return RSAKey(RSA.construct(tuple(tup), consistency_check))


def _pbkdf2(password, salt, dkLen=16, count=1000, prf=None, hmac_hash_module=None):
    hmac_hash_module = _hash_arg(hmac_hash_module)
    if count > 5_000_000:
        raise _vm.VMStop("refusing huge PBKDF2 iteration count")
    return KDF.PBKDF2(_bytes(password), _bytes(salt), dkLen, count, None, hmac_hash_module)


def _scrypt(password, salt, key_len, N, r, p, num_keys=1):
    if N * r > 1 << 22:
        raise _vm.VMStop("refusing huge scrypt cost")
    return KDF.scrypt(_bytes(password), _bytes(salt), key_len, N, r, p, num_keys)


def _hmac_new(key, msg=b"", digestmod=None):
    digestmod = _hash_arg(digestmod)
    return HashObj(_HMAC.new(_bytes(key), _bytes(msg), digestmod))


_HASHES = {"MD5": MD5, "SHA1": SHA1, "SHA": SHA1, "SHA224": SHA224, "SHA256": SHA256, "SHA384": SHA384,
           "SHA512": SHA512}


def _pycryptodome(root):
    cipher_mods = {"AES": AES, "DES": DES, "DES3": DES3, "Blowfish": Blowfish, "ARC2": ARC2, "CAST": CAST}
    stream_mods = {"ARC4": ARC4, "ChaCha20": ChaCha20, "Salsa20": Salsa20, "ChaCha20_Poly1305": ChaCha20_Poly1305}
    mods = {
        root: {},
        f"{root}.Cipher": {},
        f"{root}.Util": {},
        f"{root}.Util.Padding": {"pad": Padding.pad, "unpad": Padding.unpad},
        f"{root}.Util.number": {"long_to_bytes": number.long_to_bytes, "bytes_to_long": number.bytes_to_long},
        f"{root}.Util.Counter": {"new": Counter.new},
        f"{root}.PublicKey": {},
        f"{root}.PublicKey.RSA": {"import_key": _import_key, "importKey": _import_key, "construct": _construct},
        f"{root}.Protocol": {},
        f"{root}.Protocol.KDF": {"PBKDF2": _pbkdf2, "scrypt": _scrypt},
        f"{root}.Hash": {},
        f"{root}.Hash.HMAC": {"new": _hmac_new},
        f"{root}.Random": {"get_random_bytes": _rand, "new": lambda: _vm.Opaque("Crypto.Random.new()")},
        f"{root}.Cipher.PKCS1_OAEP": {"new": _pk_new(PKCS1_OAEP)},
        f"{root}.Cipher.PKCS1_v1_5": {"new": lambda key, *a, **k: _pkcs15_decrypt_wrapper(PKCS1_v1_5.new(_unwrap(key)))},
    }
    for name, m in {**cipher_mods}.items():
        mods[f"{root}.Cipher.{name}"] = _block_mod(m)
    for name, m in stream_mods.items():
        mods[f"{root}.Cipher.{name}"] = {"new": _cipher_new(m)}
    for name, m in _HASHES.items():
        hm = _HashMod(name, m)
        mods[f"{root}.Hash.{name}"] = {"new": hm.new, "digest_size": m.digest_size}
    return mods


# ------------------------------------------------------------ cryptography

class Fernet(_vm.HelperObject):
    ALLOWED = frozenset({"decrypt", "encrypt", "extract_timestamp"})

    def __init__(self, key, backend=None):
        k = base64.urlsafe_b64decode(_bytes(key))
        if len(k) != 32:
            raise ValueError("Fernet key must be 32 url-safe base64-encoded bytes.")
        self._sign, self._enc = k[:16], k[16:]

    def decrypt(self, token, ttl=None):
        data = base64.urlsafe_b64decode(_bytes(token))
        if len(data) < 57 or data[0] != 0x80:
            raise ValueError("InvalidToken")
        iv, ct = data[9:25], data[25:-32]
        return Padding.unpad(AES.new(self._enc, AES.MODE_CBC, iv).decrypt(ct), 16)

    def encrypt(self, data):
        iv = _rand(16)
        body = b"\x80" + (1_700_000_000).to_bytes(8, "big") + iv + \
            AES.new(self._enc, AES.MODE_CBC, iv).encrypt(Padding.pad(_bytes(data), 16))
        return base64.urlsafe_b64encode(body + hmac.new(self._sign, body, hashlib.sha256).digest())

    def extract_timestamp(self, token):
        return int.from_bytes(base64.urlsafe_b64decode(_bytes(token))[1:9], "big")


def _fernet_generate_key():
    return base64.urlsafe_b64encode(_rand(32))


class AESGCM(_vm.HelperObject):
    ALLOWED = frozenset({"decrypt", "encrypt"})
    MODE = AES.MODE_GCM

    def __init__(self, key):
        self._key = _bytes(key)

    def decrypt(self, nonce, data, associated_data=None):
        data = _bytes(data)
        c = AES.new(self._key, self.MODE, nonce=_bytes(nonce))
        if associated_data:
            c.update(_bytes(associated_data))
        return c.decrypt(data[:-16])

    def encrypt(self, nonce, data, associated_data=None):
        c = AES.new(self._key, self.MODE, nonce=_bytes(nonce))
        if associated_data:
            c.update(_bytes(associated_data))
        ct, tag = c.encrypt_and_digest(_bytes(data))
        return ct + tag


class ChaCha20Poly1305(AESGCM):
    def decrypt(self, nonce, data, associated_data=None):
        data = _bytes(data)
        c = ChaCha20_Poly1305.new(key=self._key, nonce=_bytes(nonce))
        if associated_data:
            c.update(_bytes(associated_data))
        return c.decrypt(data[:-16])


class _Alg(_vm.HelperObject):
    """cryptography's algorithms.AES(key) etc.: just remembers the key."""

    ALLOWED = frozenset({"key", "name", "block_size", "key_size", "nonce"})

    def __init__(self, mod, name, key, nonce=None):
        self.mod, self.name, self.key, self.nonce = mod, name, _bytes(key), _bytes(nonce)
        self.block_size = getattr(mod, "block_size", 1) * 8
        self.key_size = len(self.key) * 8


class _Mode(_vm.HelperObject):
    ALLOWED = frozenset({"name", "initialization_vector", "nonce", "tag"})

    def __init__(self, name, iv=None, tag=None, min_tag_length=16):
        self.name = name
        self.initialization_vector = self.nonce = _bytes(iv)
        self.tag = _bytes(tag)


class _Ctx(_vm.HelperObject):
    """cryptography's encryptor()/decryptor() context, buffered until finalize()."""

    ALLOWED = frozenset({"update", "finalize", "finalize_with_tag", "authenticate_additional_data", "tag"})

    def __init__(self, alg, mode, decrypt):
        self.alg, self.modeobj, self.dec, self.buf, self.out_len, self.aad = alg, mode, decrypt, b"", 0, b""
        self.tag = None

    def _cipher(self):
        a, m = self.alg, self.modeobj
        if a.name == "ChaCha20":
            return ChaCha20.new(key=a.key, nonce=a.nonce[4:]) if len(a.nonce) == 16 else \
                ChaCha20.new(key=a.key, nonce=a.nonce)
        if a.name == "ARC4":
            return ARC4.new(a.key)
        mod = a.mod
        if m is None or m.name == "ECB":
            return mod.new(a.key, mod.MODE_ECB)
        if m.name == "CTR":
            return mod.new(a.key, mod.MODE_CTR, nonce=b"", initial_value=m.nonce)
        if m.name == "GCM":
            c = mod.new(a.key, mod.MODE_GCM, nonce=m.nonce)
            if self.aad:
                c.update(self.aad)
            return c
        if m.name in ("CFB", "CFB8"):
            return mod.new(a.key, mod.MODE_CFB, iv=m.nonce, segment_size=128 if m.name == "CFB" else 8)
        return mod.new(a.key, getattr(mod, f"MODE_{m.name}"), m.nonce)

    def authenticate_additional_data(self, data):
        self.aad += _bytes(data)

    def update(self, data):
        # buffer everything and emit on finalize: block modes need whole blocks and
        # update() returning b"" is legal for them
        self.buf += _bytes(data)
        return b""

    def finalize(self):
        c = self._cipher()
        out = c.decrypt(self.buf) if self.dec else c.encrypt(self.buf)
        if not self.dec and self.modeobj is not None and self.modeobj.name == "GCM":
            self.tag = c.digest()
        self.buf = b""
        return out

    def finalize_with_tag(self, tag):
        return self.finalize()


class _CipherCtx(_vm.HelperObject):
    ALLOWED = frozenset({"decryptor", "encryptor", "algorithm", "mode"})

    def __init__(self, algorithm, mode=None, backend=None):
        self.algorithm, self.mode = algorithm, mode

    def decryptor(self):
        return _Ctx(self.algorithm, self.mode, True)

    def encryptor(self):
        return _Ctx(self.algorithm, self.mode, False)


class _Padder(_vm.HelperObject):
    ALLOWED = frozenset({"update", "finalize"})

    def __init__(self, block_bits, unpad):
        self.n, self.unpad, self.buf = block_bits // 8, unpad, b""

    def update(self, data):
        self.buf += _bytes(data)
        return b""

    def finalize(self):
        buf, self.buf = self.buf, b""
        return Padding.unpad(buf, self.n) if self.unpad else Padding.pad(buf, self.n)


class _PKCS7(_vm.HelperObject):
    ALLOWED = frozenset({"padder", "unpadder"})

    def __init__(self, block_size):
        self.block_size = block_size

    def padder(self):
        return _Padder(self.block_size, False)

    def unpadder(self):
        return _Padder(self.block_size, True)


class _KDF(_vm.HelperObject):
    ALLOWED = frozenset({"derive", "verify"})

    def __init__(self, algorithm=None, length=32, salt=b"", iterations=1000, backend=None):
        self.alg, self.length, self.salt, self.iterations = algorithm, length, _bytes(salt), iterations

    def derive(self, key_material):
        mod = self.alg.mod if isinstance(self.alg, _HashMod) else SHA256
        return _pbkdf2(key_material, self.salt, self.length, self.iterations, hmac_hash_module=mod)

    def verify(self, key_material, expected):
        return None


def _alg(mod, name):
    def make(key, nonce=None):
        return _Alg(mod, name, key, nonce)
    make.__name__ = name
    return make


def _mode(name):
    def make(iv=None, tag=None, min_tag_length=16):
        return _Mode(name, iv, tag)
    make.__name__ = name
    return make


def _cryptography():
    hz = "cryptography.hazmat"
    return {
        "cryptography": {},
        "cryptography.fernet": {"Fernet": _FernetClass(), "InvalidToken": ValueError},
        "cryptography.hazmat": {},
        f"{hz}.backends": {"default_backend": lambda: None},
        f"{hz}.primitives": {},
        f"{hz}.primitives.ciphers": {"Cipher": _CipherCtx},
        f"{hz}.primitives.ciphers.algorithms": {
            "AES": _alg(AES, "AES"), "AES128": _alg(AES, "AES"), "AES256": _alg(AES, "AES"),
            "TripleDES": _alg(DES3, "TripleDES"), "Blowfish": _alg(Blowfish, "Blowfish"),
            "ChaCha20": _alg(ChaCha20, "ChaCha20"), "ARC4": _alg(ARC4, "ARC4")},
        f"{hz}.primitives.ciphers.modes": {n: _mode(n) for n in ("CBC", "ECB", "CTR", "GCM", "CFB", "CFB8", "OFB")},
        f"{hz}.primitives.ciphers.aead": {"AESGCM": AESGCM, "ChaCha20Poly1305": ChaCha20Poly1305},
        f"{hz}.primitives.padding": {"PKCS7": _PKCS7},
        f"{hz}.primitives.hashes": {n: (lambda m=m, n=n: _HashMod(n, m)) for n, m in _HASHES.items()},
        f"{hz}.primitives.kdf": {},
        f"{hz}.primitives.kdf.pbkdf2": {"PBKDF2HMAC": _KDF},
    }


class _FernetClass(_vm.HelperObject):
    """``Fernet(key)`` and ``Fernet.generate_key()``."""

    ALLOWED = frozenset({"generate_key"})

    def __call__(self, key, backend=None):
        return Fernet(key, backend)

    def generate_key(self):
        return _fernet_generate_key()


def all_modules() -> dict[str, dict]:
    return {**_pycryptodome("Crypto"), **_pycryptodome("Cryptodome"), **_cryptography()}

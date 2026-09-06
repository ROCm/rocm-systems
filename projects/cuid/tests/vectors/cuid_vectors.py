#!/usr/bin/env python3
"""CUID conformance vectors - the cross-layer format contract.

This is the single source of truth for the CUID wire format. Every producer -
the amdgpu kernel driver and this library - must reproduce every vector below
bit for bit. The vectors are normative in all their columns: the 16-octet
payload, the full HMAC digest where one is involved, and the rendered UUID.

Matching the UUID but not the payload means an offsetting pair of errors, and
the next field that changes will diverge.

Run directly to self-check and to regenerate tests/vectors/cuid_vectors.txt,
which is what the C++ suite reads. Every HMAC digest here is independently
reproducible with:

    openssl dgst -sha256 -mac HMAC -macopt key:<ascii-key> <payload.bin>
"""

import hashlib
import hmac
import sys

# ---------------------------------------------------------------------------
# Constants. See lib/src/hmac.h and the kernel's amdgpu_cuid.c; all three must
# agree byte for byte.
# ---------------------------------------------------------------------------
DEFAULT_SEED = b"AMD-CUID-DEFAULT-SEED-v1"  # 24 octets, no NUL, unpadded
TEMP_KEY = b"AMD-CUID-TEMP-KEY-v1"          # 20 octets, no NUL, unpadded
TEST_SEED = bytes(range(32))                # 00..1f, not a placeholder

# On-wire Component Type values.
PLATFORM, CPU, GPU, NIC, NPU, OTHER = 0x0, 0x1, 0x2, 0x3, 0x4, 0xF

AUX_FORMAT_PCIE = 1
AUX_FORMAT_CPU = 2


def pack_primary(serial, unit_id, revision_id, device_id, vendor_id, component_type, aux=0):
    """The 122-bit primary payload, LSB-first into 16 octets."""
    p = serial & ((1 << 64) - 1)
    p |= (unit_id & 0xFF) << 64
    p |= (revision_id & 0xFF) << 72
    p |= (device_id & 0xFFFF) << 80
    p |= (vendor_id & 0xFFFF) << 96
    p |= ((unit_id >> 8) & 0x1F) << 112
    p |= (aux & 0x1) << 117
    p |= (component_type & 0xF) << 118
    assert p < (1 << 122), "payload overflows 122 bits"
    return bytes((p >> (8 * i)) & 0xFF for i in range(16))


def to_uuidv8(r):
    """Insert the RFC 9562 version/variant bits; payload LSB goes to the front.

    Lossless over payload bits 0:121; payload 122:127 are zero padding and are
    the six bits that fall off the end.
    """
    u = bytearray(16)
    u[0:6] = r[0:6]
    u[6] = ((r[6] & 0xF0) >> 4) | 0x80
    u[7] = ((r[6] & 0x0F) << 4) | ((r[7] & 0xF0) >> 4)
    u[8] = 0x80 | ((r[7] & 0x0F) << 2) | ((r[8] & 0xC0) >> 6)
    for i in range(9, 15):
        u[i] = ((r[i - 1] & 0x3F) << 2) | ((r[i] & 0xC0) >> 6)
    # Payload 120:121 - the Component Type's high two bits - not 126:127.
    u[15] = ((r[14] & 0x3F) << 2) | (r[15] & 0x03)
    return bytes(u)


def from_uuidv8(u):
    r = bytearray(16)
    r[0:6] = u[0:6]
    r[6] = ((u[6] & 0x0F) << 4) | ((u[7] & 0xF0) >> 4)
    r[7] = ((u[7] & 0x0F) << 4) | ((u[8] & 0x3C) >> 2)
    for i in range(8, 14):
        r[i] = ((u[i] & 0x03) << 6) | ((u[i + 1] & 0xFC) >> 2)
    r[14] = ((u[14] & 0x03) << 6) | ((u[15] & 0xFC) >> 2)
    r[15] = u[15] & 0x03
    return bytes(r)


def derive(key, raw_primary):
    """HMAC-SHA256(key, the 16 primary octets), folded into the derived layout.

    0:63 = hash[0:63]; 64:71 reserved; 72:116 = hash[64:108] (45 bits, not 46);
    117 = the auxiliary bit copied from the primary; 118:121 reserved.
    """
    d = hmac.new(key, raw_primary, hashlib.sha256).digest()
    out = bytearray(16)
    out[0:8] = d[0:8]
    out[8] = 0
    out[9:14] = d[8:13]
    out[14] = (d[13] & 0x1F) | (raw_primary[14] & 0x20)
    out[15] = 0
    return d, bytes(out)


def pack_aux_input(fmt, machine_id, routing_id, revision_id, device_id, vendor_id,
                   component_type):
    """The 256-bit auxiliary input structure, LSB-first into 32 octets.

    Format 0:15, Machine ID 16:143, Routing ID 144:175, Revision 176:183,
    Device 184:199, Vendor 200:215, Component Type 216:219, Reserved 220:255.
    The widths sum to exactly 256.
    """
    assert len(machine_id) == 16
    v = fmt & 0xFFFF
    v |= int.from_bytes(machine_id, "little") << 16
    v |= (routing_id & 0xFFFFFFFF) << 144
    v |= (revision_id & 0xFF) << 176
    v |= (device_id & 0xFFFF) << 184
    v |= (vendor_id & 0xFFFF) << 200
    v |= (component_type & 0xF) << 216
    assert v < (1 << 256)
    return bytes((v >> (8 * i)) & 0xFF for i in range(32))


def routing_id(segment, bus, device, function):
    return ((segment & 0xFFFF) << 16) | ((bus & 0xFF) << 8) | ((device & 0x1F) << 3) | (
        function & 0x7)


def aux_serial(structure):
    return int.from_bytes(hashlib.sha256(structure).digest()[:8], "little")


def uuid_str(u):
    h = u.hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ---------------------------------------------------------------------------
# The vectors.
# ---------------------------------------------------------------------------
# P-1 and P-2 are the two Radeon Pro W6800s the two layers were first verified
# against by hand. They are the anchor: if these move, something is wrong.
P1 = dict(serial=0x06C5349BD3AAABD4, unit_id=0, revision_id=0x00,
          device_id=0x73A3, vendor_id=0x1002, component_type=GPU)
P2 = dict(P1, serial=0x8E8C71777252EBFF)
U1 = dict(P1, unit_id=0x0123)

# A-1: machine ID and BDF chosen to be obviously synthetic.
AUX_MACHINE_ID = bytes.fromhex("0123456789abcdef0123456789abcdef")


def build():
    rows = []

    def primary(name, kw, aux=0):
        raw = pack_primary(aux=aux, **kw)
        u = to_uuidv8(raw)
        assert from_uuidv8(u) == raw, f"{name}: framing is lossy"
        assert (u[6] >> 4) == 8, f"{name}: version nibble is not 8"
        assert (u[8] >> 6) == 0b10, f"{name}: variant bits are not 10b"
        rows.append(("primary", name, raw.hex(), "", uuid_str(u)))
        return raw

    p1 = primary("P-1", P1)
    primary("P-2", P2)
    primary("U-1", U1)

    for name, t in [("T-PLATFORM", PLATFORM), ("T-CPU", CPU), ("T-GPU", GPU),
                    ("T-NIC", NIC), ("T-NPU", NPU), ("T-OTHER", OTHER)]:
        primary(name, dict(P1, component_type=t))

    for name, key in [("D-1", DEFAULT_SEED), ("D-2", TEST_SEED)]:
        digest, raw = derive(key, p1)
        u = to_uuidv8(raw)
        assert from_uuidv8(u) == raw
        rows.append(("derived", name, raw.hex(), digest.hex(), uuid_str(u)))

    st = pack_aux_input(AUX_FORMAT_PCIE, AUX_MACHINE_ID, routing_id(0, 0x63, 0, 0),
                        0x00, 0x73A3, 0x1002, GPU)
    rows.append(("aux-input", "A-1", st.hex(), hashlib.sha256(st).hexdigest(), ""))
    pa = pack_primary(aux_serial(st), 0, 0x00, 0x73A3, 0x1002, GPU, aux=1)
    assert (pa[14] >> 5) & 1, "A-1: auxiliary bit not set"
    rows.append(("primary", "A-1", pa.hex(), "", uuid_str(to_uuidv8(pa))))

    digest, da = derive(TEMP_KEY, pa)
    assert (da[14] >> 5) & 1, "A-2: auxiliary bit not carried into the derived value"
    rows.append(("derived", "A-2", da.hex(), digest.hex(), uuid_str(to_uuidv8(da))))

    return rows


def selfcheck():
    """Properties the vectors alone cannot pin down."""
    # The framing discards exactly payload bits 122:127.
    base = to_uuidv8(bytes(16))
    dropped = [i for i in range(128)
               if to_uuidv8(bytes(1 << (i & 7) if j == i >> 3 else 0
                                  for j in range(16))) == base]
    assert dropped == [122, 123, 124, 125, 126, 127], f"framing drops {dropped}"

    # Round-trip over arbitrary payloads.
    import random
    rng = random.Random(20260820)
    for _ in range(10000):
        p = rng.getrandbits(122)
        raw = bytes((p >> (8 * i)) & 0xFF for i in range(16))
        assert from_uuidv8(to_uuidv8(raw)) == raw

    # Every component type renders distinctly - an NPU must not collide with a
    # Platform, which is what happens when payload 120:121 are dropped.
    seen = {}
    for t in range(16):
        s = uuid_str(to_uuidv8(pack_primary(component_type=t,
                                            **{k: v for k, v in P1.items()
                                               if k != "component_type"})))
        assert s not in seen, f"component type {t} collides with {seen[s]}"
        seen[s] = t


HEADER = """# CUID conformance vectors - generated by cuid_vectors.py, do not edit.
# kind<TAB>name<TAB>payload-hex<TAB>hmac-hex<TAB>uuid
"""


def render(rows):
    # Trailing empty fields are dropped rather than emitted as a bare tab. An
    # aux-input row has no UUID and a primary row has no HMAC; the HMAC sits in
    # the middle so its empty field survives, but a trailing tab does not --
    # every whitespace-normalising tool in both trees strips it, and the file
    # would then differ from what the generator produces on the next run. Both
    # readers treat a missing trailing field as empty, so nothing is lost.
    return HEADER + "".join("\t".join(r).rstrip("\t") + "\n" for r in rows)


if __name__ == "__main__":
    selfcheck()
    text = render(build())
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cuid_vectors.txt")
    if "--check" in sys.argv:
        with open(out) as f:
            if f.read() != text:
                print(f"{out} is out of date; rerun {sys.argv[0]}", file=sys.stderr)
                sys.exit(1)
        print("vectors up to date")
    else:
        with open(out, "w") as f:
            f.write(text)
        print(text, end="")

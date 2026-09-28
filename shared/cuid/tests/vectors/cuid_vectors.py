#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CUID conformance vectors - the cross-layer format contract.

This is the single source of truth for the CUID wire format. Every producer -
the amdgpu kernel driver and the userspace CUID library - must reproduce
every vector below bit for bit. The vectors are normative in all their
columns: the 16-octet payload, the full HMAC digest where one is involved,
and the rendered UUID.

Matching the UUID but not the payload means an offsetting pair of errors, and
the next field that changes will diverge.

Run directly to self-check and to regenerate cuid_vectors.txt beside this
file; run with --check to verify it is up to date. Every HMAC digest here is
independently reproducible with:

    openssl dgst -sha256 -mac HMAC -macopt hexkey:<key-hex> <payload.bin>
"""

import hashlib
import hmac
import sys

# ---------------------------------------------------------------------------
# Constants. The kernel driver and the userspace library must agree with these
# byte for byte.
# ---------------------------------------------------------------------------
# D-1 and AD-2 derive from a fixed test key. There is no default key: every
# host generates or is given its own.
TEST_KEY_1 = bytes(0xA5 ^ n for n in range(32))
TEMP_LABEL = b"AMD-CUID-TEMP-v2"             # 16 octets, no NUL
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


def temp_key(machine_id):
    """K_app: the machine-id keys an application-specific HMAC key.

    machine-id(5) asks applications not to expose the raw value, so it keys
    HMAC-SHA256 over a fixed label and only the result is used.
    """
    assert len(machine_id) == 16
    return hmac.new(machine_id, TEMP_LABEL, hashlib.sha256).digest()


def pack_aux_input(fmt, routing_id, revision_id, device_id, vendor_id, component_type):
    """The 256-bit auxiliary input structure, LSB-first into 32 octets.

    Format 0:15, Machine ID 16:143 (zero: the machine-id keys the HMAC
    instead), Routing ID 144:175, Revision 176:183, Device 184:199,
    Vendor 200:215, Component Type 216:219, Reserved 220:255. The widths sum to
    exactly 256.
    """
    v = fmt & 0xFFFF
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


def aux_digest(k_app, structure):
    return hmac.new(k_app, structure, hashlib.sha256).digest()


def aux_serial(k_app, structure):
    return int.from_bytes(aux_digest(k_app, structure)[:8], "little")


def serial_from_octets(octets):
    """A serial read out of hardware: the octets as they lie, little-endian.

    The PCIe Device Serial Number capability and a NIC's MAC address are both
    read as a run of octets and become a 64-bit value with no swap. Stating it
    in a vector is the point: the orientation appears in no baseline
    requirement.
    """
    assert len(octets) <= 8
    return int.from_bytes(bytes(octets) + bytes(8 - len(octets)), "little")


def uuid_str(u):
    h = u.hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ---------------------------------------------------------------------------
# The vectors.
# ---------------------------------------------------------------------------
# P-1 and P-2 are the anchor: if these move, something is wrong.
P1 = dict(serial=0x06C5349BD3AAABD4, unit_id=0, revision_id=0x00,
          device_id=0x73A3, vendor_id=0x1002, component_type=GPU)
P2 = dict(P1, serial=0x8E8C71777252EBFF)
U1 = dict(P1, unit_id=0x0123)
# The top of the 13-bit UnitID field. Payload 112:116 all set, and bit 117,
# the Auxiliary Value Identifier, still clear. A producer that gave UnitID six
# bits instead of five would mark a maximal UnitID as auxiliary; nothing in the
# format flags it, so only a vector can.
U_MAX = dict(P1, unit_id=0x1FFF)

# A-1: machine ID and BDF chosen to be obviously synthetic.
AUX_MACHINE_ID = bytes.fromhex("0123456789abcdef0123456789abcdef")
AUX_KEY = temp_key(AUX_MACHINE_ID)


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
    u_max = primary("U-MAX", U_MAX)
    assert (u_max[14] >> 5) & 1 == 0, "U-MAX: a maximal UnitID set the auxiliary bit"
    assert u_max[14] & 0x1F == 0x1F, "U-MAX: payload 112:116 not all set"

    for name, t in [("T-PLATFORM", PLATFORM), ("T-CPU", CPU), ("T-GPU", GPU),
                    ("T-NIC", NIC), ("T-NPU", NPU), ("T-OTHER", OTHER)]:
        primary(name, dict(P1, component_type=t))

    for name, key in [("D-1", TEST_KEY_1), ("D-2", TEST_SEED)]:
        digest, raw = derive(key, p1)
        u = to_uuidv8(raw)
        assert from_uuidv8(u) == raw
        rows.append(("derived", name, raw.hex(), digest.hex(), uuid_str(u)))

    st = pack_aux_input(AUX_FORMAT_PCIE, routing_id(0, 0x63, 0, 0), 0x00, 0x73A3, 0x1002, GPU)
    assert st[2:18] == bytes(16), "A-1: Machine ID field is not zero"
    rows.append(("aux-input", "A-1", st.hex(), aux_digest(AUX_KEY, st).hex(), ""))
    pa = pack_primary(aux_serial(AUX_KEY, st), 0, 0x00, 0x73A3, 0x1002, GPU, aux=1)
    assert (pa[14] >> 5) & 1, "A-1: auxiliary bit not set"
    rows.append(("primary", "A-1", pa.hex(), "", uuid_str(to_uuidv8(pa))))

    digest, da = derive(AUX_KEY, pa)
    assert (da[14] >> 5) & 1, "A-2: auxiliary bit not carried into the derived value"
    rows.append(("derived", "A-2", da.hex(), digest.hex(), uuid_str(to_uuidv8(da))))

    # A-CPU: the CPU auxiliary structure. Two things are pinned here and
    # nowhere else: the CPU Format value, and that the Routing ID is
    # the physical package ID. Every socket has UnitID 0, so without it two
    # sockets of one host would share a temporary CUID. This vector is socket
    # 0, whose Routing ID is zero.
    st_cpu = pack_aux_input(AUX_FORMAT_CPU, 0, 0x02, 0x1480, 0x1022, CPU)
    assert st_cpu[18:22] == b"\x00\x00\x00\x00", "A-CPU: socket 0 Routing ID is not zero"
    rows.append(("aux-input", "A-CPU", st_cpu.hex(), aux_digest(AUX_KEY, st_cpu).hex(), ""))
    pc = pack_primary(aux_serial(AUX_KEY, st_cpu), 0, 0x02, 0x1480, 0x1022, CPU, aux=1)
    assert (pc[14] >> 5) & 1, "A-CPU: auxiliary bit not set"
    rows.append(("primary", "A-CPU", pc.hex(), "", uuid_str(to_uuidv8(pc))))

    # A-NIC: an auxiliary component that is not a GPU and not a CPU, so that the
    # Component Type travels through the auxiliary structure as well as through
    # the primary. Both fields are four bits in different places; a producer
    # that packs one correctly can still drop the other.
    st_nic = pack_aux_input(AUX_FORMAT_PCIE, routing_id(0, 0x41, 0, 1),
                            0x01, 0x1458, 0x14E4, NIC)
    rows.append(("aux-input", "A-NIC", st_nic.hex(), aux_digest(AUX_KEY, st_nic).hex(), ""))
    pn = pack_primary(aux_serial(AUX_KEY, st_nic), 0, 0x01, 0x1458, 0x14E4, NIC, aux=1)
    assert (pn[14] >> 5) & 1, "A-NIC: auxiliary bit not set"
    assert (pn[14] >> 6) | (pn[15] << 2) == NIC, "A-NIC: component type lost"
    rows.append(("primary", "A-NIC", pn.hex(), "", uuid_str(to_uuidv8(pn))))

    # An auxiliary CPU and an auxiliary NIC must not collide with each other or
    # with the auxiliary GPU: K_app is the same in all three, so the only
    # things separating them are the fields this file exists to pin.
    assert len({pa.hex(), pc.hex(), pn.hex()}) == 3, "auxiliary components collide"

    # S-DSN: the eight octets of a PCIe Device Serial Number capability, as they
    # appear in configuration space from dsn_cap_offset + 4, and the 64-bit
    # serial they become. No swap. A byte-swapped reader still produces a
    # well-formed payload; it just names a different card.
    dsn_octets = bytes.fromhex("d4abaad39b34c506")
    dsn_serial = serial_from_octets(dsn_octets)
    assert dsn_serial == P1["serial"], "S-DSN does not yield the P-1 serial"
    rows.append(("serial", "S-DSN", dsn_octets.hex(), "%016x" % dsn_serial, ""))

    # S-MAC: a NIC with no DSN and no vendor-specific capability falls back to
    # its MAC address. Six octets, octet 0 at payload bits 0:7, zero-extended:
    # the same orientation as the DSN.
    mac_octets = bytes.fromhex("b42e99a1c30f")
    mac_serial = serial_from_octets(mac_octets)
    rows.append(("serial", "S-MAC", mac_octets.hex(), "%016x" % mac_serial, ""))
    # An all-zero MAC is an unconfigured interface, not an identity: every such
    # NIC on every machine would otherwise share one CUID.
    rows.append(("serial", "S-MAC-ZERO", "000000000000", "%016x" % 0, ""))

    # AD-1: derivation from an *adopted* primary -- a firmware UUID carried
    # verbatim rather than a packed payload. The HMAC message is the sixteen
    # octets as firmware wrote them, version and variant bits included: framing
    # them again would drop six bits and shift the rest, so two platforms
    # differing only in those bits would yield the same derived CUID. The
    # auxiliary bit is not copied across either, because in an adopted primary
    # that position is opaque firmware data and means nothing.
    adopted = bytes.fromhex("6ba7b8109dad11d180b400c04fd430c8")  # v1, variant 10b
    assert (adopted[6] >> 4) != 8, "AD-1: the adopted UUID must not be a v8"
    rows.append(("adopted", "AD-1", adopted.hex(), "", uuid_str(adopted)))
    digest_ad = hmac.new(TEST_KEY_1, adopted, hashlib.sha256).digest()
    out = bytearray(16)
    out[0:8] = digest_ad[0:8]
    out[9:14] = digest_ad[8:13]
    out[14] = digest_ad[13] & 0x1F   # no auxiliary bit: the primary has no such bit
    rows.append(("derived", "AD-2", bytes(out).hex(), digest_ad.hex(),
                 uuid_str(to_uuidv8(bytes(out)))))

    # UnitID = (XCC count << 6) | first logical XCC, over an 8-XCC part in
    # every mode: SPX (1 partition), DPX (2), QPX (4), CPX (8). The column
    # normally holding an HMAC instead holds the 13-bit UnitID as 4 hex
    # digits; there is no UUID, because a UnitID is not a CUID by itself.
    def unit_id(name, mask):
        shifted = mask >> (mask & -mask).bit_length() - 1
        assert not (shifted & (shifted + 1)), f"{name}: mask {mask:#x} is not contiguous"
        first = (mask & -mask).bit_length() - 1
        uid = (bin(mask).count("1") << 6) | first
        rows.append(("unit-id", name, "%08x" % mask, "%04x" % uid, ""))
        return uid

    assert unit_id("U8-SPX", 0xFF) == 0x200
    assert unit_id("U8-DPX-0", 0x0F) == 0x100
    assert unit_id("U8-DPX-1", 0xF0) == 0x104
    assert unit_id("U8-QPX-0", 0x03) == 0x080
    assert unit_id("U8-QPX-1", 0x0C) == 0x082
    assert unit_id("U8-QPX-2", 0x30) == 0x084
    assert unit_id("U8-QPX-3", 0xC0) == 0x086
    for i in range(8):
        assert unit_id(f"U8-CPX-{i}", 1 << i) == 0x040 + i

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
#
# kind      col 3                     col 4                  col 5
# primary   the 16 packed octets      -                      rendered UUID
# derived   the 16 derived octets     the full HMAC digest   rendered UUID
# aux-input the 32-octet structure,   its HMAC-SHA256        -
#           Machine ID field zero     under K_app
# serial    the octets as hardware    the 64-bit serial      -
#           presents them             they become, 16 hex
# adopted   a firmware UUID carried   -                      the same 16 octets
#           verbatim, 16 octets                              rendered
# unit-id   the 32-bit logical XCC    the 13-bit UnitID,      -
#           mask, 8 hex digits        4 hex digits
#
# K_app = HMAC-SHA256(key = machine-id 0123456789abcdef0123456789abcdef as 16
# octets, msg = "AMD-CUID-TEMP-v2"). A-1, A-CPU and A-NIC take their serial from
# the first 8 octets of the aux-input HMAC; A-2 is A-1 derived under K_app.
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

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
import uuid

# ---------------------------------------------------------------------------
# Constants. The kernel driver and the userspace library must agree with these
# byte for byte.
# ---------------------------------------------------------------------------
# D-1 and AD-2 derive from a fixed test key. There is no default key: every
# host generates or is given its own.
TEST_KEY_1 = bytes(0xA5 ^ n for n in range(32))
APP_ID = uuid.uuid5(uuid.uuid5(uuid.NAMESPACE_DNS, "amd.com"), "com.amd.cuid.v1").bytes
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


def pack_temp_input(fmt, machine_id, routing_id, revision_id, device_id, vendor_id,
                    component_type, unit_id=0, segment_high=0):
    """S, the temporary CUID's 256-bit input structure, LSB-first into 32 octets.

    Format 0:15, Machine ID 16:143 (the 16 machine-id octets in file order),
    Routing ID 144:175, Revision 176:183, Device 184:199, Vendor 200:215,
    Component Type 216:219, Reserved 220:223, UnitID 224:239 (13 bits),
    Segment high 240:255 (segment >> 16). The widths sum to exactly 256.
    """
    assert len(machine_id) == 16
    v = fmt & 0xFFFF
    v |= int.from_bytes(machine_id, "little") << 16
    v |= (routing_id & 0xFFFFFFFF) << 144
    v |= (revision_id & 0xFF) << 176
    v |= (device_id & 0xFFFF) << 184
    v |= (vendor_id & 0xFFFF) << 200
    v |= (component_type & 0xF) << 216
    v |= (unit_id & 0x1FFF) << 224
    v |= (segment_high & 0xFFFF) << 240
    assert v < (1 << 256)
    return bytes((v >> (8 * i)) & 0xFF for i in range(32))


def temporary(structure):
    """HMAC-SHA256(key = S, msg = APP_ID), its first 16 octets with version 8,
    variant 10b and payload bit 117 (rendered octet 15 bit 7) set."""
    d = hmac.new(structure, APP_ID, hashlib.sha256).digest()
    b = bytearray(d[:16])
    b[6] = (b[6] & 0x0F) | 0x80
    b[8] = (b[8] & 0x3F) | 0x80
    b[15] |= 0x80
    return d, bytes(b)


def routing_id(segment, bus, device, function):
    return ((segment & 0xFFFF) << 16) | ((bus & 0xFF) << 8) | ((device & 0x1F) << 3) | (
        function & 0x7)


def serial_from_octets(octets):
    """A serial read out of hardware: the octets as they lie, little-endian.

    The PCIe Device Serial Number capability and a NIC's MAC address are both
    read as a run of octets and become a 64-bit value with no swap. Stating it
    in a vector is the point: the orientation appears in no baseline
    requirement.
    """
    assert len(octets) <= 8
    return int.from_bytes(bytes(octets) + bytes(8 - len(octets)), "little")


def partition_unit_id(n, k):
    """UnitID of partition k of n: 0 for the whole device (n = 1)."""
    assert 0 <= k < n
    return 0 if n == 1 else n * (n - 1) // 2 + 1 + k


def unit_from_id(u):
    """The (n, k) a UnitID names; (1, 0) for the whole device."""
    if u == 0:
        return 1, 0
    assert u >= 2, "UnitID 1 is a whole device in SLC mode"
    n = 2
    while (n + 1) * n // 2 + 1 <= u:
        n += 1
    return n, u - n * (n - 1) // 2 - 1


def uuid_str(u):
    h = u.hex()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


# ---------------------------------------------------------------------------
# The vectors.
# ---------------------------------------------------------------------------
# P-1 and P-2 are the anchor: if these move, something is wrong.
P1 = dict(serial=0xFEDCBA9876543210, unit_id=0, revision_id=0x00,
          device_id=0x73A3, vendor_id=0x1002, component_type=GPU)
P2 = dict(P1, serial=0x0123456789ABCDEF)
U1 = dict(P1, unit_id=0x0123)
# The top of the 13-bit UnitID field. Payload 112:116 all set, and bit 117,
# the Auxiliary Value Identifier, still clear. A producer that gave UnitID six
# bits instead of five would mark a maximal UnitID as auxiliary; nothing in the
# format flags it, so only a vector can.
U_MAX = dict(P1, unit_id=0x1FFF)

# TV-*: machine ID and BDFs chosen to be obviously synthetic.
TEMP_MACHINE_ID = bytes.fromhex("0123456789abcdef0123456789abcdef")


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

    def temp(name, fmt, routing, rev, dev, ven, typ, unit=0, seg_hi=0):
        st = pack_temp_input(fmt, TEMP_MACHINE_ID, routing, rev, dev, ven, typ, unit, seg_hi)
        assert st[2:18] == TEMP_MACHINE_ID, f"{name}: Machine ID not at octets 2:17"
        digest, u = temporary(st)
        r = from_uuidv8(u)
        assert (u[6] >> 4) == 8 and (u[8] >> 6) == 0b10, f"{name}: not an RFC 9562 v8"
        assert (r[14] >> 5) & 1, f"{name}: payload bit 117 not set"
        rows.append(("temporary", name, st.hex(), digest.hex(), uuid_str(u)))
        return st, u

    # TV-GPU: a whole GPU. TV-PART1/2 are its two partitions in DPX, UnitIDs 2
    # and 3: they share the BDF, so only the UnitID at 224:239 keeps them apart
    # from it and each other.
    gpu = routing_id(0, 0x63, 0, 0)
    st_gpu, tv_gpu = temp("TV-GPU", AUX_FORMAT_PCIE, gpu, 0x00, 0x73A3, 0x1002, GPU)
    st_p1, tv_p1 = temp("TV-PART1", AUX_FORMAT_PCIE, gpu, 0x00, 0x73A3, 0x1002, GPU,
                        partition_unit_id(2, 0))
    st_p2, tv_p2 = temp("TV-PART2", AUX_FORMAT_PCIE, gpu, 0x00, 0x73A3, 0x1002, GPU,
                        partition_unit_id(2, 1))
    assert st_p2[28:30] == b"\x03\x00", "TV-PART2: UnitID not at octets 28:29"

    # TV-VMD: the TV-GPU part behind VMD domain 10000. The Routing ID keeps the
    # segment's low 16 bits, which are those of 0000, so only the segment's high
    # half at 240:255 keeps it apart from TV-GPU.
    st_vmd, tv_vmd = temp("TV-VMD", AUX_FORMAT_PCIE, routing_id(0x10000, 0x63, 0, 0), 0x00,
                          0x73A3, 0x1002, GPU, 0, 0x10000 >> 16)
    assert st_vmd[:30] == st_gpu[:30] and st_vmd[30:32] == b"\x01\x00", "TV-VMD: segment high"

    # TV-NIC: PCI function 1, which is its UnitID; the Component Type travels
    # through S at 216:219.
    _, tv_nic = temp("TV-NIC", AUX_FORMAT_PCIE, routing_id(0, 0x41, 0, 1), 0x01, 0x1658, 0x14E4,
                     NIC, 1)

    # TV-CPU: socket 0, whose Routing ID is its physical package ID, zero.
    # Family 0x19 model 0x11 stepping 1: DeviceID 0x1911, RevisionID 0x01.
    st_cpu, tv_cpu = temp("TV-CPU", AUX_FORMAT_CPU, 0, 0x01, 0x1911, 0x1022, CPU)
    assert st_cpu[18:22] == bytes(4), "TV-CPU: socket 0 Routing ID is not zero"

    # TV-PLAT: the platform takes the CPU Format with every field zero but the
    # Component Type, which keeps it apart from socket 0.
    _, tv_plat = temp("TV-PLAT", AUX_FORMAT_CPU, 0, 0x00, 0x0000, 0x0000, PLATFORM)

    assert len({tv_gpu, tv_p1, tv_p2, tv_vmd, tv_nic, tv_cpu, tv_plat}) == 7, \
        "temporary components collide"
    other = bytearray(TEMP_MACHINE_ID)
    other[15] ^= 1
    assert temporary(pack_temp_input(AUX_FORMAT_PCIE, bytes(other), gpu, 0x00, 0x73A3, 0x1002,
                                     GPU))[1] != tv_gpu, "machine-id does not reach the CUID"

    # S-DSN: the eight octets of a PCIe Device Serial Number capability, as they
    # appear in configuration space from dsn_cap_offset + 4, and the 64-bit
    # serial they become. No swap. A byte-swapped reader still produces a
    # well-formed payload; it just names a different card.
    dsn_octets = bytes.fromhex("1032547698badcfe")
    dsn_serial = serial_from_octets(dsn_octets)
    assert dsn_serial == P1["serial"], "S-DSN does not yield the P-1 serial"
    rows.append(("serial", "S-DSN", dsn_octets.hex(), "%016x" % dsn_serial, ""))

    # S-MAC: a NIC with no DSN and no vendor-specific capability falls back to
    # its MAC address. Six octets, octet 0 at payload bits 0:7, zero-extended:
    # the same orientation as the DSN.
    mac_octets = bytes.fromhex("021122334455")
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

    # Partition k of a GPU split into n partitions has UnitID n*(n-1)/2 + 1 + k;
    # n = 1 is the whole device, UnitID 0. The payload column holds n and k as
    # one octet each and the HMAC column the 13-bit UnitID as 4 hex digits;
    # there is no UUID, because a UnitID is not a CUID by itself.
    def unit_id(name, n, k):
        uid = partition_unit_id(n, k)
        assert unit_from_id(uid) == (n, k), f"{name}: {uid:#x} does not invert"
        rows.append(("unit-id", name, "%02x%02x" % (n, k), "%04x" % uid, ""))
        return uid

    assert unit_id("U-WHOLE", 1, 0) == 0
    assert [unit_id(f"U-DPX-{k}", 2, k) for k in range(2)] == [2, 3]
    assert [unit_id(f"U-TPX-{k}", 3, k) for k in range(3)] == [4, 5, 6]
    assert [unit_id(f"U-QPX-{k}", 4, k) for k in range(4)] == [7, 8, 9, 10]
    assert [unit_id(f"U-CPX6-{k}", 6, k) for k in range(6)] == list(range(16, 22))
    assert [unit_id(f"U-CPX8-{k}", 8, k) for k in range(8)] == list(range(29, 37))
    assert unit_id("U-N127-LAST", 127, 126) <= 0x1FFF
    assert partition_unit_id(128, 127) > 0x1FFF, "127 is not the largest n that fits"

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
# temporary the 32-octet structure S   HMAC-SHA256(S, app id) rendered temporary CUID
# serial    the octets as hardware    the 64-bit serial      -
#           presents them             they become, 16 hex
# adopted   a firmware UUID carried   -                      the same 16 octets
#           verbatim, 16 octets                              rendered
# unit-id   n and k, one octet each:  the 13-bit UnitID,      -
#           partition k of n          4 hex digits
#
# TV-*: S holds machine-id 0123456789abcdef0123456789abcdef at octets 2:17; the
# HMAC message is app id deeda863-4023-5bac-865a-9923e61370d7 as 16 octets. The
# CUID is the first 16 digest octets with version 8, variant 10b and payload bit
# 117 (rendered octet 15 bit 7) set.
"""


def render(rows):
    # Trailing empty fields are dropped rather than emitted as a bare tab. An
    # serial row has no UUID and a primary row has no HMAC; the HMAC sits in
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

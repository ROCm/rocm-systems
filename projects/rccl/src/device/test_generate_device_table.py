#!/usr/bin/env python3
"""Unit tests for src/device/generate.py device_table.h dispatch generation.

These guard the -fgpu-rdc (non device-linker) code path introduced to avoid
taking the address of any ncclDevFunc_* in a function-pointer table (issue
#8129). The generator emits, from a single header:

  * the function-pointer table (ncclDevFuncTable_*), used ONLY when
    RCCL_DEVICE_LINKER (or the legacy USE_INDIRECT_FUNCTION_CALL) is defined,
    and declared `static` so unused copies are dead-stripped; and
  * a compile-time templated binary-search dispatcher (Caller* /
    NCCL_CALL_FUNCTIONS_*) for the pure-RDC build, whose leaves call each
    ncclDevFunc_* directly by name (nothing address-taken).

The header is mode-agnostic (generated once); which arm is active is decided at
compile time by the macros. So these tests assert the *structure/gating* of the
generated text rather than compiling it.
"""

import os
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
GENERATE_PY = os.path.join(HERE, "generate.py")

# A small, fast slice of collectives. "AllReduce RING SIMPLE Sum f32" expands to
# both an unguarded primary and an arch-guarded variant, which exercises the
# guarded-out (trap) leaf below. The AllReduce LL128 entries are reg-variant
# (see ll128_reg_variant_colls), so they carry a "_1"/"_2" reg suffix while every
# other kernel omits the reg field -- the pure-RDC dispatcher must call each by
# its exact declared name (regression: #ll128-reg-split).
#
# SendRecv is emitted as TWO latency-protocol kernel variants (reg_values_of):
#   reg=0 -> legacy LL send/recv kernel, built unguarded on every arch (the default)
#   reg=1 -> LL128 send/recv kernel, arch-guarded to gfx942/gfx950/gfx1250 + ENABLE_LL128.
#            Emitted for every unroll: a hole would shift that table's later indices.
ONLY_FUNCS = "AllReduce RING SIMPLE Sum f32|AllReduce RING LL128 Sum f32|SendRecv"


def _read_generated(tmpdir, name):
    with open(os.path.join(tmpdir, name)) as f:
        return f.read()


def _generate(tmpdir, ifc="OFF", all_unrolls="OFF"):
    """Run generate.py into tmpdir and return the device_table.h contents."""
    # argv: gensrc, IFC, (unused), local_gpu_only, rocshmem, all_unrolls, ONLY_FUNCS
    # local_gpu_only=OFF avoids needing rocminfo/a local GPU.
    subprocess.run(
        [sys.executable, GENERATE_PY, tmpdir, ifc, "OFF", "OFF", "OFF", all_unrolls, ONLY_FUNCS],
        check=True,
        capture_output=True,
        text=True,
    )
    return _read_generated(tmpdir, "device_table.h")


def _unroll_tables(header):
    """Map generated unroll -> table entries, guarded slots reduced to their symbol.

    A table is emitted for every NCCL_UNROLL_* enum value, so the empty ones are
    dropped here: they were not generated for this build and no caller wants them.
    """
    tables = {}
    for unroll, body in re.findall(
        r"ncclDevFuncTable_(\d+)\[\] = \{(.*?)nullptr\};", header, re.S
    ):
        slots = {}
        # Guarded slots emit symbol + "#else" nullptr at the same index; keep the symbol.
        for m in re.finditer(r"/\*\s*(\d+)\*/ (\w+),", body):
            slots.setdefault(int(m.group(1)), m.group(2))
        if slots:
            tables[unroll] = [slots[i] for i in sorted(slots)]
    return tables


def _strip_unroll(sym):
    # ncclDevFunc_<coll>_<algo>_<proto>_<redop>_<ty>_<acc>_<pipeline>_<unroll>[_<reg>]
    parts = sym.split("_")
    del parts[8]
    return "_".join(parts)


class DeviceTableGenerationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(GENERATE_PY):
            raise unittest.SkipTest("generate.py not found next to test")
        cls._dir = tempfile.mkdtemp(prefix="rccl_devtable_")
        cls.header = _generate(cls._dir)
        cls.host_table = _read_generated(cls._dir, "host_table.cpp")

    def test_forward_declarations_are_plain(self):
        # noinline is applied only by DEFINE_ncclDevFunc (common.h), gated on
        # RCCL_DEVICE_LINKER. The generated forward declarations must stay plain:
        # a stray noinline here leaks into the definition (attribute is a union
        # across decl+def) and would force pure-RDC funcs noinline.
        self.assertIn("__device__ void ncclDevFunc_", self.header)
        self.assertNotIn("noinline", self.header)
        self.assertNotIn("RCCL_DEVFUNC_ATTR", self.header)

    def test_table_is_static_and_runtime_dispatch_gated(self):
        # Table only for the runtime-dispatch builds, and internal linkage.
        self.assertIn(
            "#if defined(USE_INDIRECT_FUNCTION_CALL) || defined(RCCL_DEVICE_LINKER)",
            self.header,
        )
        self.assertIn("static __device__ ncclDevFuncPtr_t const ncclDevFuncTable_", self.header)

    def test_pure_rdc_dispatch_block_present(self):
        # Compile-time binary search only when NEITHER runtime-dispatch macro is set.
        self.assertIn(
            "#if !defined(USE_INDIRECT_FUNCTION_CALL) && !defined(RCCL_DEVICE_LINKER)",
            self.header,
        )
        self.assertIn("NCCL_CALL_FUNCTIONS_", self.header)
        # One explicit leaf specialization per index, dispatched by name.
        self.assertIn("struct Caller1<0, 1>", self.header)
        self.assertRegex(self.header, r"Caller1<0, \d+>::call1")

    def test_pure_rdc_dispatch_calls_only_declared_symbols(self):
        # Every ncclDevFunc_* called by name in a pure-RDC Caller leaf must be
        # one of the forward-declared symbols. The reg-variant split makes the
        # symbol name conditional (reg suffix only when reg != 0), so a leaf that
        # reconstructs the name from all fields (appending a stray "_0") would
        # reference an undeclared symbol and fail the -fgpu-rdc / --no-device-linker
        # link. This asserts the two symbol sets agree.
        declared = set(re.findall(r"__device__ void (ncclDevFunc_\w+)\(\);", self.header))
        self.assertTrue(declared, "no forward declarations found")
        called = set(
            re.findall(r"noexcept \{ (ncclDevFunc_\w+)\(\); \}", self.header)
        )
        self.assertTrue(called, "no pure-RDC dispatch leaves found")
        undeclared = called - declared
        self.assertEqual(
            set(),
            undeclared,
            "pure-RDC dispatch calls symbols that were never declared: %s" % sorted(undeclared),
        )
        # Sanity: no kernel ever gets a bogus "_0" reg suffix.
        self.assertFalse(any(re.search(r"_LL128_.*_0$", s) for s in called))
        # Sanity: AllReduce LL128 must appear as a reg-variant PAIR -- a reg=1 and
        # a reg=2 symbol. Match the reg field specifically: a bare endswith("_1"/"_2")
        # is not enough because reg=0 kernels omit the reg field and end in the unroll
        # value (which is also 1/2), so that check passes even if the split were removed.
        self.assertTrue(
            any(re.search(r"_AllReduce_RING_LL128_Sum_f32_\d+_\d+_\d+_1$", s) for s in called),
            "registered (reg=1) AllReduce LL128 symbol missing",
        )
        self.assertTrue(
            any(re.search(r"_AllReduce_RING_LL128_Sum_f32_\d+_\d+_\d+_2$", s) for s in called),
            "non-registered (reg=2) AllReduce LL128 symbol missing",
        )

    def test_guarded_out_leaf_traps_not_noop(self):
        # Arch-guarded-out slots must fail fast (matching the old nullptr table
        # entries), not silently no-op.
        self.assertIn("__builtin_trap();", self.header)

    # ---- SendRecv LL / LL128 reg-variant codegen (ll128-p2p-send-recv) --------
    # These pin the two-kernel split so a regression in reg_values_of /
    # get_arch_guard / func_validate for SendRecv fails here instead of only at
    # device link or at runtime.

    def _sendrecv_decls(self):
        # All forward-declared SendRecv device-function symbols.
        return set(
            re.findall(r"__device__ void (ncclDevFunc_SendRecv\w*)\(\);", self.header)
        )

    def test_sendrecv_emits_ll_and_ll128_reg_variants(self):
        # Both kernels must be generated: the legacy LL kernel (reg=0, no reg
        # suffix) and the LL128 kernel (reg=1, trailing "_1"). If reg_values_of
        # regressed to ["0"] only the LL kernel would exist.
        decls = self._sendrecv_decls()
        self.assertTrue(decls, "no SendRecv forward declarations generated")
        ll = [s for s in decls if re.fullmatch(r"ncclDevFunc_SendRecv_\w+?_\d+_\d+_\d+", s)]
        ll128 = [s for s in decls if re.fullmatch(r"ncclDevFunc_SendRecv_\w+?_\d+_\d+_\d+_1", s)]
        self.assertTrue(ll, "legacy LL SendRecv kernel (reg=0) missing")
        self.assertTrue(ll128, "LL128 SendRecv kernel (reg=1, '_1' suffix) missing")

    def test_sendrecv_ll128_is_arch_guarded_and_ll_is_not(self):
        # Every reg=1 (LL128) SendRecv declaration must sit inside the arch guard...
        guarded = re.findall(
            r"#if \(defined\(__gfx942__\) \|\| defined\(__gfx950__\) \|\| defined\(__gfx1250__\)\)"
            r" && defined\(ENABLE_LL128\)\n"
            r"__device__ void (ncclDevFunc_SendRecv\w*_1)\(\);\n#endif",
            self.header,
        )
        self.assertTrue(guarded, "LL128 SendRecv (reg=1) declaration is not arch-guarded")
        # ...and every emitted reg=1 SendRecv symbol is one of those guarded ones
        # (none leaks out unguarded onto the default-built archs). A reg=1 symbol
        # has FOUR trailing numeric fields (acc, pipeline, unroll, reg); a bare
        # endswith("_1") would also catch the reg=0 unroll=1 kernel (..._0_0_1).
        ll128 = {
            s
            for s in self._sendrecv_decls()
            if re.fullmatch(r"ncclDevFunc_SendRecv_\w+?_\d+_\d+_\d+_1", s)
        }
        self.assertEqual(
            set(),
            ll128 - set(guarded),
            "LL128 SendRecv kernels emitted without the arch guard: %s"
            % sorted(ll128 - set(guarded)),
        )
        # The legacy LL kernel must stay unguarded (built on every arch): its
        # bare declaration line has no surrounding #if.
        self.assertRegex(
            self.header,
            r"\n__device__ void ncclDevFunc_SendRecv_\w+?_\d+_\d+_\d+\(\);\n",
            "legacy LL SendRecv (reg=0) declaration should be unguarded",
        )

    def test_sendrecv_ll128_covers_every_generated_unroll(self):
        # Skipping reg=1 on the gfx1250 unrolls leaves a hole that shifts every later
        # index in that table away from the host ids (derived from the first unroll).
        decls = self._sendrecv_decls()
        generated = set(_unroll_tables(self.header))
        missing = sorted(
            u
            for u in generated
            if not any(re.fullmatch(r"ncclDevFunc_SendRecv_\w+?_\d+_\d+_%s_1" % u, s) for s in decls)
        )
        self.assertEqual(
            [], missing, "LL128 SendRecv (reg=1) missing for generated unrolls: %s" % missing
        )

    def test_device_tables_are_index_aligned_across_unrolls(self):
        # Host ids come from the first generated unroll but index every
        # ncclDevFuncTable_*, so all tables must hold the same functions in order.
        tables = _unroll_tables(self.header)
        self.assertTrue(tables, "no unroll tables generated")
        base_unroll, base = sorted(tables.items())[0]
        base_shape = [_strip_unroll(s) for s in base]
        for unroll, entries in sorted(tables.items()):
            self.assertEqual(
                base_shape,
                [_strip_unroll(s) for s in entries],
                "ncclDevFuncTable_%s is not index-aligned with ncclDevFuncTable_%s"
                % (unroll, base_unroll),
            )

    def test_all_unrolls_opt_in_generates_every_unroll(self):
        # BUILD_ALL_UNROLLS fills in the skipped unrolls and drops the gfx1250 restriction.
        with tempfile.TemporaryDirectory(prefix="rccl_devtable_all_") as tmpdir:
            header = _generate(tmpdir, all_unrolls="ON")
            host = _read_generated(tmpdir, "host_table.cpp")
        self.assertEqual({"1", "2", "4", "8", "16", "32"}, set(_unroll_tables(header)))
        # Both host tables flip under the flag, and no other test reads them on this path.
        generated = self._unroll_table("ncclDevFuncUnrollGenerated", r"true|false", host)
        self.assertEqual(
            ["true"] * 6,
            list(generated.values()),
            "BUILD_ALL_UNROLLS must mark every unroll generated",
        )
        pinned = self._unroll_table("ncclDevFuncUnrollArch", r'nullptr|"gfx\w+"', host)
        self.assertEqual(
            ["nullptr"] * 6,
            list(pinned.values()),
            "BUILD_ALL_UNROLLS compiles every unroll for the target, so none stays pinned",
        )
        self.assertNotIn("#if defined(__gfx1250__)\n", header)

    # ---- unroll arch restriction (host/device agreement) ---------------------
    # commSetUnrollFactor rejects an RCCL_UNROLL_FACTOR whose device functions were
    # not compiled for the running GPU, using ncclDevFuncUnrollArch[] emitted into
    # host_table.cpp. That table is a claim about device_table.h, and nothing else
    # checks the two agree: if it silently went all-nullptr the runtime check would
    # degrade to the arch-blind behaviour that dispatched into an empty table and
    # trapped. These tests hold host and device sides in lockstep.

    _ARCH_MACRO = re.compile(r"__(gfx\w+)__")

    def _unroll_table(self, name, value_pattern, host_table=None):
        """Parse a `<name>[NCCL_NUM_UNROLLS]` initializer into {unroll: raw value}."""
        block = re.search(
            r"%s\[NCCL_NUM_UNROLLS\] = \{\n(.*?)\n\};" % re.escape(name),
            self.host_table if host_table is None else host_table,
            re.S,
        )
        self.assertIsNotNone(block, "%s[] not emitted into host_table.cpp" % name)
        entries = re.findall(
            r"^\s*(%s), // unroll (\d+)$" % value_pattern, block.group(1), re.M
        )
        self.assertTrue(entries, "no %s[] entries parsed" % name)
        # The host indexes these tables by the NCCL_UNROLL_* ordinal, not by the
        # trailing comment: unrollAvailability() subscripts them with comm->unroll,
        # where NCCL_UNROLL_1 is 0 and each step doubles the factor. Keying the dict
        # off the comment below would hide a reordered all_unrolls, which emits rows
        # that read correctly but sit at the wrong ordinals.
        order = [unroll for _, unroll in entries]
        self.assertEqual(
            [str(2 ** i) for i in range(len(order))],
            order,
            "%s[] rows are not in NCCL_UNROLL_* order (1, 2, 4, ...), so the "
            "table is misaligned against the enum the host subscripts it with"
            % name,
        )
        return {unroll: value for value, unroll in entries}

    def _entry_guards(self, unroll):
        """Enclosing #if condition of each real slot in ncclDevFuncTable_<unroll>[].

        None for a slot that is not guarded, [] for a table with no slots (an unroll
        this build did not generate), None if the table is absent entirely.
        """
        # Every table closes with a trailing "nullptr};" sentinel, and that is the
        # only place that string appears -- guarded-out slots read "nullptr,". Stopping
        # there is what keeps the match inside this table. Two anchors that look right
        # are not: "\n};" ends nowhere in a table and runs on to the struct Caller
        # close, and requiring a newline before the sentinel skips past an *empty*
        # table (whose body is just "= {\nnullptr};") into the next one's slots.
        block = re.search(
            r"ncclDevFuncTable_%s\[\] = \{\n(.*?)nullptr\};" % unroll,
            self.header,
            re.S,
        )
        if block is None:
            return None
        guards, current, in_else = [], None, False
        for line in block.group(1).splitlines():
            if line.startswith("#if"):
                current, in_else = line[len("#if"):].strip(), False
            elif line.startswith("#else"):
                in_else = True
            elif line.startswith("#endif"):
                current, in_else = None, False
            elif "ncclDevFunc_" in line and not in_else:
                guards.append(current)
        return guards

    def _restricted_arch(self, unroll):
        """The lone arch that can run every slot of ncclDevFuncTable_<unroll>[], else None.

        The archs that can run EVERY slot, i.e. the intersection of the per-slot arch
        sets, with an unguarded slot counting as all archs. Per-slot rather than
        requiring each slot to name one lone arch: a slot may legitimately be built
        more widely than the unroll is dispatched (the LL128 SendRecv kernel is), and
        that does not make the unroll usable on the extra archs, because the other
        slots are still nullptr there. The base unrolls come out unrestricted, their
        intersection being several archs wide.
        """
        guards = self._entry_guards(unroll)
        if not guards:
            return None
        archs = None
        for guard in guards:
            if guard is None:
                continue  # unguarded: every arch, so it narrows nothing
            slot = set(self._ARCH_MACRO.findall(guard))
            archs = slot if archs is None else archs & slot
        return archs.pop() if archs is not None and len(archs) == 1 else None

    def test_unroll_arch_matches_device_table_guards(self):
        arch_table = self._unroll_table(
            "ncclDevFuncUnrollArch", r'nullptr|"gfx\w+"'
        )
        checked = 0
        for unroll, declared in arch_table.items():
            # No slots means this build did not generate the unroll, and the arch
            # table deliberately still names its arch (it never consults
            # local_unroll), so there is nothing here to cross-check against.
            if not self._entry_guards(unroll):
                continue
            restricted = self._restricted_arch(unroll)
            expected = '"%s"' % restricted if restricted else "nullptr"
            self.assertEqual(
                expected,
                declared,
                "ncclDevFuncUnrollArch[unroll %s] is %s but ncclDevFuncTable_%s[] is %s"
                % (
                    unroll,
                    declared,
                    unroll,
                    "compiled for %s only" % restricted
                    if restricted
                    else "built for every arch",
                ),
            )
            checked += 1
        self.assertTrue(checked, "no unroll tables were cross-checked")

    def test_unroll_arch_flags_the_single_arch_unrolls(self):
        # The cross-check above passes trivially if generate.py ever stops restricting
        # any unroll. A multi-arch build must still single out the unrolls that only
        # one arch compiles, or there is nothing for the runtime check to catch.
        arch_table = self._unroll_table(
            "ncclDevFuncUnrollArch", r'nullptr|"gfx\w+"'
        )
        self.assertTrue(
            any(value != "nullptr" for value in arch_table.values()),
            "no unroll factor is arch-restricted; ncclDevFuncUnrollArch[] is all nullptr",
        )

    def test_manifest_guard_matches_the_shard_it_guards(self):
        # cmake/DeviceLinker.cmake filters shards on the manifest column while the compiler
        # obeys the #if, so any drift drops a symbol the dispatch table still declares.
        rows = []
        with open(os.path.join(self._dir, "specialized_files.txt")) as f:
            for line in f:
                if line.strip():
                    parts = line.rstrip("\n").split(" ", 2)
                    rows.append((parts[0], parts[2] if len(parts) > 2 else ""))
        self.assertTrue(rows, "generator wrote no specialized_files.txt rows")
        for filename, guard in rows:
            with open(os.path.join(self._dir, "specialized", filename)) as f:
                shard = f.read()
            emitted = re.findall(r"^#if (.*)$", shard, re.M)
            self.assertEqual(
                [guard] if guard else [],
                emitted,
                "%s: manifest guard %r does not match the #if in the shard %r"
                % (filename, guard, emitted),
            )

    def test_no_obsolete_table_omit_macro(self):
        # RCCL_DEVICE_TABLE_OMIT was retired by the static-table change.
        self.assertNotIn("RCCL_DEVICE_TABLE_OMIT", self.header)

    def test_specialized_shards_do_not_omit(self):
        # Specialized shards no longer #define RCCL_DEVICE_TABLE_OMIT.
        spec_dir = os.path.join(self._dir, "specialized")
        self.assertTrue(os.path.isdir(spec_dir))
        for name in os.listdir(spec_dir):
            with open(os.path.join(spec_dir, name)) as f:
                self.assertNotIn("RCCL_DEVICE_TABLE_OMIT", f.read())

    def test_ifc_build_keeps_table_and_no_rdc_dispatch(self):
        # Don't break the legacy indirect-function-call path: with IFC on, the
        # table is still emitted and the pure-RDC dispatcher is not.
        with tempfile.TemporaryDirectory(prefix="rccl_devtable_ifc_") as d:
            header = _generate(d, ifc="ON")
        self.assertIn("static __device__ ncclDevFuncPtr_t const ncclDevFuncTable_", header)


if __name__ == "__main__":
    unittest.main()

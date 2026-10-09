#!/usr/bin/env python3
"""CPU overlap oracle using the current FMM bodies and real red-black tree.

Run directly with Python 3 and a C compiler. No GPU or installed ROCm needed.
"""

import ctypes
import random
import subprocess
import tempfile
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"
PAGE = 4096


def function(text, name):
    start = text.index("static ", text.index(name) - 32)
    brace = text.index("{", start)
    depth = 1
    end = brace + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    return text[start:end]


def translation_unit():
    text = (SRC / "fmm.c").read_text()
    extent = function(text, "userptr_page_extent(const")
    remove = function(text, "vm_remove_object(manageable")
    retire = function(text, "fmm_retire_stale_userptrs(Hsa")
    # Exercise the exact insertion and maximum-update statements.
    first = text.index(
        "\t\tobj->userptr = addr;",
        text.index("static HSAKMT_STATUS fmm_register_user_memory("),
    )
    last = text.index("\n\t}", first)
    insert = text[first:last]
    return (
        r"""
#include <stddef.h>
#include <stdint.h>
#include "rbtree.h"
#define PAGE_SIZE 4096UL
#define PAGE_ALIGN_UP(x) (((x) + PAGE_SIZE - 1) & ~(PAGE_SIZE - 1))
typedef void HsaKFDContext;
typedef struct {
    rbtree_node_t node, user_node;
    void *start, *userptr;
    uint64_t userptr_size;
    void *registered_device_id_array, *mapped_device_id_array, *metadata;
    void *registered_node_id_array, *mapped_node_id_array;
    unsigned mapped_device_id_array_size, mapping_count, registration_count, node_id;
    unsigned id;
} vm_object_t;
typedef struct {
    rbtree_t tree, user_tree;
    uint64_t userptr_max_extent;
} manageable_aperture_t;
#define vm_object_entry(n, u) ((vm_object_t *)((char *)(n) - offsetof(vm_object_t, user_node)))
static manageable_aperture_t app;
static vm_object_t *objects[100000];
static unsigned retired[100000], retired_count;
static int fail_unmap, fail_release;
static int _fmm_unmap_from_gpu(HsaKFDContext *ctx, manageable_aperture_t *a,
    void *p, void *ids, unsigned n, vm_object_t *obj) {
    if (obj->mapping_count != 1) abort();
    return fail_unmap;
}
static void hsakmt_gpuid_to_nodeid(HsaKFDContext *ctx, unsigned gpu, unsigned *node) { *node = gpu; }
"""
        + extent
        + "\n"
        + remove
        + r"""
static int __fmm_release_locked(HsaKFDContext *ctx, vm_object_t *obj, manageable_aperture_t *a) {
    if (fail_release) return fail_release;
    if (obj->registration_count != 1) abort();
    retired[retired_count++] = obj->id;
    objects[obj->id] = NULL;
    vm_remove_object(a, obj);
    return 0;
}
"""
        + retire
        + r"""
void add(unsigned id, uint64_t address, uint64_t size, unsigned mapped) {
    HsaKFDContext *ctx = NULL;
    manageable_aperture_t *aperture = &app;
    vm_object_t *obj = calloc(1, sizeof(*obj));
    void *addr = (void *)address;
    unsigned gpu_id = 0;
    uint64_t extent;
    if (!obj || objects[id]) abort();
    obj->id = id; obj->start = addr;
    obj->node.key = rbtree_key(address, size);
    hsakmt_rbtree_insert(&app.tree, &obj->node);
"""
        + insert
        + r"""
    obj->mapped_device_id_array_size = mapped;
    obj->mapping_count = 7;
    obj->registration_count = 9;
    objects[id] = obj;
}
void drop(unsigned id) {
    vm_remove_object(&app, objects[id]); objects[id] = NULL;
}
void reset(void) {
    for (unsigned i = 0; i < 100000; ++i) if (objects[i]) drop(i);
    rbtree_init(&app.tree); rbtree_init(&app.user_tree);
    app.userptr_max_extent = 0;
}
unsigned count(void) { return retired_count; }
unsigned identity(unsigned i) { return retired[i]; }
uint64_t bound(void) { return app.userptr_max_extent; }
int run(uint64_t address, uint64_t size, int unmap_error, int release_error) {
    retired_count = 0; fail_unmap = unmap_error; fail_release = release_error;
    return fmm_retire_stale_userptrs(NULL, &app, (void *)address, size);
}
"""
    )


def verify(lib):
    lib.add.argtypes = [ctypes.c_uint, ctypes.c_uint64, ctypes.c_uint64, ctypes.c_uint]
    lib.run.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_int, ctypes.c_int]
    lib.bound.restype = ctypes.c_uint64
    checks = 0
    live = {}

    def reset():
        lib.reset()
        live.clear()

    def add(identity, address, size, mapped=1):
        lib.add(identity, address, size, mapped)
        live[identity] = (address, size)
        expected = max(
            ((a + s + PAGE - 1) // PAGE * PAGE - a // PAGE * PAGE)
            for a, s in live.values()
        )
        assert lib.bound() >= expected

    def run(address, size):
        nonlocal checks
        end = (address + size + PAGE - 1) // PAGE * PAGE
        expected = {
            i
            for i, (a, s) in live.items()
            if a // PAGE * PAGE < end and (a + s + PAGE - 1) // PAGE * PAGE > address
        }
        assert lib.run(address, size, 0, 0) == 0
        actual = [lib.identity(i) for i in range(lib.count())]
        assert len(actual) == len(set(actual)) and set(actual) == expected, (
            address,
            size,
            expected,
            actual,
        )
        for i in expected:
            del live[i]
        if not live:
            assert lib.bound() == 0
        checks += 1

    reset()
    run(0, PAGE)
    # Before-range overlap, smaller duplicate, unaligned span, adjacent ranges.
    for i, (a, s) in enumerate(
        [
            (PAGE, 4 * PAGE),
            (3 * PAGE, PAGE),
            (3 * PAGE, 3 * PAGE),
            (4 * PAGE - 1, 2),
            (6 * PAGE, PAGE),
            (1, PAGE),
        ]
    ):
        add(i, a, s)
    run(3 * PAGE, PAGE)
    run(0, PAGE)  # Lower bound saturates at zero.
    run(5 * PAGE, PAGE)  # Adjacency excludes the remaining registration.
    run(6 * PAGE, PAGE)
    add(20, PAGE, 100 * PAGE)
    add(21, 200 * PAGE, PAGE)
    lib.drop(20)
    del live[20]
    assert lib.bound() >= 100 * PAGE  # Conservative bound retained.
    run(200 * PAGE, PAGE)
    add(22, 400 * PAGE, 1)
    assert lib.bound() == PAGE
    run(400 * PAGE, PAGE)
    for error_args in [(5, 0), (0, 7)]:
        reset()
        add(0, PAGE, PAGE)
        assert lib.run(PAGE, PAGE, *error_args) == sum(error_args)
        assert lib.count() == 0
        run(PAGE, PAGE)
    rng = random.Random(28509)
    for trial in range(40):
        reset()
        next_id = 0
        for step in range(250):
            choice = rng.randrange(4)
            if choice < 2:
                a = rng.randrange(1, 1024 * PAGE)
                s = rng.randrange(1, 64 * PAGE)
                add(next_id, a, s, rng.randrange(2))
                next_id += 1
            elif choice == 2 and live:
                i = rng.choice(list(live))
                lib.drop(i)
                del live[i]
                if not live:
                    assert lib.bound() == 0
            else:
                run(rng.randrange(1024) * PAGE, rng.randrange(1, 32) * PAGE)
        run(0, 2048 * PAGE)
    reset()
    print(
        f"PASS: {checks} independent overlap checks; 10000 deterministic mixed operations; error propagation"
    )


def main():
    with tempfile.TemporaryDirectory(prefix="fmm-retirement-") as directory:
        directory = Path(directory)
        unit = directory / "retirement.c"
        unit.write_text(translation_unit())
        library = directory / "retirement.so"
        subprocess.run(
            [
                "cc",
                "-shared",
                "-fPIC",
                "-O2",
                "-I",
                str(SRC),
                str(unit),
                str(SRC / "rbtree.c"),
                "-o",
                str(library),
            ],
            check=True,
        )
        verify(ctypes.CDLL(str(library)))


if __name__ == "__main__":
    main()

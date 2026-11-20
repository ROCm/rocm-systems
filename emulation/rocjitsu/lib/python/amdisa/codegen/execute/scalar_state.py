# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Emit the VM-owned scalar moves and EXEC updates not handled by rj-arith."""

from amdisa.codegen.execute import ExecuteContext

# SAVEEXEC preserves the previous EXEC in the destination before updating EXEC.
_EXEC_OPERATIONS = {
    'and': '(old_exec & src)',
    'or': '(old_exec | src)',
    'xor': '(old_exec ^ src)',
    'andn1': '(old_exec & (~src))',
    'andn2': '(src & (~old_exec))',
    'orn1': '(old_exec | (~src))',
    'orn2': '(src | (~old_exec))',
    'and_not0': '(old_exec & (~src))',
    'and_not1': '(src & (~old_exec))',
    'or_not0': '(old_exec | (~src))',
    'or_not1': '(src | (~old_exec))',
    'nand': '(~(old_exec & src))',
    'nor': '(~(old_exec | src))',
    'xnor': '(~(old_exec ^ src))',
}


def gen_scalar_state(ctx: ExecuteContext) -> str | None:
    """Generate scalar move/EXEC register effects, preserving source aliasing."""
    if ctx.cls not in ('scalar_mov', 'scalar_cmov', 'scalar_saveexec', 'scalar_wrexec'):
        return None
    wide = ctx.dtype == 'b64'
    source, destination = ctx.src_ops[0], ctx.dst_ops[0]
    read = f'amdgpu::RegisterAccess(wf).read_scalar{"64" if wide else ""}({source})'
    write = f'amdgpu::RegisterAccess(wf).write_scalar{"64" if wide else ""}'
    if ctx.cls in ('scalar_mov', 'scalar_cmov'):
        statement = f'{write}({destination}, {read});'
        if ctx.cls == 'scalar_cmov':
            return f'  if (wf.read_scc())\n    {statement}'
        return f'  {statement}'

    expression = _EXEC_OPERATIONS[ctx.op]
    if not wide:
        expression = f'({expression} & 0xffffffffULL)'
    source_read = f'static_cast<uint64_t>({read})' if wide else read
    exec_read = 'wf.exec_raw()' if wide else 'static_cast<uint32_t>(wf.exec_raw())'
    exec_write = 'wf.write_exec' if wide else 'wf.write_exec_lo'
    lines = [f'  uint64_t src = {source_read};', f'  uint64_t old_exec = {exec_read};']
    if ctx.cls == 'scalar_saveexec':
        saved = 'old_exec' if wide else 'static_cast<uint32_t>(old_exec)'
        lines += [
            f'  {write}({destination}, {saved});',
            f'  {exec_write}({expression});',
            f'  wf.write_scc(({exec_read} != 0ULL));',
        ]
    else:
        lines += [
            f'  uint64_t result = {expression};',
            f'  {write}({destination}, static_cast<uint{64 if wide else 32}_t>(result));',
            f'  {exec_write}(result);',
            '  wf.write_scc((result != 0ULL));',
        ]
    return '\n'.join(lines)

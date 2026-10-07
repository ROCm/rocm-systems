import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test, vi } from 'vitest';
import BenchmarkResultDialog from '../../src/components/benchmarks/BenchmarkResultDialog.jsx';
import { formatDuration } from '../../src/utils/formatters.js';
vi.mock('@mui/material', async (importOriginal) => {
  const original = await importOriginal();
  return { ...original, Dialog: ({ open, children }) => open ? createElement('div', { role: 'dialog' }, children) : null };
});

test('benchmark details disclose authoritative simulator mode, thread count and arbitrary environment facts', () => {
  const record = { run: { runId: 'fictional-details', timestamp: '2026-10-05T14:00:00Z', commitTimestamp: '2026-10-05T13:00:00Z', configurations: [{ target: 'gfx1250', mode: 'MT', threadCount: 8 }], environment: [{ key: 'tool', label: 'Generic tool', value: '-full+value' }], provenance: { rocjitsuCommitSha: 'a'.repeat(40) } }, test: { name: 'Fictional workload', target: 'gfx1250', mode: 'MT', suite: 'Triton', status: 'completed', durationSeconds: 0 } };
  const html = renderToStaticMarkup(createElement(BenchmarkResultDialog, { record, onClose() {} }));
  expect(html).toContain('gfx1250 · MT · Triton');
  expect(html).toContain('Simulator threads');
  expect(html).toContain('Generic tool');
  expect(html).toContain('-full+value');
  expect(html).toContain(formatDuration(0));
  expect(html).toContain('fictional-details');
});

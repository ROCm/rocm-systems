import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { expect, test, vi } from 'vitest';
import RunSelector from '../../src/components/compare/RunSelector.jsx';
const captured = vi.hoisted(() => ({ autocomplete: null }));
vi.mock('@mui/material', async (importOriginal) => {
  const original = await importOriginal();
  return { ...original, Autocomplete: (props) => { captured.autocomplete = props; return createElement(original.Autocomplete, props); } };
});

test('run selection searches branch, mode and complete attempt identifiers without clipping selected facts', () => {
  const options = Array.from({ length: 60 }, (_, i) => ({ runId: `fictional-attempt-${i}`, branch: i === 59 ? 'topic/find-me' : 'develop', modes: i === 59 ? ['MT'] : ['ST'], timestamp: '2026-10-05T14:00:00Z', commitTimestamp: '2026-10-05T13:00:00Z', provenance: { rocjitsuCommitSha: 'a'.repeat(40) } }));
  const onChange = vi.fn();
  const html = renderToStaticMarkup(createElement(RunSelector, { label: 'Candidate run', options, value: options[59], onChange }));
  expect(captured.autocomplete.filterOptions(options, { inputValue: 'topic/find-me' })).toEqual([options[59]]);
  expect(captured.autocomplete.filterOptions(options, { inputValue: 'MT' })).toEqual([options[59]]);
  expect(captured.autocomplete.filterOptions(options, { inputValue: '' })).toHaveLength(50);
  expect(html).toContain('topic/find-me · MT');
  expect(html).toContain('fictional-attempt-59');
  captured.autocomplete.onChange(null, options[59]);
  expect(onChange).toHaveBeenCalledWith(options[59].runId);
});

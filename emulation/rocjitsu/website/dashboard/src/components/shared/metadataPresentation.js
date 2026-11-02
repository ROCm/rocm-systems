export function metadataLabel(label, key) {
  return !label || label === key ? key.slice(key.indexOf('.') + 1) : label;
}

const sectionLabels = {
  machine: 'Machine', execution: 'Execution & measurement', corpus: 'Benchmark corpus',
  rocjitsu: 'Rocjitsu', runtime: 'Runtime', auxiliary: 'Additional tools & settings',
};

// Keep arbitrary published keys, including facts present in only one attempt.
export function groupMetadataFacts(facts) {
  const groups = new Map();
  for (const fact of facts) {
    const namespace = fact.key.includes('.') ? fact.key.split('.')[0] : 'other';
    const label = Object.hasOwn(sectionLabels, namespace) ? sectionLabels[namespace] : namespace === 'other' ? 'Other environment facts' : namespace;
    if (!groups.has(label)) groups.set(label, []);
    groups.get(label).push(fact);
  }
  return [...groups].map(([label, rows]) => ({ label, rows }));
}

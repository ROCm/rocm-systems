import { hasDisplayValue } from '../utils/values';

const legacyDetails = [
  ['rocmSdkVersion', 'ROCm SDK'],
  ['pythonVersion', 'Python'],
  ['torchVersion', 'PyTorch'],
  ['tritonCommitSha', 'Triton commit'],
  ['tensileLiteCommitSha', 'TensileLite commit'],
];

function displayValue(value) {
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'string' || typeof value === 'number') return String(value);
  return null;
}

export function provenanceDetails(provenance = {}) {
  const details = [];
  const seenKeys = new Set();
  const seenLabels = new Set();

  const addDetail = (key, label, value) => {
    const normalizedKey = hasDisplayValue(key) ? String(key).trim() : '';
    const normalizedLabel = hasDisplayValue(label) ? String(label).trim() : '';
    const normalizedValue = displayValue(value);
    const labelKey = normalizedLabel.toLocaleLowerCase();
    if (!normalizedKey || !normalizedLabel || !hasDisplayValue(normalizedValue)) return;
    if (seenKeys.has(normalizedKey) || seenLabels.has(labelKey)) return;
    seenKeys.add(normalizedKey);
    seenLabels.add(labelKey);
    details.push({ key: normalizedKey, label: normalizedLabel, value: normalizedValue });
  };

  if (Array.isArray(provenance.details)) {
    provenance.details.forEach((detail) => {
      if (detail && typeof detail === 'object') addDetail(detail.key, detail.label, detail.value);
    });
  }

  legacyDetails.forEach(([key, label]) => addDetail(key, label, provenance[key]));
  return details;
}

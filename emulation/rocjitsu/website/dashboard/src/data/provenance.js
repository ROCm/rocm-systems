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


  const addDetail = (key, label, value) => {
    const normalizedKey = hasDisplayValue(key) ? String(key) : '';
    const normalizedLabel = hasDisplayValue(label) ? String(label) : '';
    const normalizedValue = displayValue(value);
    if (!normalizedKey || !normalizedLabel || normalizedValue === null) return;
    if (seenKeys.has(normalizedKey)) return;
    seenKeys.add(normalizedKey);

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

export function runEnvironmentDetails(run) {
  return new Map([
    ...provenanceDetails(run?.provenance),
    ...(run?.provenance?.details ?? []),
    ...(run?.environment ?? []),
  ].map((detail) => [detail.key, detail]));
}

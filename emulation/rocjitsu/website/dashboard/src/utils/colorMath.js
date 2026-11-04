// HSL (hue 0-360, saturation and lightness 0-1) to #rrggbb. One owner for the dashboard's colour maths.
export function hslToHex(hue, saturation, lightness) {
  const chroma = (1 - Math.abs(2 * lightness - 1)) * saturation;
  // Standard HSL conversion: channel offsets are 0 (r), 8 (g), 4 (b) in twelfths of the hue circle.
  const channel = (n) => {
    const k = (n + (hue / 30)) % 12;
    return lightness - (chroma / 2) * Math.max(-1, Math.min(k - 3, 9 - k, 1));
  };
  return `#${[channel(0), channel(8), channel(4)].map((v) => Math.round(v * 255).toString(16).padStart(2, '0')).join('')}`;
}

// Keeps a colour's hue and lowers its saturation, so text can match a marker's
// hue without being as vivid. Input and output are #rrggbb palette values.
export function softenColor(hex, saturationScale = 0.55, lightnessShift = -0.02) {
  const [r, g, b] = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16) / 255);
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const lightness = (max + min) / 2;
  const delta = max - min;
  let hue = 0;
  let saturation = 0;
  if (delta) {
    saturation = delta / (1 - Math.abs(2 * lightness - 1));
    if (max === r) hue = ((g - b) / delta + 6) % 6;
    else if (max === g) hue = (b - r) / delta + 2;
    else hue = (r - g) / delta + 4;
  }
  return hslToHex(hue * 60, saturation * saturationScale, Math.min(1, Math.max(0, lightness + lightnessShift)));
}

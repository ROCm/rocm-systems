import { alpha } from '@mui/material/styles';

export function chartAreaGradient(color, topOpacity = 0.2) {
  return {
    type: 'linear',
    x: 0,
    y: 0,
    x2: 0,
    y2: 1,
    colorStops: [
      { offset: 0, color: alpha(color, topOpacity) },
      { offset: 0.7, color: alpha(color, topOpacity * 0.05) },
      { offset: 1, color: alpha(color, 0) },
    ],
  };
}

export function chartLineStyle(color, width = 2.6) {
  return {
    color,
    width,
    shadowBlur: 7,
    shadowColor: alpha(color, 0.22),
    shadowOffsetY: 2,
  };
}

export function chartPointStyle(color, surfaceColor) {
  return {
    color,
    borderColor: surfaceColor,
    borderWidth: 2,
    shadowBlur: 9,
    shadowColor: alpha(color, 0.34),
  };
}

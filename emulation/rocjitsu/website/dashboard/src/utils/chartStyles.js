import { alpha } from '@mui/material/styles';

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

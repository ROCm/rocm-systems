import { isSupportedNumber } from './jsonValues.js';

// Check the decimal magnitude without first rounding its significant digits.
function validateNumericToken(token) {
  const number = Number(token);
  const [mantissa, exponent = '0'] = token.replace(/^-/, '').toLowerCase().split('e');
  const [whole, fraction = ''] = mantissa.split('.');
  const raw = whole + fraction;
  const digits = raw.replace(/^0+/, '');
  if (!digits) return; // Genuine zero, including signed/exponent zero.
  const point = whole.length + Number(exponent) - (raw.length - digits.length);
  const maximum = String(Number.MAX_SAFE_INTEGER);
  const integer = point > 0 && point <= maximum.length
    ? digits.slice(0, point).padEnd(point, '0') : '';
  if (!isSupportedNumber(number) || number === 0 || point > maximum.length
    || point === maximum.length && (integer > maximum
      || integer === maximum && /[1-9]/.test(digits.slice(point)))) {
    throw new Error(`Unsupported numeric JSON token: ${token}`);
  }
}

// Native JSON parsing validates syntax and produces values; this scan additionally
// rejects duplicate decoded keys before last-write-wins can hide declarations.
export function parsePublishedJson(text) {
  const result = JSON.parse(text);
  let position = 0;
  const whitespace = () => { while (/\s/.test(text[position] ?? '') && position < text.length) position += 1; };
  const string = () => {
    const start = position;
    position += 1;
    while (text[position] !== '"') {
      position += text[position] === '\\' ? 2 : 1;
    }
    position += 1;
    return JSON.parse(text.slice(start, position));
  };
  const value = () => {
    whitespace();
    const character = text[position];
    if (character === '"') { string(); return; }
    if (character === '{') {
      position += 1;
      const keys = new Set();
      whitespace();
      while (text[position] !== '}') {
        const key = string();
        if (keys.has(key)) throw new Error(`Duplicate JSON key: ${key}`);
        keys.add(key);
        whitespace();
        position += 1; // colon, already checked by JSON.parse
        value();
        whitespace();
        if (text[position] === '}') break;
        position += 1;
        whitespace();
      }
      position += 1;
      return;
    }
    if (character === '[') {
      position += 1;
      whitespace();
      while (text[position] !== ']') {
        value();
        whitespace();
        if (text[position] === ']') break;
        position += 1;
      }
      position += 1;
      return;
    }
    const start = position;
    while (position < text.length && !/[\s,}\]]/.test(text[position])) position += 1;
    const token = text.slice(start, position);
    if (/^-?\d/.test(token)) validateNumericToken(token);
  };
  value();
  return result;
}

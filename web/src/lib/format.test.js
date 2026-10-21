import { describe, expect, it } from 'vitest';
import { delta, fixed, scaleLinear } from './format.js';

describe('format', () => {
  it('scaleLinear maps domain ends', () => {
    const s = scaleLinear([0.5, 0.75], [0, 400]);
    expect(s(0.5)).toBe(0);
    expect(s(0.75)).toBe(400);
    expect(s(0.625)).toBe(200);
  });

  it('delta formats sign', () => {
    expect(delta(0.0091)).toBe('+0.009');
    expect(delta(-0.0123)).toBe('−0.012');
    expect(delta(0)).toBe('±0.000');
  });

  it('fixed rounds to digits', () => {
    expect(fixed(0.70554)).toBe('0.706');
    expect(fixed(0.7, 2)).toBe('0.70');
  });
});

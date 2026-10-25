import { describe, expect, it } from 'vitest';
import { densityGrid, EXTENT, inkOn, LEGEND, PALETTE, pointRadius } from './map-style.js';
import { TEXTLESS } from './map.js';

describe('map style', () => {
  it('has a colour for every legend entry in both themes', () => {
    for (const theme of [PALETTE.light, PALETTE.dark]) {
      expect(Object.keys(theme).sort()).toEqual(LEGEND.map(([k]) => k).sort());
      expect(new Set(Object.values(theme)).size).toBe(LEGEND.length);
    }
  });

  it('grows dots with zoom but keeps them between 1.1 and 5 pixels', () => {
    const fit = 600 / 66000;
    const sizes = [1, 2, 4, 8, 60].map((z) => pointRadius(fit * z, fit, 27866));
    expect(sizes[0]).toBeGreaterThanOrEqual(1.1);
    for (let k = 1; k < sizes.length; k += 1) expect(sizes[k]).toBeGreaterThanOrEqual(sizes[k - 1]);
    expect(sizes.at(-1)).toBe(5);
    expect(pointRadius(fit, fit, 10)).toBe(5);
  });

  it('scales density to 0..1, peaking where the cards are and skipping textless ones', () => {
    const x = new Int16Array([0, 0, 0, 30000, TEXTLESS]), y = new Int16Array([0, 0, 0, 30000, TEXTLESS]);
    const size = 32, d = densityGrid(x, y, size, EXTENT, TEXTLESS);
    expect(d.length).toBe(size * size);
    const at = (wx, wy) => d[Math.floor(((wy + EXTENT) / (2 * EXTENT)) * size) * size + Math.floor(((wx + EXTENT) / (2 * EXTENT)) * size)];
    expect(Math.max(...d)).toBe(1);
    expect(at(0, 0)).toBe(1);
    expect(at(30000, 30000)).toBeGreaterThan(0);
    expect(at(30000, 30000)).toBeLessThan(at(0, 0));
    expect(at(-30000, 30000)).toBe(0);
  });

  it('picks a dark glyph on light swatches and white on dark ones', () => {
    expect(inkOn(PALETTE.dark.w)).toBe('#15171c');
    expect(inkOn(PALETTE.light.b)).toBe('#ffffff');
  });
});

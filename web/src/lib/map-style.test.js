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

describe('region labels', async () => {
  const { labelBudget, labelStyle } = await import('./map-style.js');
  it('draws a few labels on a small map and more as the view zooms in, up to a cap', () => {
    expect(labelBudget(200, 200, 1)).toBe(4);
    const z = [1, 2, 4, 9, 50].map((k) => labelBudget(620, 800, k));
    for (let k = 1; k < z.length; k += 1) expect(z[k]).toBeGreaterThanOrEqual(z[k - 1]);
    expect(z[0]).toBeLessThan(z[3]);
    expect(z[4]).toBe(z[3]);
  });
  it('the largest regions lead with bigger, bolder labels', () => {
    const lead = labelStyle(0, 1), rest = labelStyle(10, 1);
    expect(lead.size).toBeGreaterThan(rest.size);
    expect(lead.weight).toBeGreaterThan(rest.weight);
    expect(lead.alpha).toBeGreaterThan(rest.alpha);
    expect(labelStyle(0, 8).size).toBeGreaterThan(lead.size);
  });
});

describe('dataBounds', async () => {
  const { dataBounds } = await import('./map-style.js');
  it('spans the placed cards and skips textless ones', () => {
    const x = new Int16Array([-100, 300, TEXTLESS, 50]), y = new Int16Array([20, -40, TEXTLESS, 500]);
    expect(dataBounds(x, y, TEXTLESS)).toEqual({ x0: -100, y0: -40, x1: 300, y1: 500 });
  });
  it('is null when no card is placed', () => expect(dataBounds(new Int16Array([TEXTLESS]), new Int16Array([TEXTLESS]), TEXTLESS)).toBeNull());
});

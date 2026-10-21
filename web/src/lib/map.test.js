import { describe, expect, it } from 'vitest';
import { buildGrid, decodeMap, hitTest, toScreen, toWorld, TEXTLESS, regionLabel } from './map.js';

function pack(records) {
  const buf = new ArrayBuffer(records.length * 5);
  const view = new DataView(buf);
  records.forEach(([x, y, c], i) => { view.setInt16(i * 5, x, true); view.setInt16(i * 5 + 2, y, true); view.setUint8(i * 5 + 4, c); });
  return buf;
}

describe('map', () => {
  it('decodeMap reads 5-byte records', () => {
    const m = decodeMap(pack([[-32000, 5, 1], [TEXTLESS, TEXTLESS, 0], [32000, -7, 10]]));
    expect(m.n).toBe(3);
    expect([...m.x]).toEqual([-32000, TEXTLESS, 32000]);
    expect([...m.y]).toEqual([5, TEXTLESS, -7]);
    expect([...m.color]).toEqual([1, 0, 10]);
  });

  it('decodeMap rejects a truncated buffer', () => {
    expect(() => decodeMap(new ArrayBuffer(7))).toThrow();
  });

  it('buildGrid skips textless sentinel', () => {
    const m = decodeMap(pack([[0, 0, 1], [TEXTLESS, TEXTLESS, 0], [600, 0, 2]]));
    const grid = buildGrid(m.x, m.y, 512);
    expect([...grid.values()].flat().sort()).toEqual([0, 2]);
  });

  it('hitTest returns nearest within radius and -1 outside', () => {
    const m = decodeMap(pack([[0, 0, 1], [100, 0, 1], [1000, 1000, 1]]));
    const grid = buildGrid(m.x, m.y, 512);
    expect(hitTest(grid, m.x, m.y, 80, 0, 50)).toBe(1);
    expect(hitTest(grid, m.x, m.y, 10, 5, 50)).toBe(0);
    expect(hitTest(grid, m.x, m.y, 500, 500, 50)).toBe(-1);
    expect(hitTest(grid, m.x, m.y, 1020, 990, 50)).toBe(2);
  });

  it('toWorld inverts toScreen', () => {
    const v = { scale: 0.01, tx: 200, ty: 150 };
    const [sx, sy] = toScreen(v, 1234, -5678);
    const [wx, wy] = toWorld(v, sx, sy);
    expect(wx).toBeCloseTo(1234, 6);
    expect(wy).toBeCloseTo(-5678, 6);
  });
});

describe('regionLabel', () => {
  it('renames tagger jargon for readers', () => {
    expect(regionLabel('pp counters matter')).toBe('+1/+1 counters matter');
    expect(regionLabel('thoughtseize')).toBe('hand disruption');
  });
  it('passes plain labels through', () => {
    expect(regionLabel('spot removal')).toBe('spot removal');
  });
});

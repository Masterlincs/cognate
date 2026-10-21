import { describe, expect, it } from 'vitest';
import { resolveSelection, visibleNeighbours } from './explorer.js';

const ids = ['a1', 'b2', 'c3'];

describe('explorer state', () => {
  it('resolveSelection finds a known id', () => {
    expect(resolveSelection(ids, { card: 'b2' })).toEqual({ index: 1, unknown: false });
  });

  it('resolveSelection flags unknown id', () => {
    expect(resolveSelection(ids, { card: 'zz' })).toEqual({ index: -1, unknown: true });
    expect(resolveSelection(ids, { card: '' })).toEqual({ index: -1, unknown: false });
  });

  it('visibleNeighbours applies filters and keeps order', () => {
    const cards = { 1: { color_identity: 'R', mv: 1, type_line: 'Instant', is_creature: false }, 2: { color_identity: 'G', mv: 3, type_line: 'Creature — Elf', is_creature: true } };
    const saved = [[2, 0.9], [1, 0.8]];
    expect(visibleNeighbours(saved, (i) => cards[i], {})).toEqual(saved);
    expect(visibleNeighbours(saved, (i) => cards[i], { color: 'R' })).toEqual([[1, 0.8]]);
    expect(visibleNeighbours(saved, (i) => cards[i], { creature: 'creature', maxMv: 2 })).toEqual([]);
  });

  it('visibleNeighbours drops entries whose card is not loaded yet', () => {
    expect(visibleNeighbours([[5, 0.5]], () => undefined, {})).toEqual([]);
  });

  it('activeFilters is false for an empty filter set', async () => {
    const { activeFilters } = await import('./explorer.js');
    expect(activeFilters({ color: '', minMv: null, maxMv: null, creature: '', type: '' })).toBe(false);
    expect(activeFilters({ color: 'U', minMv: null, maxMv: null, creature: '', type: '' })).toBe(true);
  });
});

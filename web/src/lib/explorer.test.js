import { describe, expect, it } from 'vitest';
import { hasMore, resolveSelection, savedList } from './explorer.js';

const ids = ['a1', 'b2', 'c3'];

describe('explorer state', () => {
  it('resolveSelection finds a known id', () => {
    expect(resolveSelection(ids, { card: 'b2' })).toEqual({ index: 1, unknown: false });
  });

  it('resolveSelection flags unknown id', () => {
    expect(resolveSelection(ids, { card: 'zz' })).toEqual({ index: -1, unknown: true });
    expect(resolveSelection(ids, { card: '' })).toEqual({ index: -1, unknown: false });
  });

  it('activeFilters is false for an empty filter set', async () => {
    const { activeFilters } = await import('./explorer.js');
    expect(activeFilters({ color: '', minMv: null, maxMv: null, creature: '', type: '' })).toBe(false);
    expect(activeFilters({ color: 'U', minMv: null, maxMv: null, creature: '', type: '' })).toBe(true);
  });
});

describe('savedList', () => {
  const whole = [[1, 0.9]], lines = [[2, 0.8]], abilities = [{ neighbours: [[3, 0.7]] }, { neighbours: [[4, 0.6]] }];
  it('shows the whole-card list by default', () => {
    expect(savedList({ ability: null, rank: 'whole', whole, lines, abilities })).toBe(whole);
  });
  it('shows the line-match list when ranked by line', () => {
    expect(savedList({ ability: null, rank: 'line', whole, lines, abilities })).toBe(lines);
  });
  it('an ability overrides the ranking', () => {
    expect(savedList({ ability: 1, rank: 'line', whole, lines, abilities })).toEqual([[4, 0.6]]);
  });
  it('line ranking before its shard loads is empty, not the whole-card list', () => {
    expect(savedList({ ability: null, rank: 'line', whole, lines: null, abilities })).toEqual([]);
  });
  it('uses face matches only when no ability is selected', () => {
    const faces = [[5, 0.5]];
    expect(savedList({ ability: null, rank: 'face', whole, lines, faces, abilities })).toBe(faces);
    expect(savedList({ ability: 0, rank: 'face', whole, lines, faces, abilities })).toEqual([[3, 0.7]]);
    expect(savedList({ ability: null, rank: 'face', whole, lines, faces: null, abilities })).toEqual([]);
  });
});

describe('hasMore', () => {
  const saved = Array.from({ length: 50 }, (_, i) => [i, 1 - i / 100]);
  it('offers more while unfiltered neighbours remain, even before their cards load', () => {
    expect(hasMore({ saved, visible: saved.slice(0, 12), shown: 12, filtering: false })).toBe(true);
    expect(hasMore({ saved, visible: saved, shown: 48, filtering: false })).toBe(true);
    expect(hasMore({ saved, visible: saved, shown: 60, filtering: false })).toBe(false);
  });
  it('counts only matching cards while filtering', () => {
    expect(hasMore({ saved, visible: saved.slice(0, 10), shown: 12, filtering: true })).toBe(false);
    expect(hasMore({ saved, visible: saved.slice(0, 20), shown: 12, filtering: true })).toBe(true);
  });
});

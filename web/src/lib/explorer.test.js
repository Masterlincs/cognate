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

describe('filterChips', async () => {
  const { clearFilter, filterChips } = await import('./explorer.js');
  const none = { color: '', colorMode: 'include', colorBy: 'identity', minMv: null, maxMv: null, creature: '', type: '' };
  it('has no chips without filters', () => expect(filterChips(none)).toEqual([]));
  it('names each active filter once, in display order', () => {
    const f = { ...none, color: 'UR', colorMode: 'exact', creature: 'noncreature', type: 'goblin', minMv: 2, maxMv: 4 };
    expect(filterChips(f)).toEqual([
      { key: 'color', label: 'Identity: exactly Blue, Red' },
      { key: 'creature', label: 'Non-creatures' },
      { key: 'type', label: 'Type: goblin' },
      { key: 'mv', label: 'MV 2–4' },
    ]);
  });
  it('describes colourless, printed colour, at-most and one-sided mana ranges', () => {
    expect(filterChips({ ...none, color: 'C', colorMode: 'exact' })[0].label).toBe('Identity: colourless');
    expect(filterChips({ ...none, color: 'G', colorMode: 'within', colorBy: 'color' })[0].label).toBe('Colour: at most Green');
    expect(filterChips({ ...none, color: 'W' })[0].label).toBe('Identity: including White');
    expect(filterChips({ ...none, minMv: 3 })[0].label).toBe('MV ≥ 3');
    expect(filterChips({ ...none, maxMv: 0 })[0].label).toBe('MV ≤ 0');
    expect(filterChips({ ...none, minMv: 2, maxMv: 2 })[0].label).toBe('MV 2');
  });
  it('clearFilter resets only the named filter, colour with its match settings', () => {
    const f = { ...none, color: 'B', colorMode: 'within', colorBy: 'color', minMv: 1, maxMv: 5, type: 'elf', creature: 'creature' };
    expect(clearFilter(f, 'color')).toEqual({ ...f, color: '', colorMode: 'include', colorBy: 'identity' });
    expect(clearFilter(f, 'mv')).toEqual({ ...f, minMv: null, maxMv: null });
    expect(clearFilter(f, 'type')).toEqual({ ...f, type: '' });
    expect(clearFilter(f, 'creature')).toEqual({ ...f, creature: '' });
    expect(f.color).toBe('B');
  });
});

describe('randomIndex', async () => {
  const { randomIndex } = await import('./explorer.js');
  const seq = (...xs) => () => xs.shift() ?? 0;
  it('takes the drawn index when it passes', () => expect(randomIndex(10, () => true, seq(0.55))).toBe(5));
  it('draws again past rejected indexes', () => expect(randomIndex(10, (i) => i === 7, seq(0.1, 0.2, 0.75))).toBe(7));
  it('falls back to a scan when draws keep missing, and -1 when nothing passes', () => {
    expect(randomIndex(10, (i) => i === 3, () => 0.95)).toBe(3);
    expect(randomIndex(10, () => false)).toBe(-1);
    expect(randomIndex(0, () => true)).toBe(-1);
  });
});

import { describe, expect, it } from 'vitest';
import { filterNeighbours, normalize, parseUrl, searchNames, serializeUrl, validateIds, validateRelease, withIds, validateReleasePointer, validateShard } from './lib.js';

const names = { names: ['Café Lantern', 'Blue Door', 'Green Thread // Thread Unbound'], aliases: [['Thread Unbound', 2]] };
const ids = { ids: ['a', 'b', 'c'] };
const release = { schema: 'mtgemb-release-v2', n_cards: 3, shard_size: 2, k: 2 };

describe('search and ranking helpers', () => {
  it('adds IDs to open suggestions without reopening a dismissed list', () => {
    const open = searchNames('thread', names);
    expect(open[0].id).toBe('');
    expect(withIds(open, ids.ids)).toEqual([{ index: 2, id: 'c', name: 'Green Thread // Thread Unbound' }]);
    expect(withIds([], ids.ids)).toEqual([]);
  });
  it('normalizes accents and searches prefixes before substring matches', () => {
    expect(normalize(' CAFÉ ')).toBe('cafe');
    expect(searchNames('thread unbound', names, ids.ids)[0]).toMatchObject({ index: 2, id: 'c' });
    expect(searchNames('thread', names, ids.ids).map((entry) => entry.id)).toEqual(['c']);
    const order = { names: ['The Silver Lamp', 'Silver Lamp Keeper'], aliases: [] };
    expect(searchNames('silver lamp', order).map((entry) => entry.index)).toEqual([1, 0]);
    expect(searchNames('silver lamp', order).map((entry) => entry.id)).toEqual(['', '']);
  });

  it('filters only saved neighbours using the selected constraints', () => {
    const cards = [{ color_identity: 'U', mv: 2, type_line: 'Instant', is_creature: false }, { color_identity: '', mv: 4, type_line: 'Artifact', is_creature: false }];
    expect(filterNeighbours([[1, 0.8], [2, 0.7]], (index) => cards[index - 1], { color: 'C', minMv: 3, maxMv: null, creature: 'noncreature', type: 'arti' })).toEqual([[2, 0.7]]);
  });
});

describe('URL state', () => {
  it('round trips card and filters', () => {
    const state = { card: 'oracle-123', color: 'G', minMv: 2, maxMv: null, creature: 'creature', type: 'human' };
    expect(parseUrl(`?${serializeUrl(state)}`)).toEqual({ ...state, maxMv: null });
  });
});

describe('bundle validation', () => {
  const cards = [
    { id: 'a', faces: [{ name: 'A' }], color_identity: '', mv: 1, type_line: 'Artifact', is_creature: false, has_text: true },
    { id: 'b', faces: [{ name: 'B' }], color_identity: 'U', mv: 2, type_line: 'Creature', is_creature: true, has_text: true }
  ];

  it('validates names without waiting for the separate ID list', () => {
    expect(validateRelease(release, names)).toBe(true);
    expect(validateRelease(release)).toBe(true);
    for (const bad of [null, [], 'names', 0]) expect(() => validateRelease(release, bad)).toThrow(/missing or malformed/);
    expect(() => validateRelease(release, { ...names, names: ['short'] })).toThrow(/counts/);
    expect(() => validateRelease(release, { ...names, ids: ids.ids })).toThrow(/counts/);
  });

  it('rejects a v1 schema', () => {
    expect(() => validateRelease({ ...release, schema: 'mtgemb-release-v1' }, names)).toThrow(/Unsupported release schema.*mtgemb-release-v2/);
  });

  it('rejects an ID list with the wrong count, duplicates, or non-ascending values', () => {
    expect(validateIds(release, ids)).toBe(true);
    expect(() => validateIds(release, { ids: ['a', 'b'] })).toThrow(/count/);
    expect(() => validateIds(release, { ids: ['a', 'a', 'c'] })).toThrow(/unique and strictly ascending/);
    expect(() => validateIds(release, { ids: ['b', 'a', 'c'] })).toThrow(/unique and strictly ascending/);
  });

  it('rejects a current pointer that names a different release', () => {
    expect(validateReleasePointer({ release_id: 'release-1' }, { release_id: 'release-1' })).toBe(true);
    expect(() => validateReleasePointer({ release_id: 'release-2' }, { release_id: 'release-1' })).toThrow(/does not match/);
  });

  it('checks shard lengths and neighbour index ranges', () => {
    expect(validateShard(cards, [[[1, 0.9]], [[0, 0.8]]], release, 0, ['a', 'b'])).toBe(true);
    expect(() => validateShard(cards, [[]], release, 0)).toThrow(/length/);
    expect(() => validateShard(cards, [[[3, 0.9]], [[0, 0.8]]], release, 0)).toThrow(/index/);
    expect(() => validateShard([cards[1], cards[0]], [[[1, 0.9]], [[0, 0.8]]], release, 0, ['a', 'b'])).toThrow(/Card ID does not match/);
  });
});

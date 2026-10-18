import { describe, expect, it } from 'vitest';
import { filterNeighbours, normalize, parseUrl, resolveAbilityIndex, searchNames, serializeUrl, validateIds, validateRelease, validateTags, withIds, validateReleasePointer, validateShard } from './lib.js';

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
    expect(parseUrl(`?${serializeUrl(state)}`)).toEqual({ ...state, ability: null, rank: 'whole', maxMv: null });
  });
  it('round trips an ability index and falls back to Whole card for invalid indexes', () => {
    const state = { card: 'oracle-123', ability: 1, color: '', minMv: null, maxMv: null, creature: '', type: '' };
    expect(parseUrl(`?${serializeUrl(state)}`).ability).toBe(1);
    expect(parseUrl('?card=oracle-123&ability=bad').ability).toBeNull();
    expect(parseUrl('?card=oracle-123&ability=-1').ability).toBeNull();
    expect(resolveAbilityIndex(1, ['one', 'two'])).toBe(1);
    expect(resolveAbilityIndex(2, ['one', 'two'])).toBeNull();
  });
  it('round trips exploratory line ranking', () => {
    const state = { card: 'oracle-123', rank: 'line' };
    expect(parseUrl(`?${serializeUrl(state)}`).rank).toBe('line');
    expect(parseUrl('?card=oracle-123').rank).toBe('whole');
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

  it('validates v3 ability manifests, alignment, text and neighbours', () => {
    const v3 = {
      schema: 'mtgemb-release-v3', n_cards: 2, shard_size: 2, k: 2, ability_k: 50,
      abilities_status: 'exploratory: not evaluated on the tag proxy',
      files: Object.fromEntries(['names.json', 'ids.json', 'cards/000.json', 'neighbours/000.json', 'abilities/000.json'].map((file) => [file, { sha256: 'a'.repeat(64), bytes: 1 }]))
    };
    const v3Names = { names: ['Multi', 'Blank'], aliases: [] };
    const v3Cards = [
      { id: 'a', faces: [{ oracle_text: 'Draw a card.\nGain 1 life.' }], color_identity: '', mv: 1, type_line: 'Artifact', is_creature: false, has_text: true },
      { id: 'b', faces: [{ oracle_text: '' }], color_identity: '', mv: 1, type_line: 'Artifact', is_creature: false, has_text: false }
    ];
    const v3Neighbours = [[[1, 0.8]], []];
    const v3Abilities = [
      [
        { text: 'Draw a card.', neighbours: [[1, 0.9]] },
        { text: 'Gain 1 life.', neighbours: [[1, 0.8]] }
      ],
      []
    ];
    expect(validateRelease(v3, v3Names)).toBe(true);
    expect(validateShard(v3Cards, v3Neighbours, v3, 0, ['a', 'b'], v3Abilities)).toBe(true);
    const missingManifest = { ...v3, files: { ...v3.files } };
    delete missingManifest.files['abilities/000.json'];
    expect(() => validateRelease(missingManifest, v3Names)).toThrow(/abilities\/000.json/);
    expect(() => validateShard(v3Cards, v3Neighbours, v3, 0, ['a', 'b'], [v3Abilities[0]])).toThrow(/Ability shard.*length/);
    expect(() => validateShard(v3Cards, v3Neighbours, v3, 0, ['a', 'b'], [[{ ...v3Abilities[0][0], neighbours: [[0, 0.9]] }, v3Abilities[0][1]], []])).toThrow(/Ability neighbour index or score/);
    expect(() => validateShard(v3Cards, v3Neighbours, v3, 0, ['a', 'b'], [[{ ...v3Abilities[0][0], neighbours: [[2, 0.9]] }, v3Abilities[0][1]], []])).toThrow(/Ability neighbour index or score/);
    expect(() => validateShard(v3Cards, v3Neighbours, v3, 0, ['a', 'b'], [[v3Abilities[0][0]], []])).toThrow(/does not match/);
    expect(() => validateShard(v3Cards, v3Neighbours, v3, 0, ['a', 'b'], [v3Abilities[0], [{ text: 'Invented line.', neighbours: [] }]])).toThrow(/Textless card/);
    // exporter lines are the reminder-free model text, so they need not equal oracle_text lines
    const reminder = [{ ...v3Cards[0], faces: [{ oracle_text: 'Flying (This creature can block.)\n(Reminder only.)\nDraw a card.' }] }, v3Cards[1]];
    const modelLines = [[{ text: 'Flying', neighbours: [[1, 0.9]] }, { text: 'Draw a card.', neighbours: [] }], []];
    expect(validateShard(reminder, v3Neighbours, v3, 0, ['a', 'b'], modelLines)).toBe(true);
  });

  it('validates v4 tags, ability tag subsets, and line neighbours while retaining v3', () => {
    const files = ['names.json', 'ids.json', 'tags.json', 'cards/000.json', 'neighbours/000.json', 'abilities/000.json', 'lines/000.json'];
    const v4 = {
      schema: 'mtgemb-release-v4', n_cards: 2, shard_size: 2, k: 2, ability_k: 50,
      abilities_status: 'exploratory', line_k: 50, line_status: 'exploratory line matches', tags_status: 'proxy tags',
      files: Object.fromEntries(files.map((file) => [file, { sha256: 'a'.repeat(64), bytes: 1 }]))
    };
    const v4Names = { names: ['Multi', 'Blank'], aliases: [] };
    const v4Tags = { labels: ['draw', 'life'] };
    const v4Cards = [
      { ...cards[0], faces: [{ oracle_text: 'Draw a card.\nGain 1 life.' }], tags: [0] },
      { ...cards[1], has_text: false, tags: [] }
    ];
    const v4Neighbours = [[[1, 0.8]], []];
    const v4Abilities = [[{ text: 'Draw a card.', neighbours: [[1, 0.9]], tags: [0] }, { text: 'Gain 1 life.', neighbours: [], tags: [0] }], []];
    const v4Lines = [[[1, 0.7]], []];
    expect(validateRelease(v4, v4Names)).toBe(true);
    expect(validateTags(v4, v4Tags)).toBe(true);
    expect(validateShard(v4Cards, v4Neighbours, v4, 0, ['a', 'b'], v4Abilities, v4Lines, v4Tags)).toBe(true);
    for (const missing of ['tags.json', 'lines/000.json']) {
      const manifest = { ...v4, files: { ...v4.files } };
      delete manifest.files[missing];
      expect(() => validateRelease(manifest, v4Names)).toThrow(new RegExp(missing.replace('.', '\\.')));
    }
    expect(() => validateShard([{ ...v4Cards[0], tags: [2] }, v4Cards[1]], v4Neighbours, v4, 0, ['a', 'b'], v4Abilities, v4Lines, v4Tags)).toThrow(/tag index/);
    expect(() => validateShard(v4Cards, v4Neighbours, v4, 0, ['a', 'b'], [[{ ...v4Abilities[0][0], tags: [1] }, v4Abilities[0][1]], []], v4Lines, v4Tags)).toThrow(/subset/);
    expect(() => validateShard(v4Cards, v4Neighbours, v4, 0, ['a', 'b'], v4Abilities, [[[0, 0.7]], []], v4Tags)).toThrow(/Line neighbour index or score/);
  });
});

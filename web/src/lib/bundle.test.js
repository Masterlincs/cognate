import { describe, expect, it } from 'vitest';
import { filterNeighbours, normalize, parseUrl, resolveAbilityIndex, searchKeys, serializeUrl, validateIds, validateRelease, validateTags, validateReleasePointer, validateShard } from './bundle.js';

const names = { names: ['Café Lantern', 'Blue Door', 'Green Thread // Thread Unbound'], aliases: [['Thread Unbound', 2]] };
const ids = { ids: ['a', 'b', 'c'] };
const release = { schema: 'mtgemb-release-v2', n_cards: 3, shard_size: 2, k: 2 };

describe('search and ranking helpers', () => {
  it('normalizes accents and case', () => {
    expect(normalize(' CAFÉ ')).toBe('cafe');
  });

  // the first search normalises every name once (it took about 0.6 s on the throttled mobile profile)
  it('normalises a names list once and reuses it', () => {
    const list = { names: ['Café Lantern', 'Blue Door'], aliases: [['Lantern', 0]] };
    const keys = searchKeys(list);
    expect(keys).toEqual([['cafe lantern', 'lantern'], ['blue door']]);
    expect(searchKeys(list)).toBe(keys);
    expect(searchKeys({ ...list })).not.toBe(keys);
  });

  it('filters only saved neighbours using the selected constraints', () => {
    const cards = [{ color_identity: 'U', mv: 2, type_line: 'Instant', is_creature: false }, { color_identity: '', mv: 4, type_line: 'Artifact', is_creature: false }];
    expect(filterNeighbours([[1, 0.8], [2, 0.7]], (index) => cards[index - 1], { color: 'C', minMv: 3, maxMv: null, creature: 'noncreature', type: 'arti' })).toEqual([[2, 0.7]]);
  });
  it('keeps order with no filters and drops cards not loaded yet', () => {
    const cards = { 1: { color_identity: 'R', mv: 1, type_line: 'Instant', is_creature: false }, 2: { color_identity: 'G', mv: 3, type_line: 'Creature — Elf', is_creature: true } };
    const saved = [[2, 0.9], [1, 0.8]];
    expect(filterNeighbours(saved, (i) => cards[i], {})).toEqual(saved);
    expect(filterNeighbours(saved, (i) => cards[i], { creature: 'creature', maxMv: 2 })).toEqual([]);
    expect(filterNeighbours([[5, 0.5]], () => undefined, {})).toEqual([]);
  });
});

describe('URL state', () => {
  it('round trips card and filters', () => {
    const state = { card: 'oracle-123', color: 'G', minMv: 2, maxMv: null, creature: 'creature', type: 'human' };
    expect(parseUrl(`?${serializeUrl(state)}`)).toEqual({ ...state, colorMode: 'include', colorBy: 'identity', view: 'list', ability: null, rank: 'whole', maxMv: null, dropped: [] });
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
  it('round trips face ranking and drops it for bundles without face matches', () => {
    expect(parseUrl(`?${serializeUrl({ card: 'oracle-123', rank: 'face' })}`).rank).toBe('face');
    expect(parseUrl('?rank=unknown')).toMatchObject({ rank: 'whole', dropped: [] });
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
    for (const bad of [null, [], 'names', 0]) expect(() => validateRelease(release, bad)).toThrow(/missing or not in the correct format/);
    expect(() => validateRelease(release, { ...names, names: ['short'] })).toThrow(/does not match the card count/);
    expect(() => validateRelease(release, { ...names, ids: ids.ids })).toThrow(/does not match the card count/);
  });

  it('rejects a v1 schema', () => {
    expect(() => validateRelease({ ...release, schema: 'mtgemb-release-v1' }, names)).toThrow(/cannot read \(mtgemb-release-v1\).*mtgemb-release-v2/);
  });

  it('rejects an ID list with the wrong count, duplicates, or non-ascending values', () => {
    expect(validateIds(release, ids)).toBe(true);
    expect(() => validateIds(release, { ids: ['a', 'b'] })).toThrow(/count/);
    expect(() => validateIds(release, { ids: ['a', 'a', 'c'] })).toThrow(/unique and in ascending order/);
    expect(() => validateIds(release, { ids: ['b', 'a', 'c'] })).toThrow(/unique and in ascending order/);
  });

  it('rejects a current pointer that names a different release', () => {
    expect(validateReleasePointer({ release_id: 'release-1' }, { release_id: 'release-1' })).toBe(true);
    expect(() => validateReleasePointer({ release_id: 'release-2' }, { release_id: 'release-1' })).toThrow(/does not match/);
  });

  it('checks shard lengths and neighbour index ranges', () => {
    expect(validateShard(cards, [[[1, 0.9]], [[0, 0.8]]], release, 0, ['a', 'b'])).toBe(true);
    expect(() => validateShard(cards, [[]], release, 0)).toThrow(/wrong number of cards/);
    expect(() => validateShard(cards, [[[3, 0.9]], [[0, 0.8]]], release, 0)).toThrow(/card number or score/);
    expect(() => validateShard([cards[1], cards[0]], [[[1, 0.9]], [[0, 0.8]]], release, 0, ['a', 'b'])).toThrow(/ID of card 0 does not match/);
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
    expect(() => validateShard(v3Cards, v3Neighbours, v3, 0, ['a', 'b'], [v3Abilities[0]])).toThrow(/ability matches has the wrong number of cards/);
    expect(() => validateShard(v3Cards, v3Neighbours, v3, 0, ['a', 'b'], [[{ ...v3Abilities[0][0], neighbours: [[0, 0.9]] }, v3Abilities[0][1]], []])).toThrow(/ability match .*card number or score/);
    expect(() => validateShard(v3Cards, v3Neighbours, v3, 0, ['a', 'b'], [[{ ...v3Abilities[0][0], neighbours: [[2, 0.9]] }, v3Abilities[0][1]], []])).toThrow(/ability match .*card number or score/);
    expect(() => validateShard(v3Cards, v3Neighbours, v3, 0, ['a', 'b'], [[v3Abilities[0][0]], []])).toThrow(/does not match/);
    expect(() => validateShard(v3Cards, v3Neighbours, v3, 0, ['a', 'b'], [v3Abilities[0], [{ text: 'Invented line.', neighbours: [] }]])).toThrow(/has no rules text/);
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
    expect(() => validateShard([{ ...v4Cards[0], tags: [2] }, v4Cards[1]], v4Neighbours, v4, 0, ['a', 'b'], v4Abilities, v4Lines, v4Tags)).toThrow(/tag number/);
    expect(() => validateShard(v4Cards, v4Neighbours, v4, 0, ['a', 'b'], [[{ ...v4Abilities[0][0], tags: [1] }, v4Abilities[0][1]], []], v4Lines, v4Tags)).toThrow(/tags that the card does not have/);
    expect(() => validateShard(v4Cards, v4Neighbours, v4, 0, ['a', 'b'], v4Abilities, [[[0, 0.7]], []], v4Tags)).toThrow(/line matches .*card number or score/);
  });
});

describe('parseUrl drops invalid filter values', () => {
  it('ignores unknown colours, creature modes and non-numeric mana values', () => {
    const s = parseUrl('?card=x&color=purple&creature=maybe&min=abc&max=3');
    expect(s).toMatchObject({ card: 'x', color: '', creature: '', minMv: null, maxMv: 3 });
  });
  it('keeps valid ones', () => {
    expect(parseUrl('?color=C&creature=noncreature&min=0')).toMatchObject({ color: 'C', creature: 'noncreature', minMv: 0 });
  });
});

describe('colour combinations', () => {
  const ci = ['', 'R', 'WR', 'WUR', 'U'];
  const run = (color, colorMode) => filterNeighbours(ci.map((_, i) => [i, 1]), (i) => ({ color_identity: ci[i], mv: 1, type_line: '', is_creature: false }), { color, colorMode }).map(([i]) => ci[i]);
  it('include: has every picked colour', () => expect(run('WR', 'include')).toEqual(['WR', 'WUR']));
  it('exact: exactly the picked colours', () => expect(run('WR', 'exact')).toEqual(['WR']));
  it('within: no colour outside the picked ones (colourless counts)', () => expect(run('WR', 'within')).toEqual(['', 'R', 'WR']));
  it('C means colourless in every mode', () => expect(run('C', 'within')).toEqual(['']));
  it('defaults to include', () => expect(run('R')).toEqual(['R', 'WR', 'WUR']));
});

describe('colour URL state', () => {
  it('canonicalises colour order and round trips the mode', () => {
    const s = parseUrl('?color=rw&cm=exact');
    expect(s).toMatchObject({ color: 'WR', colorMode: 'exact', dropped: [] });
    expect(serializeUrl(s)).toBe('color=WR&cm=exact');
  });
  it('omits the default mode and the mode for colourless', () => {
    expect(serializeUrl({ color: 'G', colorMode: 'include' })).toBe('color=G');
    expect(serializeUrl({ color: 'C', colorMode: 'exact' })).toBe('color=C');
  });
  it('reports dropped filters by parameter name', () => {
    expect(parseUrl('?color=WW&cm=sideways&min=abc&creature=maybe').dropped).toEqual(['color', 'cm', 'min', 'creature']);
    expect(parseUrl('?color=CW').dropped).toEqual(['color']);
    expect(parseUrl('?min=&type=elf').dropped).toEqual([]);
    // a negative bound hides every card, so it is rejected like any other bad value
    expect(parseUrl('?min=0&max=-1')).toMatchObject({ minMv: 0, maxMv: null, dropped: ['max'] });
  });
});

describe('colour or colour identity', () => {
  // a red card with a white symbol in its text, a devoid (colourless) red card, an old-bundle card without colours
  const cards = [{ color_identity: 'WR', colors: 'R' }, { color_identity: 'R', colors: '' }, { color_identity: 'G' }].map((c) => ({ mv: 1, type_line: '', is_creature: false, ...c }));
  const run = (f) => filterNeighbours(cards.map((_, i) => [i, 1]), (i) => cards[i], f).map(([i]) => i);
  it('identity is the default basis', () => expect(run({ color: 'W' })).toEqual([0]));
  it('colour uses the printed colours', () => {
    expect(run({ color: 'W', colorBy: 'color' })).toEqual([]);
    expect(run({ color: 'R', colorBy: 'color', colorMode: 'exact' })).toEqual([0]);
    expect(run({ color: 'C', colorBy: 'color' })).toEqual([1]);
  });
  it('falls back to identity for a card without colours', () => expect(run({ color: 'G', colorBy: 'color' })).toEqual([2]));
  it('round trips the basis and the view, omitting defaults', () => {
    const s = parseUrl('?color=W&cb=color&view=grid');
    expect(s).toMatchObject({ color: 'W', colorBy: 'color', view: 'grid', dropped: [] });
    expect(serializeUrl(s)).toBe('color=W&cb=color&view=grid');
    expect(serializeUrl({ color: 'W', colorBy: 'identity', view: 'list' })).toBe('color=W');
    expect(serializeUrl({ colorBy: 'color' })).toBe('');
  });
  it('drops a bad basis and ignores a bad view', () => {
    expect(parseUrl('?color=W&cb=paint&view=wall')).toMatchObject({ colorBy: 'identity', view: 'list', dropped: ['cb'] });
  });
});

describe('v5 bundles', () => {
  const files = ['names.json', 'ids.json', 'tags.json', 'cards/000.json', 'neighbours/000.json', 'abilities/000.json', 'lines/000.json'];
  const v5 = {
    schema: 'mtgemb-release-v5', n_cards: 2, shard_size: 2, k: 200, ability_k: 50,
    abilities_status: 'exploratory', line_k: 200, line_status: 'exploratory', tags_status: 'proxy',
    files: Object.fromEntries(files.map((file) => [file, { sha256: 'a'.repeat(64), bytes: 1 }]))
  };
  const base = { id: 'a', faces: [{ oracle_text: 'Draw a card.' }], color_identity: 'U', mv: 1, type_line: 'Instant', is_creature: false, has_text: true, tags: [], colors: 'U', image_back: null };
  const shard = (a, b) => validateShard([a, { ...base, id: 'b', ...b }], [[], []], v5, 0, ['a', 'b'], undefined, [[], []], { labels: ['x'] });
  it('accepts k and line_k of 200 with colours and back images', () => {
    expect(validateRelease(v5, { names: ['A', 'B'], aliases: [] })).toBe(true);
    expect(shard(base, { colors: '', image_back: 'https://cards.scryfall.io/normal/back/x.jpg' })).toBe(true);
  });
  it('rejects a line_k that differs from k', () => expect(() => validateRelease({ ...v5, line_k: 50 })).toThrow(/line or tag settings/));
  it('rejects a k other than 200, even with a matching line_k', () => {
    expect(() => validateRelease({ ...v5, k: 0, line_k: 0 })).toThrow(/k/);
    expect(() => validateRelease({ ...v5, k: 50, line_k: 50 })).toThrow(/k/);
    expect(() => validateRelease({ ...v5, k: undefined })).toThrow(/k/);
  });
  it('rejects malformed colours or back images', () => {
    expect(() => shard(base, { colors: 'RW' })).toThrow(/not in the correct format/);
    expect(() => shard(base, { colors: undefined })).toThrow(/not in the correct format/);
    expect(() => shard(base, { image_back: 3 })).toThrow(/not in the correct format/);
  });
});

describe('v6 face match bundles', () => {
  const files = ['names.json', 'ids.json', 'tags.json', 'cards/000.json', 'neighbours/000.json', 'abilities/000.json', 'lines/000.json', 'faces/000.json'];
  const v6 = { schema: 'mtgemb-release-v6', n_cards: 2, shard_size: 2, k: 200, ability_k: 50, line_k: 200, face_k: 200,
    abilities_status: 'exploratory', line_status: 'exploratory', face_status: 'exploratory face matches', tags_status: 'proxy',
    files: Object.fromEntries(files.map((file) => [file, { sha256: 'a'.repeat(64), bytes: 1 }])) };
  const base = { id: 'a', faces: [{ oracle_text: 'Draw a card.' }], color_identity: 'U', mv: 1, type_line: 'Instant', is_creature: false, has_text: true, tags: [], colors: 'U', image_back: null };
  it('requires v6 face metadata and a manifest entry for each face shard', () => {
    expect(validateRelease(v6)).toBe(true);
    expect(() => validateRelease({ ...v6, face_k: 50 })).toThrow(/face/);
    expect(() => validateRelease({ ...v6, face_status: undefined })).toThrow(/face/);
    const missing = { ...v6, files: { ...v6.files } };
    delete missing.files['faces/000.json'];
    expect(() => validateRelease(missing)).toThrow(/faces\/000.json/);
  });
  it('validates face match lists with v5 line-list rules', () => {
    const cards = [base, { ...base, id: 'b', has_text: false }];
    const schema = { ...v6, n_cards: 3 };
    const check = (faces) => validateShard(cards, [[], []], schema, 0, ['a', 'b'], undefined, [[], []], { labels: ['x'] }, faces);
    expect(check([[[1, 0.9], [2, 0.9]], []])).toBe(true); // tied scores are valid
    const tooMany = Array.from({ length: 201 }, () => [1, 0.1]);
    for (const bad of [[[[0, 0.9]], []], [[[3, 0.9]], []], [[[1, Infinity]], []], [[[1, 0.8], [2, 0.9]], []], [[[1, 0.8]], [[0, 0.1]]], [tooMany, []]]) {
      expect(() => check(bad)).toThrow();
    }
  });
});

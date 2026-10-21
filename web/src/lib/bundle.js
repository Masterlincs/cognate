export function normalize(value) {
  return value.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLocaleLowerCase().trim();
}

export function searchNames(query, names, ids = null, limit = 12) {
  const q = normalize(query);
  if (!q) return [];
  const exacts = [];
  const prefixes = [];
  const substrings = [];
  const aliases = Array.from({ length: names.names.length }, () => []);
  for (const [name, index] of names.aliases) aliases[index].push(name);
  for (let i = 0; i < names.names.length; i += 1) {
    const strings = [names.names[i], ...aliases[i]].map(normalize);
    if (strings.some((s) => s === q)) exacts.push(i);
    else if (strings.some((s) => s.startsWith(q))) prefixes.push(i);
    else if (strings.some((s) => s.includes(q))) substrings.push(i);
  }
  return [...exacts, ...prefixes, ...substrings].slice(0, limit).map((index) => ({ index, id: ids?.[index] ?? '', name: names.names[index] }));
}

// C = colourless; otherwise picked is a WUBRG subset matched as include (has all picked),
// exact (the same colours) or within (no colour outside picked; colourless cards pass)
export function colourMatch(identity, picked, mode = 'include') {
  if (picked === 'C') return identity === '';
  const has = (c) => identity.includes(c);
  if (mode === 'within') return [...identity].every((c) => picked.includes(c));
  return [...picked].every(has) && (mode !== 'exact' || identity.length === picked.length);
}

export function filterNeighbours(entries, cardByIndex, filters) {
  const query = normalize(filters.type ?? '');
  return entries.filter(([index]) => {
    const card = cardByIndex(index);
    if (!card) return false;
    if (filters.color && !colourMatch(card.color_identity, filters.color, filters.colorMode)) return false;
    if (Number.isFinite(filters.minMv) && card.mv < filters.minMv) return false;
    if (Number.isFinite(filters.maxMv) && card.mv > filters.maxMv) return false;
    if (query && !normalize(card.type_line).includes(query)) return false;
    if (filters.creature === 'creature' && !card.is_creature) return false;
    if (filters.creature === 'noncreature' && card.is_creature) return false;
    return true;
  });
}

export function parseUrl(search) {
  const params = new URLSearchParams(search);
  // a hand-edited link with a bad value drops that filter rather than hiding every result; dropped lists them
  const dropped = [];
  const given = (key) => params.get(key) !== null && params.get(key) !== '';
  const bad = (key, fallback) => { dropped.push(key); return fallback; };
  const num = (key) => !given(key) ? null : Number.isFinite(Number(params.get(key))) ? Number(params.get(key)) : bad(key, null);
  const oneOf = (key, allowed, fallback = '') => !given(key) ? fallback : allowed.includes(params.get(key)) ? params.get(key) : bad(key, fallback);
  const raw = (params.get('color') ?? '').toUpperCase();
  const color = !raw ? '' : raw === 'C' ? 'C' : /^[WUBRG]+$/.test(raw) && new Set(raw).size === raw.length ? [...'WUBRG'].filter((c) => raw.includes(c)).join('') : bad('color', '');
  const colorMode = oneOf('cm', ['include', 'exact', 'within'], 'include');
  const ability = Number(params.get('ability'));
  const minMv = num('min'), maxMv = num('max');
  return { card: params.get('card') || '', ability: given('ability') && Number.isInteger(ability) && ability >= 0 ? ability : null, rank: params.get('rank') === 'line' ? 'line' : 'whole', color, colorMode, minMv, maxMv, creature: oneOf('creature', ['creature', 'noncreature']), type: params.get('type') || '', dropped };
}

export function serializeUrl(state) {
  const params = new URLSearchParams();
  if (state.card) params.set('card', state.card);
  if (Number.isInteger(state.ability) && state.ability >= 0) params.set('ability', state.ability);
  if (state.rank === 'line') params.set('rank', 'line');
  if (state.color) params.set('color', state.color);
  if (state.color && state.color !== 'C' && state.colorMode && state.colorMode !== 'include') params.set('cm', state.colorMode);
  if (Number.isFinite(state.minMv)) params.set('min', state.minMv);
  if (Number.isFinite(state.maxMv)) params.set('max', state.maxMv);
  if (state.creature) params.set('creature', state.creature);
  if (state.type) params.set('type', state.type);
  return params.toString();
}

export function resolveAbilityIndex(index, abilities) {
  return Number.isInteger(index) && index >= 0 && index < abilities.length ? index : null;
}

export function validateRelease(release, names) {
  if (!['mtgemb-release-v2', 'mtgemb-release-v3', 'mtgemb-release-v4'].includes(release?.schema)) throw new Error(`Unsupported release schema: ${release?.schema ?? 'missing'}; expected mtgemb-release-v2, mtgemb-release-v3 or mtgemb-release-v4.`);
  if (!Number.isInteger(release.n_cards) || release.n_cards < 1) throw new Error('Invalid card count in release.json.');
  if (!Number.isInteger(release.shard_size) || release.shard_size < 1) throw new Error('Invalid shard_size.');
  const shards = Math.ceil(release.n_cards / release.shard_size);
  const v3plus = ['mtgemb-release-v3', 'mtgemb-release-v4'].includes(release.schema);
  const v4 = release.schema === 'mtgemb-release-v4';
  const expectedFiles = ['names.json', 'ids.json', ...(v4 ? ['tags.json'] : []), ...Array.from({ length: shards }, (_, i) => [`cards/${String(i).padStart(3, '0')}.json`, `neighbours/${String(i).padStart(3, '0')}.json`, ...(v3plus ? [`abilities/${String(i).padStart(3, '0')}.json`] : []), ...(v4 ? [`lines/${String(i).padStart(3, '0')}.json`] : [])]).flat()];
  if (release.schema === 'mtgemb-release-v3' && (release.ability_k !== 50 || release.abilities_status !== 'exploratory: not evaluated on the tag proxy')) throw new Error('Invalid ability metadata in release.json.');
  if (v4 && (!Number.isInteger(release.ability_k) || release.ability_k !== 50 || typeof release.abilities_status !== 'string' || !Number.isInteger(release.line_k) || release.line_k !== 50 || typeof release.line_status !== 'string' || typeof release.tags_status !== 'string')) throw new Error('Invalid v4 line or tag metadata in release.json.');
  if (v3plus) for (const file of expectedFiles) {
    const entry = release.files?.[file];
    if (!entry || !/^[a-f0-9]{64}$/.test(entry.sha256) || !Number.isInteger(entry.bytes) || entry.bytes < 0) throw new Error(`Release manifest is missing or malformed for ${file}.`);
  }
  if (names === undefined) return true;  // release-only check before names.json arrives
  if (!names || typeof names !== 'object' || Array.isArray(names)) throw new Error('Name index is missing or malformed.');
  if (!Array.isArray(names.names) || names.names.length !== release.n_cards || 'ids' in names) throw new Error('Name index counts do not match release.json or contains IDs.');
  if (!Array.isArray(names.aliases)) throw new Error('Name aliases are missing or malformed.');
  for (const [name, index] of names.aliases) if (typeof name !== 'string' || !Number.isInteger(index) || index < 0 || index >= release.n_cards) throw new Error('Name alias index is outside the card catalogue.');
  return true;
}

export function validateTags(release, tags) {
  if (release?.schema !== 'mtgemb-release-v4') return true;
  if (!Array.isArray(tags?.labels) || tags.labels.some((label) => typeof label !== 'string' || !label.trim())) throw new Error('Tag labels are missing or malformed.');
  return true;
}

// fill IDs into the open suggestions only; a dismissed (empty) list stays dismissed
export function withIds(suggestions, ids) {
  return suggestions.map((entry) => ({ ...entry, id: ids?.[entry.index] ?? '' }));
}

export function validateIds(release, data) {
  const ids = data?.ids;
  if (!Array.isArray(ids) || ids.length !== release?.n_cards) throw new Error('ID count does not match release.json.');
  if (ids.some((id, index) => typeof id !== 'string' || !id || (index > 0 && ids[index - 1] >= id))) throw new Error('IDs must be unique and strictly ascending.');
  return true;
}

export function validateReleasePointer(current, release) {
  if (!current?.release_id || current.release_id !== release?.release_id) throw new Error('Release pointer does not match release.json.');
  return true;
}

export function validateShard(cards, neighbours, release, shard, ids, abilities, lines, tags) {
  validateCommonShard(cards, neighbours, release, shard, ids);
  if (release.schema === 'mtgemb-release-v4') {
    validateTags(release, tags);
    for (let offset = 0; offset < cards.length; offset += 1) {
      const index = shard * release.shard_size + offset;
      validateTagIndices(cards[offset].tags, tags.labels.length, `Card ${index}`);
      if (!cards[offset].has_text && cards[offset].tags.length) throw new Error(`Textless card ${index} must have no tags.`);
    }
    if (lines !== undefined) validateLineShard(cards, lines, release, shard);
  }
  if (abilities !== undefined && (release.schema === 'mtgemb-release-v3' || release.schema === 'mtgemb-release-v4')) validateAbilityShard(cards, abilities, release, shard, tags);
  return true;
}

function validateTagIndices(indices, count, label) {
  if (!Array.isArray(indices)) throw new Error(`${label} tags are malformed.`);
  let previous = -1;
  for (const tag of indices) {
    if (!Number.isInteger(tag) || tag < 0 || tag >= count || tag <= previous) throw new Error(`${label} tag index is invalid or not ascending unique.`);
    previous = tag;
  }
  return true;
}

export function validateLineShard(cards, lines, release, shard) {
  const expected = Math.min(release.shard_size, release.n_cards - shard * release.shard_size);
  if (!Array.isArray(lines) || lines.length !== expected) throw new Error(`Line shard ${shard} has the wrong length.`);
  for (let offset = 0; offset < expected; offset += 1) {
    const index = shard * release.shard_size + offset;
    const list = lines[offset];
    if (!Array.isArray(list) || list.length > release.line_k) throw new Error(`Line list ${index} is malformed.`);
    let previous = Infinity;
    for (const pair of list) {
      if (!Array.isArray(pair) || pair.length !== 2) throw new Error(`Line neighbour index or score is invalid for card ${index}.`);
      const [target, score] = pair;
      if (!Number.isInteger(target) || target < 0 || target >= release.n_cards || target === index || typeof score !== 'number' || !Number.isFinite(score)) throw new Error(`Line neighbour index or score is invalid for card ${index}.`);
      if (score > previous) throw new Error(`Line neighbour scores are not descending for card ${index}.`);
      previous = score;
    }
    if (!cards[offset].has_text && list.length) throw new Error(`Textless card ${index} must have no line neighbours.`);
  }
  return true;
}

export function validateCommonShard(cards, neighbours, release, shard, ids) {
  const expected = Math.min(release.shard_size, release.n_cards - shard * release.shard_size);
  if (expected < 0 || !Array.isArray(cards) || cards.length !== expected) throw new Error(`Card shard ${shard} has the wrong length.`);
  if (!Array.isArray(neighbours) || neighbours.length !== expected) throw new Error(`Neighbour shard ${shard} has the wrong length.`);
  for (let offset = 0; offset < expected; offset += 1) {
    const card = cards[offset];
    const index = shard * release.shard_size + offset;
    if (!card || typeof card.id !== 'string' || card.id !== undefined && card.id.length === 0 || !Array.isArray(card.faces) || typeof card.color_identity !== 'string' || typeof card.mv !== 'number' || typeof card.type_line !== 'string' || typeof card.is_creature !== 'boolean' || typeof card.has_text !== 'boolean') throw new Error(`Card ${index} is malformed.`);
    if (ids && card.id !== ids[index]) throw new Error(`Card ID does not match ID index at ${index}.`);
    const list = neighbours[offset];
    if (!Array.isArray(list) || list.length > release.k) throw new Error(`Neighbour list ${index} is malformed.`);
    let previous = Infinity;
    for (const pair of list) {
      const [target, score] = pair ?? [];
      if (!Number.isInteger(target) || target < 0 || target >= release.n_cards || target === index || typeof score !== 'number' || !Number.isFinite(score)) throw new Error(`Neighbour index or score is invalid for card ${index}.`);
      if (score > previous) throw new Error(`Neighbour scores are not descending for card ${index}.`);
      previous = score;
    }
    if (!card.has_text && list.length) throw new Error(`Textless card ${index} must have no neighbours.`);
  }
  return true;
}

export function validateAbilityShard(cards, abilities, release, shard, tags) {
  const expected = Math.min(release.shard_size, release.n_cards - shard * release.shard_size);
  if (!Array.isArray(cards) || cards.length !== expected || !Array.isArray(abilities) || abilities.length !== expected) throw new Error(`Ability shard ${shard} has the wrong length.`);
  for (let offset = 0; offset < expected; offset += 1) {
    const card = cards[offset];
    const index = shard * release.shard_size + offset;
    const entries = abilities[offset];
    if (!Array.isArray(entries)) throw new Error(`Ability list ${index} is malformed.`);
    if (!card.has_text && entries.length) throw new Error(`Textless card ${index} must have no ability entries.`);
    // Lines come from the model text (reminder text removed), not oracle_text; the exporter
    // pins them by hash, so only the shape is checked here: none, or one entry per ability (>=2).
    if (entries.length === 1) throw new Error(`Ability list ${index} does not match the card's ability lines.`);
    for (const ability of entries) {
      if (!ability || typeof ability.text !== 'string' || !ability.text.trim() || !Array.isArray(ability.neighbours) || ability.neighbours.length > release.ability_k) throw new Error(`Ability entry is malformed for card ${index}.`);
      if (release.schema === 'mtgemb-release-v4') {
        validateTagIndices(ability.tags, tags?.labels?.length ?? 0, `Ability on card ${index}`);
        if (ability.tags.some((tag) => !card.tags?.includes(tag))) throw new Error(`Ability tags are not a subset of card tags for card ${index}.`);
      }
      let previousAbilityScore = Infinity;
      for (const pair of ability.neighbours) {
        const [target, score] = pair ?? [];
        if (!Number.isInteger(target) || target < 0 || target >= release.n_cards || target === index || typeof score !== 'number' || !Number.isFinite(score) || Math.abs(score - Number(score.toFixed(4))) > 1e-9) throw new Error(`Ability neighbour index or score is invalid for card ${index}.`);
        if (score > previousAbilityScore) throw new Error(`Ability neighbour scores are not descending for card ${index}.`);
        previousAbilityScore = score;
      }
    }
  }
  return true;
}

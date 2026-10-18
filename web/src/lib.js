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

export function filterNeighbours(entries, cardByIndex, filters) {
  const query = normalize(filters.type ?? '');
  return entries.filter(([index]) => {
    const card = cardByIndex(index);
    if (!card) return false;
    if (filters.color && (filters.color === 'C' ? card.color_identity !== '' : !card.color_identity.includes(filters.color))) return false;
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
  const num = (key) => params.has(key) && params.get(key) !== '' ? Number(params.get(key)) : null;
  return { card: params.get('card') || '', color: params.get('color') || '', minMv: num('min'), maxMv: num('max'), creature: params.get('creature') || '', type: params.get('type') || '' };
}

export function serializeUrl(state) {
  const params = new URLSearchParams();
  if (state.card) params.set('card', state.card);
  if (state.color) params.set('color', state.color);
  if (state.minMv !== null && state.minMv !== '') params.set('min', state.minMv);
  if (state.maxMv !== null && state.maxMv !== '') params.set('max', state.maxMv);
  if (state.creature) params.set('creature', state.creature);
  if (state.type) params.set('type', state.type);
  return params.toString();
}

export function validateRelease(release, names) {
  if (release?.schema !== 'mtgemb-release-v2') throw new Error(`Unsupported release schema: ${release?.schema ?? 'missing'}; expected mtgemb-release-v2.`);
  if (!Number.isInteger(release.n_cards) || release.n_cards < 1) throw new Error('Invalid card count in release.json.');
  if (!Number.isInteger(release.shard_size) || release.shard_size < 1) throw new Error('Invalid shard_size.');
  if (!names) return true;
  if (!Array.isArray(names.names) || names.names.length !== release.n_cards || 'ids' in names) throw new Error('Name index counts do not match release.json or contains IDs.');
  if (!Array.isArray(names.aliases)) throw new Error('Name aliases are missing or malformed.');
  for (const [name, index] of names.aliases) if (typeof name !== 'string' || !Number.isInteger(index) || index < 0 || index >= release.n_cards) throw new Error('Name alias index is outside the card catalogue.');
  return true;
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

export function validateShard(cards, neighbours, release, shard, ids) {
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

// Release bundle and site-data loading. One promise cache per shard kind; failed loads are
// evicted so a retry can succeed. Shapes are checked with the bundle validators (SCHEMA.md).
import { isV4Plus, isV6, validateAbilityShard, validateCommonShard, validateFaceShard, validateIds, validateLineShard, validateRelease, validateReleasePointer, validateShard, validateTags } from './bundle.js';

let root = '';
let fetcher = (...args) => fetch(...args);
let release = null;
let ids = null;
let tagsReady = null;
let site = null;
const shards = new Map();
const neighbours = new Map();
const abilities = new Map();
const lines = new Map();
const faces = new Map();
const pretrained = new Map();

export const setRoot = (value) => { root = value.replace(/\/$/, ''); };
const pad = (n) => String(n).padStart(3, '0');

async function get(path, kind = 'json', fetchFn = fetcher) {
  const response = await fetchFn(`${root}/data/${path}`);
  if (!response.ok) throw Object.assign(new Error(`Could not load ${path} (${response.status}).`), { status: response.status });
  return kind === 'json' ? response.json() : response.arrayBuffer();
}

function cached(cache, key, make) {
  if (!cache.has(key)) cache.set(key, make().catch((reason) => { cache.delete(key); throw reason; }));
  return cache.get(key);
}

export async function loadRelease(fetchFn = fetcher) {
  fetcher = fetchFn;
  const current = await get('current.json');
  if (!current.release_id || /[/\\]|\.\./.test(current.release_id)) throw new Error('Invalid release pointer.');
  const next = await get(`${current.release_id}/release.json`);
  validateReleasePointer(current, next);
  validateRelease(next);
  // a new deployment between visits must not mix in the old release's cached shards
  if (next.release_id !== release?.release_id) {
    for (const cache of [shards, neighbours, abilities, lines, faces, pretrained]) cache.clear();
    tagsReady = null; ids = null; site = null;
  }
  release = next;
  const names = await get(`${release.release_id}/names.json`);
  validateRelease(release, names);
  // ids load after names so they do not share bandwidth with the search index
  const idsReady = get(`${release.release_id}/ids.json`).then((data) => { validateIds(release, data); ids = data.ids; return ids; });
  return { release, names, idsReady };
}

// tags exist from v4 on; older bundles have none
const getTags = () => (tagsReady ??= (!isV4Plus(release) ? Promise.resolve(undefined) : get(`${release.release_id}/tags.json`))
  .then((tags) => { if (tags) validateTags(release, tags); return tags; })
  .catch((reason) => { tagsReady = null; throw reason; }));

export const shardOf = (index) => Math.floor(index / release.shard_size);

// cards and neighbours load separately: showing a neighbour needs only its card shard
export function getCards(shard) {
  return cached(shards, shard, async () => {
    const [cards, tags] = await Promise.all([get(`${release.release_id}/cards/${pad(shard)}.json`), getTags()]);
    validateShard(cards, cards.map(() => []), release, shard, ids, undefined, undefined, tags);
    return cards;
  });
}

export async function getCard(index) {
  return (await getCards(shardOf(index)))[index % release.shard_size];
}

export async function getNeighbours(index) {
  const shard = shardOf(index);
  const rows = await cached(neighbours, shard, async () => {
    const [cards, rows] = await Promise.all([getCards(shard), get(`${release.release_id}/neighbours/${pad(shard)}.json`)]);
    validateCommonShard(cards, rows, release, shard, ids);
    return rows;
  });
  return rows[index % release.shard_size];
}

export async function getTagLabels() {
  return (await getTags())?.labels ?? [];
}

export async function getAbilities(index) {
  if (release.schema === 'mtgemb-release-v2') return [];
  const shard = shardOf(index);
  const rows = await cached(abilities, shard, async () => {
    const [cards, rows, tags] = await Promise.all([getCards(shard), get(`${release.release_id}/abilities/${pad(shard)}.json`), getTags()]);
    validateAbilityShard(cards, rows, release, shard, tags);
    return rows;
  });
  return rows[index % release.shard_size];
}

// v4 "line match" ranking: one ranked list per card, like neighbours
export async function getLines(index) {
  const shard = shardOf(index);
  const rows = await cached(lines, shard, async () => {
    const [cards, rows] = await Promise.all([getCards(shard), get(`${release.release_id}/lines/${pad(shard)}.json`)]);
    validateLineShard(cards, rows, release, shard);
    return rows;
  });
  return rows[index % release.shard_size];
}

// v6 candidate-side face matches are stored once per card, like line-match rows.
export async function getFaces(index) {
  if (!isV6(release)) return [];
  const shard = shardOf(index);
  const rows = await cached(faces, shard, async () => {
    const [cards, rows] = await Promise.all([getCards(shard), get(`${release.release_id}/faces/${pad(shard)}.json`)]);
    validateFaceShard(cards, rows, release, shard);
    return rows;
  });
  return rows[index % release.shard_size];
}

// Site data (map + pretrained neighbours) is optional: null when this release has none.

export async function loadSiteData(releaseId, fetchFn = fetcher) {
  try {
    const [info, map] = await Promise.all([get(`site-${releaseId}/map.json`, 'json', fetchFn), get(`site-${releaseId}/map.bin`, 'buffer', fetchFn)]);
    if (info.release_id !== releaseId) return null;
    site = { ...info, map };
    return site;
  } catch (reason) {
    if (reason.status === 404) return null;
    throw reason;
  }
}

export async function getPretrained(index) {
  if (!site) return null;
  const size = site.pretrained.shard_size;
  const rows = await cached(pretrained, Math.floor(index / size), () => get(`site-${site.release_id}/pretrained/${pad(Math.floor(index / size))}.json`));
  return rows[index % size];
}

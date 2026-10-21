// Release bundle and site-data loading. One promise cache per shard kind; failed loads are
// evicted so a retry can succeed. Shapes are checked with the bundle validators (SCHEMA.md).
import { validateAbilityShard, validateIds, validateRelease, validateReleasePointer, validateShard, validateTags } from './bundle.js';

let root = '';
let fetcher = (...args) => fetch(...args);
let release = null;
let ids = null;
let tagsReady = null;
const shards = new Map();
const abilities = new Map();
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
  release = await get(`${current.release_id}/release.json`);
  validateReleasePointer(current, release);
  validateRelease(release);
  const names = await get(`${release.release_id}/names.json`);
  validateRelease(release, names);
  // ids load after names so they do not share bandwidth with the search index
  const idsReady = get(`${release.release_id}/ids.json`).then((data) => { validateIds(release, data); ids = data.ids; return ids; });
  return { release, names, idsReady };
}

const getTags = () => (tagsReady ??= get(`${release.release_id}/tags.json`)
  .then((tags) => { validateTags(release, tags); return tags; })
  .catch((reason) => { tagsReady = null; throw reason; }));

export const shardOf = (index) => Math.floor(index / release.shard_size);

export function getShard(shard) {
  return cached(shards, shard, async () => {
    const [cards, neighbours, tags] = await Promise.all([get(`${release.release_id}/cards/${pad(shard)}.json`), get(`${release.release_id}/neighbours/${pad(shard)}.json`), getTags()]);
    validateShard(cards, neighbours, release, shard, ids, undefined, undefined, tags);
    return { cards, neighbours, tags: tags.labels };
  });
}

export async function getCard(index) {
  return (await getShard(shardOf(index))).cards[index % release.shard_size];
}

export async function getNeighbours(index) {
  return (await getShard(shardOf(index))).neighbours[index % release.shard_size];
}

export async function getTagLabels() {
  return (await getTags()).labels;
}

export async function getAbilities(index) {
  const shard = shardOf(index);
  const rows = await cached(abilities, shard, async () => {
    const [shardData, rows, tags] = await Promise.all([getShard(shard), get(`${release.release_id}/abilities/${pad(shard)}.json`), getTags()]);
    validateAbilityShard(shardData.cards, rows, release, shard, tags);
    return rows;
  });
  return rows[index % release.shard_size];
}

// Site data (map + pretrained neighbours) is optional: null when this release has none.
let site = null;

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

import { afterEach, describe, expect, it } from 'vitest';
import { getCard, getFaces, getTagLabels, loadRelease, loadSiteData } from './data.js';

const ok = (body) => ({ ok: true, status: 200, json: async () => body, arrayBuffer: async () => body });

describe('loadSiteData', () => {
  it('returns null on 404', async () => {
    const fetchFn = async () => ({ ok: false, status: 404 });
    expect(await loadSiteData('r1', fetchFn)).toBeNull();
  });

  it('returns buffer and regions', async () => {
    const buf = new ArrayBuffer(10);
    const fetchFn = async (url) => url.endsWith('map.bin') ? ok(buf) : ok({ release_id: 'r1', n_cards: 2, regions: [{ x: 1, y: 2, label: 'Burn', n: 5 }] });
    const site = await loadSiteData('r1', fetchFn);
    expect(site.map).toBe(buf);
    expect(site.regions[0].label).toBe('Burn');
  });

  it('rejects site data for another release', async () => {
    const fetchFn = async (url) => url.endsWith('map.bin') ? ok(new ArrayBuffer(10)) : ok({ release_id: 'other', regions: [] });
    expect(await loadSiteData('r1', fetchFn)).toBeNull();
  });
});

// a two-card v3 release held in memory; `prefix` names its cards so two releases differ
function v3Release(id, prefix) {
  const entry = { sha256: 'a'.repeat(64), bytes: 1 };
  const card = (i) => ({ id: `${id}-${i}`, name: `${prefix}${i}`, faces: [], color_identity: '', mv: 0, type_line: '', is_creature: false, has_text: true });
  return {
    'release.json': { schema: 'mtgemb-release-v3', release_id: id, n_cards: 2, shard_size: 256, k: 50, ability_k: 50, abilities_status: 'exploratory: not evaluated on the tag proxy',
      files: Object.fromEntries(['names.json', 'ids.json', 'cards/000.json', 'neighbours/000.json', 'abilities/000.json'].map((f) => [f, entry])) },
    'names.json': { names: [`${prefix}0`, `${prefix}1`], aliases: [] },
    'ids.json': { ids: [`${id}-0`, `${id}-1`] },
    'cards/000.json': [card(0), card(1)]
  };
}

function server(releases, current) {
  const fetchFn = async (url) => {
    const path = url.replace(/^\/data\//, '');
    if (path === 'current.json') return ok({ release_id: current.id });
    const [id, ...rest] = path.split('/');
    const body = releases[id]?.[rest.join('/')];
    return body === undefined ? { ok: false, status: 404 } : ok(body);
  };
  return fetchFn;
}

describe('release loading', () => {
  it('loads v3 cards without a tags file', async () => {
    const r = await loadRelease(server({ r3: v3Release('r3', 'A') }, { id: 'r3' }));
    await r.idsReady;
    expect((await getCard(1)).name).toBe('A1');
    expect(await getTagLabels()).toEqual([]);
  });

  it('drops cached shards when the deployed release changes', async () => {
    const releases = { ra: v3Release('ra', 'A'), rb: v3Release('rb', 'B') };
    const current = { id: 'ra' };
    const fetchFn = server(releases, current);
    await (await loadRelease(fetchFn)).idsReady;
    expect((await getCard(0)).name).toBe('A0');
    current.id = 'rb';
    await (await loadRelease(fetchFn)).idsReady;
    expect((await getCard(0)).name).toBe('B0');
  });

  it('loads and caches validated v6 face-match shards', async () => {
    const base = { id: 'r6-0', name: 'A', faces: [{ oracle_text: 'A' }], color_identity: '', mv: 1, type_line: '', is_creature: false, has_text: true, tags: [], colors: '', image_back: null };
    const entry = { sha256: 'a'.repeat(64), bytes: 1 };
    const files = ['names.json', 'ids.json', 'tags.json', 'cards/000.json', 'neighbours/000.json', 'abilities/000.json', 'lines/000.json', 'faces/000.json'];
    const rel = { 'release.json': { schema: 'mtgemb-release-v6', release_id: 'r6', n_cards: 2, shard_size: 256, k: 200, ability_k: 50, abilities_status: 'exploratory', line_k: 200, line_status: 'exploratory', face_k: 200, face_status: 'exploratory face matches', tags_status: 'proxy', files: Object.fromEntries(files.map((f) => [f, entry])) },
      'names.json': { names: ['A', 'B'], aliases: [] }, 'ids.json': { ids: ['r6-0', 'r6-1'] }, 'tags.json': { labels: [] }, 'cards/000.json': [base, { ...base, id: 'r6-1', has_text: false }], 'faces/000.json': [[[1, 0.8]], []] };
    let faceRequests = 0;
    const route = server({ r6: rel }, { id: 'r6' });
    const fetchFn = async (url) => { if (url.endsWith('/faces/000.json')) faceRequests += 1; return route(url); };
    await (await loadRelease(fetchFn)).idsReady;
    expect(await getFaces(0)).toEqual([[1, 0.8]]);
    expect(await getFaces(1)).toEqual([]);
    expect(await getFaces(0)).toEqual([[1, 0.8]]);
    expect(faceRequests).toBe(1);
  });
});

// the inline script that prune-build.mjs adds to the explorer page (prefetch.js) starts these requests before the app boots
describe('early start-up requests', () => {
  afterEach(() => { delete globalThis.__atlasPrefetch; });

  it('uses the responses the page prefetched instead of fetching again', async () => {
    const rel = v3Release('r3', 'A');
    globalThis.__atlasPrefetch = { 'current.json': Promise.resolve(ok({ release_id: 'r3' })), 'r3/release.json': Promise.resolve(ok(rel['release.json'])), 'r3/names.json': Promise.resolve(ok(rel['names.json'])) };
    const seen = [];
    const route = server({ r3: rel }, { id: 'r3' });
    const r = await loadRelease(async (url) => { seen.push(url); return route(url); });
    await r.idsReady;
    expect(r.names.names).toEqual(['A0', 'A1']);
    expect(seen).toEqual(['/data/r3/ids.json']);
  });

  it('fetches normally when a prefetched request failed', async () => {
    const failed = Promise.reject(new Error('offline'));
    failed.catch(() => {});
    globalThis.__atlasPrefetch = { 'current.json': failed };
    const r = await loadRelease(server({ r3: v3Release('r3', 'A') }, { id: 'r3' }));
    expect(r.release.release_id).toBe('r3');
  });

  it('fetches normally when a prefetched response is an HTTP error', async () => {
    const rel = v3Release('r3', 'A');
    for (const path of ['current.json', 'r3/release.json', 'r3/names.json']) {
      globalThis.__atlasPrefetch = { 'current.json': Promise.resolve(ok({ release_id: 'r3' })), 'r3/release.json': Promise.resolve(ok(rel['release.json'])), 'r3/names.json': Promise.resolve(ok(rel['names.json'])) };
      globalThis.__atlasPrefetch[path] = Promise.resolve({ ok: false, status: path === 'r3/names.json' ? 500 : 404 });
      const seen = [];
      const route = server({ r3: rel }, { id: 'r3' });
      const r = await loadRelease(async (url) => { seen.push(url); return route(url); });
      expect(r.names.names).toEqual(['A0', 'A1']);
      expect(seen).toContain(`/data/${path}`);
    }
  });

  it('drops prefetched responses for a release current.json did not name', async () => {
    const a = v3Release('ra', 'A');
    const b = v3Release('rb', 'B');
    // the page was built for ra, but current.json now names rb; ra's prefetched files are broken
    globalThis.__atlasPrefetch = { 'current.json': Promise.resolve(ok({ release_id: 'rb' })), 'ra/release.json': Promise.resolve(ok({ broken: true })), 'ra/names.json': Promise.resolve(ok({ broken: true })) };
    expect((await loadRelease(server({ ra: a, rb: b }, { id: 'rb' }))).release.release_id).toBe('rb');
    // a later load that goes back to ra must fetch it again
    expect((await loadRelease(server({ ra: a, rb: b }, { id: 'ra' }))).names.names).toEqual(['A0', 'A1']);
  });

  it('requests names.json without waiting for release.json', async () => {
    const log = [];
    const route = server({ r3: v3Release('r3', 'A') }, { id: 'r3' });
    const fetchFn = async (url) => {
      if (url.endsWith('names.json')) log.push('names requested');
      if (url.endsWith('release.json')) await new Promise((done) => setTimeout(done, 20)).then(() => log.push('release arrived'));
      return route(url);
    };
    await loadRelease(fetchFn);
    expect(log).toEqual(['names requested', 'release arrived']);
  });
});

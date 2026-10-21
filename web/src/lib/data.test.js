import { describe, expect, it } from 'vitest';
import { loadSiteData } from './data.js';

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

import { describe, expect, it } from 'vitest';
import { loadFonts } from './load-fonts.js';

const root = () => {
  const classes = new Set();
  return { classes, classList: { add: (c) => classes.add(c), remove: (c) => classes.delete(c) } };
};

describe('loadFonts', () => {
  it('leaves the page alone when the fonts load', async () => {
    const r = root();
    await loadFonts(async () => ({}), r);
    expect([...r.classes]).toEqual([]);
  });

  it('marks the page so mana symbols show as text when the fonts fail, without rejecting', async () => {
    const r = root();
    await expect(loadFonts(() => Promise.reject(new Error('offline')), r)).resolves.toBeUndefined();
    expect([...r.classes]).toEqual(['no-mana']);
  });

  it('gives up waiting on a stalled import, then shows the icons if it arrives late', async () => {
    const r = root();
    let arrive;
    await loadFonts(() => new Promise((done) => { arrive = done; }), r, 10);
    expect([...r.classes]).toEqual(['no-mana']);
    arrive({});
    await new Promise((done) => setTimeout(done, 0));
    expect([...r.classes]).toEqual([]);
  });
});

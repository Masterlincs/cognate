import { describe, expect, it } from 'vitest';
import { loadFonts } from './load-fonts.js';

const root = () => { const classes = []; return { classes, classList: { add: (c) => classes.push(c) } }; };

describe('loadFonts', () => {
  it('leaves the page alone when the fonts load', async () => {
    const r = root();
    await loadFonts(async () => ({}), r);
    expect(r.classes).toEqual([]);
  });

  it('marks the page so mana symbols show as text when the fonts fail, without rejecting', async () => {
    const r = root();
    await expect(loadFonts(() => Promise.reject(new Error('offline')), r)).resolves.toBeUndefined();
    expect(r.classes).toEqual(['no-mana']);
  });
});

import { describe, expect, it } from 'vitest';
import { prefetch, withPrefetch } from './prefetch.js';

describe('start-up prefetch', () => {
  it('requests current.json, release.json and names.json at once and keeps them by path', () => {
    const urls = [];
    const win = {};
    prefetch(win, (url) => { urls.push(url); return Promise.resolve(url); }, 'r6');
    expect(urls).toEqual(['data/current.json', 'data/r6/release.json', 'data/r6/names.json']);
    expect(Object.keys(win.__atlasPrefetch)).toEqual(['current.json', 'r6/release.json', 'r6/names.json']);
  });

  it('is inserted first in the head with the release id', () => {
    const html = '<html><head><link rel="stylesheet" href="a.css"></head></html>';
    const out = withPrefetch(html, '2026-10-16-e5-base-v2-3984a6b7');
    expect(out).toMatch(/^<html><head><script>\(.*__atlasPrefetch.*\)\(window, fetch, "2026-10-16-e5-base-v2-3984a6b7"\)<\/script><link/s);
  });

  it('refuses an unsafe release id or a page without a head', () => {
    for (const id of ['../x', 'a/b', 'a"</script>', '']) expect(() => withPrefetch('<head></head>', id)).toThrow();
    expect(() => withPrefetch('<html></html>', 'r6')).toThrow();
  });

  it('runs as the inlined script text', () => {
    const urls = [];
    const win = {};
    const script = withPrefetch('<head>', 'r6').match(/<script>(.*)<\/script>/s)[1];
    new Function('window', 'fetch', script)(win, (url) => { urls.push(url); return Promise.resolve(); });
    expect(urls).toEqual(['data/current.json', 'data/r6/release.json', 'data/r6/names.json']);
  });
});

// Keep only the deployed release (current.json) and its site data in build/data, and start that
// release's search index download from the explorer page itself.
// Old releases stay in public/data for rollback; they are just not shipped.
import { access, readdir, readFile, rm, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { withPrefetch } from '../src/lib/prefetch.js';

const build = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../build');
const data = path.join(build, 'data');
const { release_id: id } = JSON.parse(await readFile(path.join(data, 'current.json'), 'utf8'));
if (!id || /[/\\]|\.\./.test(id)) throw new Error('Invalid release pointer in current.json.');
// a build without its release data would deploy an explorer that cannot load anything
await access(path.join(data, id, 'release.json')).catch(() => { throw new Error(`current.json names ${id}, but public/data/${id} is missing.`); });
const keep = new Set(['current.json', id, `site-${id}`]);
for (const entry of await readdir(data)) {
  if (!keep.has(entry)) await rm(path.join(data, entry), { recursive: true, force: true });
}
console.log(`build/data: kept ${[...keep].join(', ')}`);
const index = path.join(build, 'index.html');
await writeFile(index, withPrefetch(await readFile(index, 'utf8'), id));
console.log(`build/index.html: prefetches ${id}`);

// Cloudflare Pages / Netlify-style `_headers`: a CSP that allows only this build's inline scripts
// (by hash), plus cache rules. Hashes are computed after the prefetch script is injected.
// browsers hash script text after the HTML parser turns CRLF into LF
const sha = (s) => `'sha256-${createHash('sha256').update(s.replace(/\r\n?/g, '\n')).digest('base64')}'`;
const hashes = new Set();
for (const file of ['index.html', 'how-it-works.html']) {
  const html = await readFile(path.join(build, file), 'utf8');
  for (const m of html.matchAll(/<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/g)) hashes.add(sha(m[1]));
}
const csp = [
  "default-src 'none'",
  `script-src 'self' ${[...hashes].join(' ')}`,
  "style-src 'self' 'unsafe-inline'",
  "img-src 'self' https://cards.scryfall.io data:",
  "font-src 'self'",
  "connect-src 'self'",
  "base-uri 'none'",
  "form-action 'none'",
  "frame-ancestors 'none'"
].join('; ');
const security = [
  `  Content-Security-Policy: ${csp}`,
  '  X-Content-Type-Options: nosniff',
  '  Referrer-Policy: no-referrer',
  '  Permissions-Policy: camera=(), microphone=(), geolocation=()',
  '  Strict-Transport-Security: max-age=31536000'
];
const headers = [
  '/*', ...security,
  '/_app/immutable/*', '  Cache-Control: public, max-age=31536000, immutable',
  '/data/current.json', '  Cache-Control: no-cache',
  `/data/site-${id}/*`, '  Cache-Control: public, max-age=3600',
  `/data/${id}/*`, '  Cache-Control: public, max-age=3600'
];
await writeFile(path.join(build, '_headers'), headers.join('\n') + '\n');
console.log(`build/_headers: ${hashes.size} inline script hashes`);

// A real 404 page, so unknown paths do not fall back to index.html with status 200.
await writeFile(path.join(build, '404.html'), `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><meta name="color-scheme" content="light dark"><title>Not found | Cognate</title></head>
<body style="font-family: system-ui, sans-serif; max-width: 32rem; margin: 4rem auto; padding: 0 1rem"><h1>Page not found</h1><p><a href="/">Go to Cognate</a></p></body></html>
`);
console.log('build/404.html written');

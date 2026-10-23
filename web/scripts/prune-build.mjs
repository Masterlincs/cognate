// Keep only the deployed release (current.json) and its site data in build/data, and start that
// release's search index download from the explorer page itself.
// Old releases stay in public/data for rollback; they are just not shipped.
import { access, readdir, readFile, rm, writeFile } from 'node:fs/promises';
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

// Keep only the deployed release (current.json) and its site data in build/data.
// Old releases stay in public/data for rollback; they are just not shipped.
import { readdir, readFile, rm } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const data = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../build/data');
const { release_id: id } = JSON.parse(await readFile(path.join(data, 'current.json'), 'utf8'));
if (!id || /[/\\]|\.\./.test(id)) throw new Error('Invalid release pointer in current.json.');
const keep = new Set(['current.json', id, `site-${id}`]);
for (const entry of await readdir(data)) {
  if (!keep.has(entry)) await rm(path.join(data, entry), { recursive: true, force: true });
}
console.log(`build/data: kept ${[...keep].join(', ')}`);

import { createHash } from 'node:crypto';
import { access, readFile } from 'node:fs/promises';
import path from 'node:path';
import { validateIds, validateRelease, validateReleasePointer, validateShard } from '../src/lib.js';

const dir = path.resolve(process.argv[2] ?? '');
if (!process.argv[2]) {
  console.error('Usage: node scripts/check-bundle.mjs <bundle-dir>');
  process.exit(2);
}
const readJson = async (file) => JSON.parse(await readFile(path.join(dir, file), 'utf8'));
const digest = (bytes) => createHash('sha256').update(bytes).digest('hex');
try {
  const release = await readJson('release.json');
  const names = await readJson('names.json');
  const ids = await readJson('ids.json');
  const pointerPath = path.join(dir, '..', 'current.json');
  try {
    await access(pointerPath);
    validateReleasePointer(JSON.parse(await readFile(pointerPath, 'utf8')), release);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  validateRelease(release, names);
  validateIds(release, ids);
  const expectedFiles = ['names.json', 'ids.json', ...Array.from({ length: Math.ceil(release.n_cards / release.shard_size) }, (_, i) => [`cards/${String(i).padStart(3, '0')}.json`, `neighbours/${String(i).padStart(3, '0')}.json`]).flat()];
  for (const file of expectedFiles) {
    const bytes = await readFile(path.join(dir, file));
    const entry = release.files?.[file];
    if (!entry || entry.bytes !== bytes.byteLength || entry.sha256 !== digest(bytes)) throw new Error(`Hash or byte count mismatch: ${file}`);
  }
  for (let shard = 0; shard < Math.ceil(release.n_cards / release.shard_size); shard += 1) {
    const number = String(shard).padStart(3, '0');
    validateShard(await readJson(`cards/${number}.json`), await readJson(`neighbours/${number}.json`), release, shard, ids.ids);
  }
  console.log(`Bundle valid: ${release.release_id} (${release.n_cards} cards)`);
} catch (error) {
  console.error(`Invalid bundle: ${error.message}`);
  process.exitCode = 1;
}

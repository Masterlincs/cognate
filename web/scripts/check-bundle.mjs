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
    const pointer = JSON.parse(await readFile(pointerPath, 'utf8'));
    if (pointer.release_id === release.release_id) validateReleasePointer(pointer, release);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
  }
  validateRelease(release, names);
  validateIds(release, ids);
  const v3plus = release.schema !== 'mtgemb-release-v2';
  const v4 = release.schema === 'mtgemb-release-v4';
  const expectedFiles = ['names.json', 'ids.json', ...(v4 ? ['tags.json'] : []), ...Array.from({ length: Math.ceil(release.n_cards / release.shard_size) }, (_, i) => [`cards/${String(i).padStart(3, '0')}.json`, `neighbours/${String(i).padStart(3, '0')}.json`, ...(v3plus ? [`abilities/${String(i).padStart(3, '0')}.json`] : []), ...(v4 ? [`lines/${String(i).padStart(3, '0')}.json`] : [])]).flat()];
  for (const file of expectedFiles) {
    const bytes = await readFile(path.join(dir, file));
    const entry = release.files?.[file];
    if (!entry || entry.bytes !== bytes.byteLength || entry.sha256 !== digest(bytes)) throw new Error(`Hash or byte count mismatch: ${file}`);
  }
  const tags = v4 ? await readJson('tags.json') : undefined;
  for (let shard = 0; shard < Math.ceil(release.n_cards / release.shard_size); shard += 1) {
    const number = String(shard).padStart(3, '0');
    validateShard(await readJson(`cards/${number}.json`), await readJson(`neighbours/${number}.json`), release, shard, ids.ids, v3plus ? await readJson(`abilities/${number}.json`) : null, v4 ? await readJson(`lines/${number}.json`) : undefined, tags);
  }
  console.log(`Bundle valid: ${release.release_id} (${release.n_cards} cards)`);
} catch (error) {
  console.error(`Invalid bundle: ${error.message}`);
  process.exitCode = 1;
}

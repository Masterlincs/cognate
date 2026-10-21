import { createHash } from 'node:crypto';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../public/data');
const releaseId = 'fixture-v3';
const dir = path.join(root, releaseId);
const count = 600;
const shardSize = 256;
const k = 50;
const abilityK = 50;
const colors = ['', 'W', 'U', 'B', 'R', 'G', 'WU', 'BR', 'GUR', 'WUBRG'];
const inventedNouns = ['Aster', 'Brindle', 'Cinder', 'Dapple', 'Ember', 'Fable', 'Glimmer', 'Harbor', 'Ivory', 'Juniper', 'Kettle', 'Lumen', 'Morrow', 'Nettle', 'Opal', 'Pebble', 'Quarry', 'Ripple', 'Saffron', 'Thistle'];
const typeKinds = ['Creature — Clockwork', 'Instant', 'Artifact', 'Enchantment — Observatory', 'Sorcery', 'Creature — Mosskin'];
const hash = (bytes) => createHash('sha256').update(bytes).digest('hex');
const writeJson = async (relative, value, files = null) => {
  const bytes = `${JSON.stringify(value)}\n`;
  await mkdir(path.dirname(path.join(dir, relative)), { recursive: true });
  await writeFile(path.join(dir, relative), bytes, 'utf8');
  if (files) files[relative] = { sha256: hash(bytes), bytes: Buffer.byteLength(bytes) };
};

const ids = Array.from({ length: count }, (_, i) => `fixture-${String(i).padStart(4, '0')}`);
const names = ids.map((_, i) => i % 31 === 0 ? `${inventedNouns[i % 20]} Fold // ${inventedNouns[(i + 7) % 20]} Unfold` : `${inventedNouns[i % 20]} ${typeKinds[i % typeKinds.length].split(' ')[0]} ${String(i).padStart(3, '0')}`);
const aliases = names.flatMap((name, i) => name.includes(' // ') ? name.split(' // ').map((face) => [face, i]) : []);
const nameIndex = { names, aliases };
const files = {};
await mkdir(path.join(dir, 'cards'), { recursive: true });
await mkdir(path.join(dir, 'neighbours'), { recursive: true });
await mkdir(path.join(dir, 'abilities'), { recursive: true });
await writeJson('names.json', nameIndex, files);
await writeJson('ids.json', { ids }, files);
for (let shard = 0; shard < Math.ceil(count / shardSize); shard += 1) {
  const start = shard * shardSize;
  const length = Math.min(shardSize, count - start);
  const cards = Array.from({ length }, (_, offset) => {
    const i = start + offset;
    const multi = i % 31 === 0;
    const textless = i === 13;
    const singleAbility = i === 14;
    const faceNames = multi ? names[i].split(' // ') : [names[i]];
    const faces = faceNames.map((name, face) => ({ name, mana_cost: `{${(i + face) % 6 + 1}}${['{W}', '{U}', '{B}', '{R}', '{G}'][i % 5]}`, type_line: typeKinds[(i + face) % typeKinds.length], oracle_text: textless ? '' : singleAbility ? `When ${name} enters the workshop, place a copper marker on a nearby invention.` : `When ${name} enters the workshop, place a ${['copper', 'glass', 'woven', 'quiet'][i % 4]} marker on a nearby invention.\nAt the next dawn, you may move that marker.` }));
    return { id: ids[i], name: names[i], layout: multi ? 'transform' : 'normal', faces, color_identity: colors[i % colors.length], mv: (i * 7 % 120) / 10, type_line: faces.map((face) => face.type_line).join(' // '), is_creature: faces.some((face) => face.type_line.startsWith('Creature')), image: i % 5 === 0 ? 'https://example.invalid/fixture-image.webp' : null, scryfall_uri: `https://scryfall.com/search?q=${encodeURIComponent(names[i])}`, has_text: !textless };
  });
  const neighbours = cards.map((card, offset) => {
    const source = start + offset;
    if (!card.has_text) return [];
    const results = [];
    for (let rank = 0; results.length < k; rank += 1) {
      const target = (source + rank + 1) % count;
      if (target !== source && target !== 13) results.push([target, Number((0.97 - results.length * 0.0091).toFixed(4))]);
    }
    return results;
  });
  const abilities = cards.map((card, offset) => {
    if (!card.has_text) return [];
    const lines = card.faces.flatMap((face) => face.oracle_text.split('\n').filter((line) => line.trim()));
    if (lines.length < 2) return [];
    return lines.map((text) => {
      const source = start + offset;
      const results = [];
      for (let rank = 0; results.length < abilityK; rank += 1) {
        const target = (source + rank + 1) % count;
        if (target !== source && target !== 13) results.push([target, Number((0.96 - results.length * 0.0087).toFixed(4))]);
      }
      return { text, neighbours: results };
    });
  });
  const shardName = String(shard).padStart(3, '0');
  await writeJson(`cards/${shardName}.json`, cards, files);
  await writeJson(`neighbours/${shardName}.json`, neighbours, files);
  await writeJson(`abilities/${shardName}.json`, abilities, files);
}
await writeJson('release.json', {
  schema: 'mtgemb-release-v3', release_id: releaseId, fixture: true, snapshot_date: '2026-10-17', created_utc: '2026-10-17T00:00:00Z',
  model: { label: 'Synthetic fixture ranking', status: 'experimental', backbone: 'synthetic', run_id: 'fixture-v3', decision: 'Fixture data only; no model claim.' },
  n_cards: count, k, ability_k: abilityK, abilities_status: 'exploratory: not evaluated on the tag proxy', shard_size: shardSize, files,
  attribution: 'Invented synthetic names and rules text for software testing only; no affiliation with Wizards of the Coast or Scryfall.'
}, null);
await mkdir(root, { recursive: true });
await writeFile(path.join(root, 'current.json'), `${JSON.stringify({ release_id: releaseId })}\n`, 'utf8');
console.log(`Wrote ${count} deterministic fixture cards to public/data/${releaseId} and pointed current.json at it`);
console.log('To go back to the real release: git checkout public/data/current.json');

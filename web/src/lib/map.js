// Card map: 5-byte records (int16 x, int16 y, uint8 WUBRG mask, little-endian) in card order.
export const TEXTLESS = -32768;
const W = 1, U = 2, B = 4, R = 8, G = 16;
export const SINGLE = { [W]: 'w', [U]: 'u', [B]: 'b', [R]: 'r', [G]: 'g' };

export function decodeMap(buf) {
  if (buf.byteLength % 5) throw new Error('Map data is truncated.');
  const n = buf.byteLength / 5;
  const view = new DataView(buf);
  const x = new Int16Array(n), y = new Int16Array(n), color = new Uint8Array(n);
  for (let i = 0; i < n; i += 1) {
    x[i] = view.getInt16(i * 5, true);
    y[i] = view.getInt16(i * 5 + 2, true);
    color[i] = view.getUint8(i * 5 + 4);
  }
  return { x, y, color, n };
}

// colour key for a WUBRG mask: one colour, 'm' for multicolour, 'c' for colourless
export const colorKey = (mask) => mask === 0 ? 'c' : SINGLE[mask] ?? 'm';

export const toScreen = (v, x, y) => [v.tx + x * v.scale, v.ty + y * v.scale];
export const toWorld = (v, sx, sy) => [(sx - v.tx) / v.scale, (sy - v.ty) / v.scale];

const key = (cx, cy) => `${cx},${cy}`;

export function buildGrid(x, y, cell = 512) {
  const grid = new Map();
  grid.cell = cell;
  for (let i = 0; i < x.length; i += 1) {
    if (x[i] === TEXTLESS) continue;
    const k = key(Math.floor(x[i] / cell), Math.floor(y[i] / cell));
    if (!grid.has(k)) grid.set(k, []);
    grid.get(k).push(i);
  }
  return grid;
}

export function hitTest(grid, x, y, wx, wy, radius) {
  const cell = grid.cell;
  let best = -1, bestD = radius * radius;
  for (let cx = Math.floor((wx - radius) / cell); cx <= Math.floor((wx + radius) / cell); cx += 1) {
    for (let cy = Math.floor((wy - radius) / cell); cy <= Math.floor((wy + radius) / cell); cy += 1) {
      for (const i of grid.get(key(cx, cy)) ?? []) {
        const d = (x[i] - wx) ** 2 + (y[i] - wy) ** 2;
        if (d < bestD || (d === bestD && (best < 0 || i < best))) { best = i; bestD = d; }
      }
    }
  }
  return best;
}

// Region labels are Scryfall Tagger tag names; a few are player jargon, renamed for display.
const READABLE = {
  anthem: 'anthems (team pump)',
  cantrip: 'cantrips (draw a card)',
  'copy-creature': 'copy a creature',
  creaturefall: 'creature-enters triggers',
  'death trigger-self': 'dies triggers (self)',
  firebreathing: 'pump for mana (+1/+0)',
  'force draw': 'make players draw',
  'free-cast-another': 'cast spells for free',
  jump: 'grant flying',
  'pp counters matter': '+1/+1 counters matter',
  'protects-self': 'self-protection',
  'reanimate-creature': 'reanimation',
  'removal-bounce': 'removal: bounce',
  'removal-exile': 'removal: exile',
  'removal-toughness': 'removal: −X/−X',
  'sacrifice outlet-artifact': 'sacrifice outlets (artifacts)',
  'shade pump': 'pump for mana (+1/+1)',
  'single target instant/sorcery': 'single-target spells',
  'synergy-sorcery': 'sorcery synergy',
  'tapper-creature': 'tapper creatures',
  thoughtseize: 'hand disruption',
  'tutor-to-hand': 'tutors (search for a card)',
  egg: 'sacrifice-for-value artifacts',
  saboteur: 'combat-damage triggers',
  'cda-power': 'power set by its own ability'
};
export const regionLabel = (label) => READABLE[label] ?? label;

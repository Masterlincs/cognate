// ids are strictly ascending (validateIds), so a binary search finds the card index
export function resolveSelection(ids, state) {
  if (!state.card) return { index: -1, unknown: false };
  let lo = 0, hi = ids.length - 1;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (ids[mid] === state.card) return { index: mid, unknown: false };
    if (ids[mid] < state.card) lo = mid + 1; else hi = mid - 1;
  }
  return { index: -1, unknown: true };
}

export const activeFilters = (f) => Boolean(f.color || f.creature || f.type || Number.isFinite(f.minMv) || Number.isFinite(f.maxMv));

// the list being shown: one ability's matches, the line-match ranking (v4), or the whole card
export const savedList = ({ ability, rank, whole, lines, faces, abilities }) =>
  ability !== null ? abilities[ability]?.neighbours ?? [] : rank === 'line' ? lines ?? [] : rank === 'face' ? faces ?? [] : whole;

// unfiltered, every saved neighbour can be shown (its card loads on demand); filtered, only matches
export const hasMore = ({ saved, visible, shown, filtering }) => (filtering ? visible.length : saved.length) > shown;

// the active filters as removable chips, [{ key, label }]; clearFilter(filters, key) removes one
const COLOUR_NAMES = { W: 'White', U: 'Blue', B: 'Black', R: 'Red', G: 'Green' };
const COLOUR_MODES = { include: 'including', exact: 'exactly', within: 'at most' };
export function filterChips(f) {
  const out = [];
  if (f.color) {
    const what = f.color === 'C' ? 'colourless' : `${COLOUR_MODES[f.colorMode] ?? 'including'} ${[...f.color].map((c) => COLOUR_NAMES[c]).join(', ')}`;
    out.push({ key: 'color', label: `${f.colorBy === 'color' ? 'Colour' : 'Identity'}: ${what}` });
  }
  if (f.creature) out.push({ key: 'creature', label: f.creature === 'creature' ? 'Creatures' : 'Non-creatures' });
  if (f.type) out.push({ key: 'type', label: `Type: ${f.type}` });
  const lo = Number.isFinite(f.minMv), hi = Number.isFinite(f.maxMv);
  if (lo || hi) out.push({ key: 'mv', label: lo && hi ? (f.minMv === f.maxMv ? `MV ${f.minMv}` : `MV ${f.minMv}–${f.maxMv}`) : lo ? `MV ≥ ${f.minMv}` : `MV ≤ ${f.maxMv}` });
  return out;
}

export function clearFilter(f, key) {
  if (key === 'color') return { ...f, color: '', colorMode: 'include', colorBy: 'identity' };
  if (key === 'mv') return { ...f, minMv: null, maxMv: null };
  return { ...f, [key]: '' };
}

// a random index in [0, n) that passes ok: a few draws, then a scan from a random start; -1 if none passes
export function randomIndex(n, ok, rand = Math.random) {
  for (let t = 0; t < 64 && n > 0; t += 1) {
    const i = Math.floor(rand() * n);
    if (ok(i)) return i;
  }
  const start = Math.floor(rand() * n);
  for (let k = 0; k < n; k += 1) if (ok((start + k) % n)) return (start + k) % n;
  return -1;
}

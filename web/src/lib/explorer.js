import { filterNeighbours } from './bundle.js';

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

export const visibleNeighbours = (saved, cardByIndex, filters) => filterNeighbours(saved, cardByIndex, filters);

export const activeFilters = (f) => Boolean(f.color || f.creature || f.type || Number.isFinite(f.minMv) || Number.isFinite(f.maxMv));

// the list being shown: one ability's matches, the line-match ranking (v4), or the whole card
export const savedList = ({ ability, rank, whole, lines, abilities }) =>
  ability !== null ? abilities[ability]?.neighbours ?? [] : rank === 'line' ? lines ?? [] : whole;

// unfiltered, every saved neighbour can be shown (its card loads on demand); filtered, only matches
export const hasMore = ({ saved, visible, shown, filtering }) => (filtering ? visible.length : saved.length) > shown;

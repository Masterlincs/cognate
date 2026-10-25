import { normalize, searchKeys } from './bundle.js';

const isWordChar = (ch) => /[a-z0-9]/.test(ch);

// 0 exact, 1 prefix, 2 word start, 3 substring, 4 every word somewhere; -1 no match
function tier(key, q, words) {
  if (key === q) return 0;
  let at = key.indexOf(q);
  if (at === 0) return 1;
  if (at > 0) {
    for (; at > 0; at = key.indexOf(q, at + 1)) if (!isWordChar(key[at - 1])) return 2;
    return 3;
  }
  return words && words.every((w) => key.includes(w)) ? 4 : -1;
}

// the search box's matches: best tier first, shorter (usually the plainer) names first within a tier
export function findCards(query, names, limit = 8) {
  const q = normalize(query).replace(/\s+/g, ' ');
  if (!q) return [];
  const words = q.includes(' ') ? q.split(' ') : null;
  const tiers = [[], [], [], [], []];
  const keys = searchKeys(names);
  for (let i = 0; i < keys.length; i += 1) {
    let best = -1;
    for (const key of keys[i]) {
      const t = tier(key, q, words);
      if (t >= 0 && (best < 0 || t < best)) best = t;
    }
    if (best >= 0) tiers[best].push(i);
  }
  const out = [];
  for (const list of tiers) {
    if (out.length >= limit) break;
    list.sort((a, b) => names.names[a].length - names.names[b].length || a - b);
    out.push(...list.slice(0, limit - out.length));
  }
  return out.map((index) => ({ index, name: names.names[index] }));
}

// recently opened cards, newest first
export const remember = (list, entry, max = 6) => [entry, ...list.filter((e) => e.id !== entry.id)].slice(0, max);

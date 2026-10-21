// Card text helpers. "Shared lines" are exact matches after masking numbers (the exam rule,
// mtgemb.exam.mask_numbers) and the card's own names; they show overlap, not model evidence.
const NUM = /\b(\d+|X|one|two|three|four|five|six|seven|eight|nine|ten)\b/g;

export const maskNumbers = (s) => s.replace(NUM, '#');

export function cardLines(card) {
  return card.faces.flatMap((f, face) => (f.oracle_text ?? '').split('\n').map((t) => t.trim()).filter((t) => t && t !== '//').map((text) => ({ face, text })));
}

const escape = (s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');

export function lineKey(card, text) {
  let s = text;
  for (const n of new Set([card.name, ...card.faces.map((f) => f.name)].filter(Boolean))) s = s.replace(new RegExp(escape(n), 'g'), '~');
  return maskNumbers(s).toLowerCase().trim();
}

export function sharedLines(a, b) {
  const keys = new Set(cardLines(a).map((l) => lineKey(a, l.text)));
  return new Set(cardLines(b).map((l) => lineKey(b, l.text)).filter((k) => keys.has(k)));
}

export const manaTokens = (cost) => [...(cost ?? '').matchAll(/\{([^}]+)\}/g)].map((m) => m[1]);

// One {X} symbol: its colours (0 = neutral pill, 2 = hybrid split) and the text drawn on it.
export function manaSymbol(sym) {
  if (sym === 'T') return { colours: [], text: '↷' };
  const parts = sym.split('/');
  const colours = parts.filter((p) => /^[WUBRGC]$/.test(p)).map((p) => p.toLowerCase());
  if (parts.length === 1) return { colours, text: sym };
  return { colours, text: parts.includes('P') ? 'φ' : parts.find((p) => /^\d+$/.test(p)) ?? '' };
}

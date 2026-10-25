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

// The Mana font class for one {X} symbol, or null when the font has no glyph for it.
const MANA_SPECIAL = { T: 'tap', Q: 'untap', '1/2': '1-2', '½': '1-2' };
export function manaClass(sym) {
  if (MANA_SPECIAL[sym]) return MANA_SPECIAL[sym];
  const s = sym.toLowerCase();
  if (/^(\d{1,2}|100|[wubrgcxyzpse]|tk|chaos)$/.test(s)) return s;
  if (/^([wubrgc2]\/[wubrg](\/p)?|[wubrg]\/p)$/.test(s)) return s.replaceAll('/', '');
  return null;
}

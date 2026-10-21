import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { cardLines, manaClass, manaTokens, maskNumbers, sharedLines } from './text.js';

const card = (name, ...texts) => ({ name, faces: texts.map((t, i) => ({ name: i ? `${name} back` : name, oracle_text: t, mana_cost: '', type_line: '' })) });

describe('text helpers', () => {
  it('maskNumbers follows the exam rule', () => {
    expect(maskNumbers('Draw two cards. X is 3; Xylo')).toBe('Draw # cards. # is #; Xylo');
  });

  it('sharedLines matches after number masking', () => {
    const shared = sharedLines(card('A', 'Flying\nDraw 2 cards.'), card('B', 'Draw 3 cards.\nTrample'));
    expect([...shared]).toEqual(['draw # cards.']);
  });

  it('sharedLines masks each card\'s own name', () => {
    const shared = sharedLines(card('Shock', 'Shock deals 2 damage to any target.'), card('Bolt', 'Bolt deals 3 damage to any target.'));
    expect(shared.has('~ deals # damage to any target.')).toBe(true);
  });

  it('sharedLines ignores face separators', () => {
    const a = card('Front', 'Haste', 'Draw a card.');
    const b = { name: 'Other', faces: [{ name: 'Other', oracle_text: 'Draw a card.\n//\nHaste' }] };
    expect([...sharedLines(a, b)].sort()).toEqual(['draw a card.', 'haste']);
  });

  it('cardLines keeps face index', () => {
    expect(cardLines(card('A', 'One\n\nTwo', 'Three'))).toEqual([{ face: 0, text: 'One' }, { face: 0, text: 'Two' }, { face: 1, text: 'Three' }]);
  });

  it('manaTokens splits hybrid and generic', () => {
    expect(manaTokens('{2}{R}{W/U}')).toEqual(['2', 'R', 'W/U']);
    expect(manaTokens('')).toEqual([]);
  });
});

describe('manaClass', () => {
  it('maps symbols to Mana font classes', () => {
    expect(['R', '3', '12', 'X', 'T', 'Q', 'TK', 'E', 'S', 'P', 'C'].map(manaClass)).toEqual(['r', '3', '12', 'x', 'tap', 'untap', 'tk', 'e', 's', 'p', 'c']);
    expect(['W/U', '2/R', 'B/P', 'G/U/P', 'C/W'].map(manaClass)).toEqual(['wu', '2r', 'bp', 'gup', 'cw']);
  });
  it('returns null for symbols the font does not draw', () => {
    expect(manaClass('HW')).toBeNull();
    expect(manaClass('<script>')).toBeNull();
  });
  it('every class it returns exists in the vendored stylesheet', () => {
    const css = readFileSync(new URL('./vendor/mana.css', import.meta.url), 'utf8');
    const seen = ['W', 'U', 'B', 'R', 'G', 'C', 'X', 'T', 'Q', 'TK', 'E', 'S', 'P', '0', '1', '9', '10', '16', '20', '1/2', 'W/B', 'U/R', 'R/W', 'G/U', 'G/W', 'B/R', 'U/B', 'W/U', 'B/G', 'R/G', '2/W', '2/U', '2/B', '2/R', '2/G', 'W/P', 'U/P', 'B/P', 'R/P', 'G/P', 'G/U/P', 'G/W/P', 'R/G/P', 'R/W/P', 'C/W', 'C/U', 'C/B', 'C/R', 'C/G'];
    for (const s of seen) expect(css.includes(`.ms-${manaClass(s)}::before`) || css.includes(`.ms-${manaClass(s)}:before`), s).toBe(true);
  });
});

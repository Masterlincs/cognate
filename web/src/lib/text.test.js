import { describe, expect, it } from 'vitest';
import { cardLines, manaTokens, maskNumbers, sharedLines, manaSymbol } from './text.js';

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

describe('manaSymbol', () => {
  it('colours a single mana symbol', () => {
    expect(manaSymbol('R')).toEqual({ colours: ['r'], text: 'R' });
    expect(manaSymbol('3')).toEqual({ colours: [], text: '3' });
    expect(manaSymbol('T')).toEqual({ colours: [], text: '↷' });
  });
  it('splits hybrid symbols into two colours with no letters', () => {
    expect(manaSymbol('W/U')).toEqual({ colours: ['w', 'u'], text: '' });
  });
  it('keeps the number of a two-or-colour hybrid', () => {
    expect(manaSymbol('2/R')).toEqual({ colours: ['r'], text: '2' });
  });
  it('marks Phyrexian mana', () => {
    expect(manaSymbol('B/P')).toEqual({ colours: ['b'], text: 'φ' });
    expect(manaSymbol('G/U/P')).toEqual({ colours: ['g', 'u'], text: 'φ' });
  });
});

import { describe, expect, it } from 'vitest';
import { findCards, remember } from './search.js';

const names = (list, aliases = []) => ({ names: list, aliases });
const found = (query, list, limit) => findCards(query, list, limit).map((m) => m.name);

describe('findCards', () => {
  it('ranks exact, then prefix, then word start, then substring', () => {
    const list = names(['Stormkin Bolt', 'Bolt Wing', 'Bolt', 'Thunderbolt']);
    expect(found('bolt', list)).toEqual(['Bolt', 'Bolt Wing', 'Stormkin Bolt', 'Thunderbolt']);
  });

  it('puts shorter names first within a tier', () => {
    const list = names(['Lightning Stormkin', 'Light-Paws, Emperor\'s Voice', 'Lightning Bolt']);
    expect(found('light', list)).toEqual(['Lightning Bolt', 'Lightning Stormkin', 'Light-Paws, Emperor\'s Voice']);
  });

  it('matches every word in any order when the whole query does not match', () => {
    const list = names(['Jace, the Mind Sculptor', 'Mind Stone', 'Jace Beleren']);
    expect(found('jace mind', list)).toEqual(['Jace, the Mind Sculptor']);
    expect(found('sculptor jace', list)).toEqual(['Jace, the Mind Sculptor']);
  });

  it('matches aliases and accents, and returns the card index', () => {
    const list = names(['Lim-Dûl the Necromancer', 'Green Thread // Thread Unbound'], [['Thread Unbound', 1]]);
    expect(findCards('lim-dul', list)).toEqual([{ index: 0, name: 'Lim-Dûl the Necromancer' }]);
    expect(found('thread unbound', list)).toEqual(['Green Thread // Thread Unbound']);
  });

  it('lists a card once and stops at the limit', () => {
    const list = names(['Bolt', 'Bolt Wing', 'Bolt Hound'], [['Bolt Again', 0]]);
    expect(found('bolt', list)).toEqual(['Bolt', 'Bolt Wing', 'Bolt Hound']);
    expect(found('bolt', list, 2)).toEqual(['Bolt', 'Bolt Wing']);
    expect(found('  ', list)).toEqual([]);
  });
});

describe('remember', () => {
  it('puts the newest pick first without duplicates, up to max', () => {
    const a = { id: 'a', name: 'A' }, b = { id: 'b', name: 'B' }, c = { id: 'c', name: 'C' };
    expect(remember([a, b], b)).toEqual([b, a]);
    expect(remember([a, b], c, 2)).toEqual([c, a]);
    expect(remember([], a)).toEqual([a]);
  });
});

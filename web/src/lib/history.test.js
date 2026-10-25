import { describe, expect, it } from 'vitest';
import { entryKey, newEntryId } from './history.js';

describe('history entry keys', () => {
  it('gives each new entry its own id', () => {
    const ids = Array.from({ length: 100 }, newEntryId);
    expect(new Set(ids).size).toBe(100);
    for (const id of ids) expect(id).not.toMatch(/^\?|^$/);
  });

  it('keys an entry by its id, or by its query when it has none', () => {
    const id = newEntryId();
    expect(entryKey(id, '?card=a')).toBe(id);
    expect(entryKey(undefined, '?card=a')).toBe('?card=a');
    expect(entryKey(null, '')).toBe('');
  });
});

import { describe, expect, it } from 'vitest';
import metrics from './generated/metrics.json';
import { PLAIN_PRINTINGS } from './printings.js';

describe('PLAIN_PRINTINGS', () => {
  it('covers every landing-page example and only uses Scryfall card images', () => {
    for (const ex of metrics.showcase.filter((s) => !s.unseen)) expect(PLAIN_PRINTINGS[ex.id], ex.name).toBeDefined();
    for (const url of Object.values(PLAIN_PRINTINGS)) expect(url).toMatch(/^https:\/\/cards\.scryfall\.io\/normal\/front\/[0-9a-f/-]+\.jpg(\?\d+)?$/);
  });
});

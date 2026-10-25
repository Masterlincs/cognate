import { describe, expect, it } from 'vitest';
import { ceilTo, contrastRows, runScore, ticks } from './study.js';
import m from './generated/metrics.json';

describe('study helpers', () => {
  it('contrastRows gives two views per contrast, in label order, skipping missing ones', () => {
    const rows = contrastRows(m.sealed, { trained_bge: 'BGE', incumbent_e5: 'old E5', pretrained_e5: 'E5' });
    expect(rows.map((r) => r.key)).toEqual(['trained_bge-full', 'trained_bge-hard_lexical', 'pretrained_e5-full', 'pretrained_e5-hard_lexical']);
    const [r] = rows;
    const c = m.sealed.contrasts.trained_bge.full;
    expect(r).toMatchObject({ label: 'BGE', view: 'full', first: true, point: c.point, lo: c.ci95[0], hi: c.ci95[1], adj: c.lower_adjusted });
    expect(rows[1].first).toBe(false);
  });

  it('contrastRows tolerates a missing report', () => {
    expect(contrastRows(undefined, { a: 'A' })).toEqual([]);
  });

  it('ticks are inclusive and free of float drift', () => {
    expect(ticks(0, 0.8, 0.2)).toEqual([0, 0.2, 0.4, 0.6, 0.8]);
    expect(ticks(0.7, 0.72, 0.005)).toEqual([0.7, 0.705, 0.71, 0.715, 0.72]);
  });

  it('ceilTo rounds up to a step and keeps exact multiples', () => {
    expect(ceilTo(0.171, 0.02)).toBeCloseTo(0.18);
    expect(ceilTo(0.16, 0.02)).toBeCloseTo(0.16);
  });

  it('runScore finds a run inside a multi-seed row', () => {
    const row = m.dev['fine-tuned E5, 4000 updates'];
    expect(runScore(row, row.runs[1])).toBe(row.full[1]);
    expect(runScore(row, row.runs[2], 'hard_lexical')).toBe(row.hard_lexical[2]);
    expect(runScore(row, 'nope')).toBeNull();
    expect(runScore(undefined, 'x')).toBeNull();
  });
});

import { describe, expect, it } from 'vitest';
import { decisionNote, modelLabel, qualifierText } from './wording.js';
import m from './generated/metrics.json';

describe('decisionNote', () => {
  it('states the promotion decision and its limits in plain words', () => {
    expect(decisionNote({ decision: 'promote', qualifier: 'provisional tag-proxy promotion on selection-exposed development' }))
      .toBe('Decision: promoted. The decision is provisional. It uses the tag proxy on development cards, and those cards also helped me choose the model.');
  });

  it('names the other report decisions', () => {
    expect(decisionNote({ decision: 'reject' })).toBe('Decision: not promoted.');
    expect(decisionNote({ decision: 'inconclusive' })).toBe('Decision: inconclusive.');
  });

  it('falls back to the raw decision and qualifier', () => {
    expect(decisionNote({ decision: 'hold', qualifier: 'some new wording' })).toBe('Decision: hold (some new wording).');
  });

  it('covers the stored promotion reports', () => {
    for (const report of Object.values(m.promotion)) expect(decisionNote(report)).not.toMatch(/selection-exposed|tag-proxy promotion/);
  });
});

describe('qualifierText', () => {
  it('keeps the sealed re-test separate from the promotion decision', () => {
    expect(qualifierText(m.sealed.qualifier)).toBe('I re-tested the registered gate once on the sealed cards, with the tag proxy. This re-test does not change the promotion decision.');
  });

  it('returns unknown text unchanged', () => expect(qualifierText('raw text')).toBe('raw text'));
});

describe('modelLabel', () => {
  const deployed = { run_id: 'run-a', encoder: { model_id: 'intfloat/e5-base-v2' }, training: { total_updates: 4000 } };

  it('builds the label from the backbone and the deployed run', () => {
    expect(modelLabel({ label: 'raw', backbone: 'intfloat/e5-base-v2', run_id: 'run-a' }, deployed)).toBe('e5-base-v2, fine-tuned for 4,000 updates');
  });

  it('leaves out the update count when the deployed run is another run', () => {
    expect(modelLabel({ label: 'raw', backbone: 'intfloat/e5-base-v2', run_id: 'run-b' }, deployed)).toBe('e5-base-v2, fine-tuned');
  });

  it('falls back to the stored label without a backbone', () => {
    expect(modelLabel({ label: 'raw label', run_id: 'run-a' }, deployed)).toBe('raw label');
    expect(modelLabel(undefined, deployed)).toBe('');
  });
});

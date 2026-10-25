// Plain-English display text for strings that come from the stored reports and release.json.
// The source strings stay unchanged; unknown strings are shown as they are.

const DECISIONS = { promote: 'promoted', reject: 'not promoted', inconclusive: 'inconclusive' };

// keyed by QUALIFIER and SEALED_QUALIFIER in src/mtgemb/report.py
const QUALIFIERS = {
  'provisional tag-proxy promotion on selection-exposed development':
    'The decision is provisional. It uses the tag proxy on development cards, and those cards also helped me choose the model.',
  'registered gate re-tested once on sealed anchors (tag proxy); does not change the promotion decision':
    'I re-tested the registered gate once on the sealed cards, with the tag proxy. This re-test does not change the promotion decision.',
};

export const qualifierText = (qualifier) => QUALIFIERS[qualifier] ?? qualifier;

export function decisionNote({ decision, qualifier } = {}) {
  const verdict = `Decision: ${DECISIONS[decision] ?? decision}`;
  if (!qualifier) return `${verdict}.`;
  return QUALIFIERS[qualifier] ? `${verdict}. ${QUALIFIERS[qualifier]}` : `${verdict} (${qualifier}).`;
}

// release.model ({label, backbone, run_id}); the update count comes from metrics.deployed for the same run
export function modelLabel(model, deployed) {
  if (!model?.backbone) return model?.label ?? '';
  const name = model.backbone.split('/').pop();
  const updates = deployed?.run_id === model.run_id ? deployed.training?.total_updates : undefined;
  return Number.isInteger(updates) ? `${name}, fine-tuned for ${updates.toLocaleString('en')} updates` : `${name}, fine-tuned`;
}

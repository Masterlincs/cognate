// Helpers for the how-it-works page; inputs are shaped like generated/metrics.json.

const VIEWS = [['full', 'full'], ['hard_lexical', 'hard-lexical']];

// Rows for a confidence-interval plot, one per contrast and view, in the order of `labels`
// ({contrast key: label}); contrasts missing from the report are skipped.
export function contrastRows(report, labels) {
  return Object.entries(labels).flatMap(([key, label]) => {
    const c = report?.contrasts?.[key];
    if (!c) return [];
    return VIEWS.map(([view, viewLabel]) => ({
      key: `${key}-${view}`, label, view: viewLabel, first: view === 'full',
      point: c[view].point, lo: c[view].ci95[0], hi: c[view].ci95[1], adj: c[view].lower_adjusted,
    }));
  });
}

// Evenly spaced ticks from lo to hi inclusive, rounded so that 0.1 + 0.2 prints as 0.3.
export function ticks(lo, hi, step) {
  const n = Math.round((hi - lo) / step);
  const digits = Math.max(0, -Math.floor(Math.log10(step)) + 1);
  return Array.from({ length: n + 1 }, (_, i) => Number((lo + i * step).toFixed(digits)));
}

// The smallest multiple of step at or above x.
export const ceilTo = (x, step) => Math.ceil(x / step - 1e-9) * step;

// One run's score from a dev row ({runs, full, hard_lexical}), or null.
export function runScore(row, runId, view = 'full') {
  const i = row?.runs.indexOf(runId) ?? -1;
  return i >= 0 ? row[view][i] : null;
}

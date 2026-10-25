<script>
  import { delta, fixed } from './format.js';
  import { ceilTo, ticks as makeTicks } from './study.js';

  // panels: [{title, note, rows}] with rows from contrastRows(); one shared axis from zero, so panels compare directly.
  // Each row: a point estimate, its 95% interval and the adjusted one-sided lower bound.
  let { panels, step = 0.02, caption = '' } = $props();
  const max = $derived(ceilTo(Math.max(...panels.flatMap((p) => p.rows.map((r) => r.hi))), step));
  const ticks = $derived(makeTicks(0, max, step));
  const pct = (x) => `${(x / max) * 100}%`;
  const describe = (r) => `${r.label}, ${r.view}: ${delta(r.point)}, 95% interval ${fixed(r.lo)} to ${fixed(r.hi)}, adjusted lower bound ${fixed(r.adj)}`;
</script>

<figure class="ci">
  <div class="legend" aria-hidden="true">
    <span><i class="k-dot"></i>difference in nDCG@10</span>
    <span><i class="k-int"></i>95% interval</span>
    <span><i class="k-adj"></i>adjusted lower bound</span>
  </div>
  {#each panels as p (p.title)}
    <div class="panel">
      <h4>{p.title}{#if p.note}<span>{p.note}</span>{/if}</h4>
      <div class="rows" aria-hidden="true">
        {#each p.rows as r (r.key)}
          <div class="row" class:first={r.first}>
            <span class="label">{#if r.first}<b>vs {r.label}</b>{/if}<small>{r.view}</small></span>
            <div class="track" title={describe(r)}>
              {#each ticks as t (t)}<span class="grid" class:zero={t === 0} style:left={pct(t)}></span>{/each}
              <span class="int" style:left={pct(r.lo)} style:width={pct(r.hi - r.lo)}></span>
              <span class="adj" style:left={pct(r.adj)}></span>
              <span class="dot" style:left={pct(r.point)}></span>
            </div>
            <span class="value num">{delta(r.point)}</span>
          </div>
        {/each}
      </div>
    </div>
  {/each}
  <div class="row axis" aria-hidden="true">
    <span class="axis-label">0 = no difference</span>
    <div class="ticks">{#each ticks as t (t)}<span class="num" style:left={pct(t)}>{t === 0 ? '0' : `+${t.toFixed(2)}`}</span>{/each}</div>
    <span></span>
  </div>
  {#if caption}<figcaption>{caption}</figcaption>{/if}
  {#each panels as p (p.title)}
    <div class="sr-only"><table>
      <caption>{p.title}{p.note ? `, ${p.note}` : ''}</caption>
      <thead><tr><th scope="col">Comparison</th><th scope="col">View</th><th scope="col">Difference</th><th scope="col">95% interval</th><th scope="col">Adjusted lower bound</th></tr></thead>
      <tbody>{#each p.rows as r (r.key)}<tr><th scope="row">vs {r.label}</th><td>{r.view}</td><td>{delta(r.point)}</td><td>{fixed(r.lo)} to {fixed(r.hi)}</td><td>{fixed(r.adj)}</td></tr>{/each}</tbody>
    </table></div>
  {/each}
</figure>

<style>
  .ci { margin: 0; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 0.82rem; color: var(--muted); margin-bottom: 14px; }
  .legend span { display: flex; align-items: center; gap: 7px; }
  .legend i { display: inline-block; flex: none; }
  .k-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--ink); }
  .k-int { width: 18px; height: 3px; border-radius: 2px; background: var(--muted); }
  .k-adj { width: 2px; height: 12px; background: var(--ink); }
  .panel + .panel { margin-top: 22px; }
  h4 { margin: 0 0 8px; font-size: 0.92rem; font-weight: 650; }
  h4 span { display: block; font-weight: 400; font-size: 0.8rem; color: var(--muted); }
  .rows { display: grid; }
  .row { display: grid; grid-template-columns: minmax(130px, 28%) 1fr 58px; gap: 14px; align-items: center; min-height: 26px; }
  .row.first:not(:first-child) { margin-top: 8px; }
  .label { font-size: 0.86rem; line-height: 1.2; }
  .label b { display: block; font-weight: 600; }
  .label small { color: var(--muted); font-size: 0.75rem; }
  .track { position: relative; align-self: stretch; }
  .grid { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--line); }
  .grid.zero { width: 2px; background: var(--muted); }
  .int { position: absolute; top: 50%; height: 3px; margin-top: -1.5px; border-radius: 2px; background: var(--muted); min-width: 3px; }
  .adj { position: absolute; top: 50%; width: 2px; height: 14px; margin: -7px 0 0 -1px; background: var(--ink); }
  .dot { position: absolute; top: 50%; width: 11px; height: 11px; margin: -5.5px 0 0 -5.5px; border-radius: 50%; background: var(--ink); box-shadow: 0 0 0 2px var(--bg); }
  .value { font-size: 0.8rem; text-align: right; }
  .row:not(.first) .value { color: var(--muted); }
  .axis { margin-top: 4px; min-height: 0; align-items: start; }
  .axis-label { font-size: 0.75rem; color: var(--muted); }
  .ticks { position: relative; height: 16px; font-size: 0.7rem; color: var(--muted); }
  .ticks span { position: absolute; transform: translateX(-50%); }
  .ticks span:first-child { transform: none; }
  figcaption { margin-top: 14px; font-size: 0.84rem; color: var(--muted); max-width: 68ch; }
  @media (max-width: 560px) {
    .row { grid-template-columns: 100px 1fr 48px; gap: 8px; }
    .label b { font-size: 0.8rem; }
    .ticks span:nth-child(even) { visibility: hidden; }
  }
</style>

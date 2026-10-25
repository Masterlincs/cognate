<script>
  import { fixed, mean } from './format.js';
  import { ticks as makeTicks } from './study.js';

  // rows: [{label, values: {key: number[]}, note?, highlight?}]; series: [{key, label}], first series drawn thicker.
  // Bars start at zero and show the mean; the per-seed values are in the hover title and the table.
  let { rows, series, max = 0.8, step = 0.2, axisLabel = '', caption = '' } = $props();
  const pct = (x) => `${(x / max) * 100}%`;
  const ticks = $derived(makeTicks(0, max, step));
  const title = (row, s) => `${row.label}, ${s.label}: ${row.values[s.key].map((v) => fixed(v)).join(', ')}`;
</script>

<figure class="bars">
  <div class="legend" aria-hidden="true">
    {#each series as s, k (s.key)}<span><i class="s{k}"></i>{s.label}</span>{/each}
  </div>
  <div class="plot" aria-hidden="true">
    {#each rows as row (row.label)}
      <div class="row" class:highlight={row.highlight}>
        <div class="label">{row.label}{#if row.note}<small>{row.note}</small>{/if}</div>
        <div class="track">
          {#each ticks as t (t)}<span class="grid" class:zero={t === 0} style:left={pct(t)}></span>{/each}
          {#each series as s, k (s.key)}
            {@const v = mean(row.values[s.key])}
            <div class="line s{k}" title={title(row, s)}>
              <div class="bar" style:width={pct(v)}></div>
              <span class="value num" style:left={pct(v)}>{fixed(v)}</span>
            </div>
          {/each}
        </div>
      </div>
    {/each}
    <div class="row axis">
      <span class="axis-label">{axisLabel}</span>
      <div class="ticks">{#each ticks as t (t)}<span class="num" style:left={pct(t)}>{t.toFixed(1)}</span>{/each}</div>
    </div>
  </div>
  {#if caption}<figcaption>{caption}</figcaption>{/if}
  <div class="sr-only"><table>
    <caption>{caption}</caption>
    <thead><tr><th scope="col">System</th>{#each series as s (s.key)}<th scope="col">{s.label}</th>{/each}</tr></thead>
    <tbody>{#each rows as row (row.label)}<tr><th scope="row">{row.label}{row.note ? ` (${row.note})` : ''}</th>{#each series as s (s.key)}<td>{row.values[s.key].map((v) => fixed(v)).join(', ')}</td>{/each}</tr>{/each}</tbody>
  </table></div>
</figure>

<style>
  .bars { margin: 0; }
  .legend { display: flex; gap: 18px; font-size: 0.82rem; color: var(--muted); margin-bottom: 12px; }
  .legend span { display: flex; align-items: center; gap: 7px; }
  .legend i { width: 14px; border-radius: 2px; background: var(--muted); }
  .legend .s0 { height: 10px; }
  .legend .s1 { height: 6px; opacity: 0.5; }
  .plot { display: grid; gap: 14px; }
  .row { display: grid; grid-template-columns: minmax(120px, 32%) 1fr; gap: 16px; align-items: center; }
  .label { font-size: 0.88rem; line-height: 1.25; }
  .label small { display: block; margin-top: 2px; color: var(--muted); font-size: 0.75rem; }
  .highlight .label { font-weight: 650; }
  /* the right margin leaves room for the value printed after the longest bar */
  .track { position: relative; display: grid; gap: 3px; margin-right: 46px; padding: 3px 0; }
  .grid { position: absolute; top: -7px; bottom: -7px; width: 1px; background: var(--line); }
  .grid.zero { background: var(--muted); }
  .line { position: relative; display: flex; align-items: center; }
  .s0 { height: 12px; }
  .s1 { height: 7px; }
  .bar { height: 100%; border-radius: 0 3px 3px 0; background: var(--wubrg-c); }
  .s1 .bar { opacity: 0.5; }
  .highlight .bar { background: var(--ink); }
  .value { position: absolute; padding-left: 6px; font-size: 0.72rem; line-height: 1; color: var(--muted); white-space: nowrap; }
  .highlight .value { color: var(--ink); font-weight: 600; }
  .axis { align-items: start; margin-top: -6px; }
  .axis-label { font-size: 0.75rem; color: var(--muted); text-align: right; }
  .ticks { position: relative; height: 16px; margin-right: 46px; font-size: 0.7rem; color: var(--muted); }
  .ticks span { position: absolute; transform: translateX(-50%); }
  .ticks span:first-child { transform: none; }
  figcaption { margin-top: 14px; font-size: 0.84rem; color: var(--muted); max-width: 68ch; }
  @media (max-width: 560px) {
    .row { grid-template-columns: minmax(0, 1fr); gap: 6px; }
    .axis-label { text-align: left; }
  }
</style>

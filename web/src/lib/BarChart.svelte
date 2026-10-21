<script>
  import { fixed, mean } from './format.js';

  // rows: [{label, values: {key: number[]}, note?, highlight?}]; series: [{key, label}]
  // Bars show the mean; a whisker spans the seed range when there is more than one seed.
  let { rows, series, max = 0.8, ticks = [0, 0.2, 0.4, 0.6, 0.8], caption = '' } = $props();
  const pct = (x) => `${(x / max) * 100}%`;
</script>

<figure class="bars">
  <div class="legend" aria-hidden="true">
    {#each series as s, k (s.key)}<span><i class="s{k}"></i>{s.label}</span>{/each}
  </div>
  <div class="rows">
    {#each rows as row (row.label)}
      <div class="row" class:highlight={row.highlight}>
        <div class="label">{row.label}{#if row.note}<small>{row.note}</small>{/if}</div>
        <div class="track">
          {#each ticks as t (t)}<span class="grid" style:left={pct(t)}></span>{/each}
          {#each series as s, k (s.key)}
            {@const vals = row.values[s.key]}
            <div class="line">
              <div class="bar s{k}" style:width={pct(mean(vals))}></div>
              {#if vals.length > 1}<span class="whisker" style:left={pct(Math.min(...vals))} style:width={pct(Math.max(...vals) - Math.min(...vals))}></span>{/if}
              <span class="value num">{fixed(mean(vals))}</span>
            </div>
          {/each}
        </div>
      </div>
    {/each}
  </div>
  <div class="axis" aria-hidden="true">
    <span></span>
    <div class="ticks">{#each ticks as t (t)}<span style:left={pct(t)}>{t.toFixed(1)}</span>{/each}</div>
  </div>
  {#if caption}<figcaption>{caption}</figcaption>{/if}
  <table class="sr-only">
    <caption>{caption}</caption>
    <thead><tr><th>Model</th>{#each series as s (s.key)}<th>{s.label}</th>{/each}</tr></thead>
    <tbody>{#each rows as row (row.label)}<tr><th>{row.label}</th>{#each series as s (s.key)}<td>{row.values[s.key].map((v) => fixed(v)).join(', ')}</td>{/each}</tr>{/each}</tbody>
  </table>
</figure>

<style>
  .bars { margin: 0; }
  .legend { display: flex; gap: 16px; font-size: 0.82rem; color: var(--muted); margin-bottom: 10px; }
  .legend span { display: flex; align-items: center; gap: 6px; }
  .legend i { width: 12px; height: 8px; border-radius: 2px; }
  .rows { display: grid; gap: 10px; }
  .row, .axis { display: grid; grid-template-columns: minmax(110px, 30%) 1fr; gap: 12px; align-items: center; }
  .label { font-size: 0.86rem; line-height: 1.25; }
  .label small { display: block; color: var(--muted); font-size: 0.74rem; }
  .highlight .label { font-weight: 650; }
  .track { position: relative; display: grid; gap: 3px; padding: 2px 0; }
  .grid { position: absolute; top: -5px; bottom: -5px; width: 1px; background: var(--line); }
  .line { position: relative; height: 12px; }
  .bar { height: 12px; border-radius: 0 3px 3px 0; }
  .s0 { background: var(--wubrg-c); }
  .s1 { background: color-mix(in srgb, var(--wubrg-c) 45%, transparent); }
  .highlight .s0 { background: var(--accent); }
  .highlight .s1 { background: color-mix(in srgb, var(--accent) 45%, transparent); }
  .legend .s0 { background: var(--accent); }
  .legend .s1 { background: color-mix(in srgb, var(--accent) 45%, transparent); }
  .whisker { position: absolute; top: 50%; height: 2px; margin-top: -1px; background: var(--ink); min-width: 2px; }
  .value { position: absolute; right: 0; top: 0; font-size: 0.72rem; color: var(--muted); line-height: 12px; }
  .ticks { position: relative; height: 16px; font: 0.7rem var(--sans); color: var(--muted); }
  .ticks span { position: absolute; transform: translateX(-50%); }
  .ticks span:first-child { transform: none; }
  .ticks span:last-child { transform: translateX(-100%); }
  figcaption { margin-top: 10px; font-size: 0.84rem; color: var(--muted); }
</style>

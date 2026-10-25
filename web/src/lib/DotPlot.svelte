<script>
  import { delta, fixed, scaleLinear } from './format.js';
  import { ticks as makeTicks } from './study.js';

  // rows: [{label, a, b}] drawn as a dot at a, a dot at b and a line between them.
  // The axis is zoomed to the data (it does not start at zero), and says so.
  let { rows, aLabel, bLabel, caption = '' } = $props();
  const all = $derived(rows.flatMap((r) => [r.a, r.b]));
  const lo = $derived(Math.floor((Math.min(...all) - 0.002) * 200) / 200);
  const hi = $derived(Math.ceil((Math.max(...all) + 0.002) * 200) / 200);
  const x = $derived(scaleLinear([lo, hi], [0, 100]));
  const ticks = $derived(makeTicks(lo, hi, 0.005));
</script>

<figure class="dots">
  <div class="legend" aria-hidden="true"><span><i class="a"></i>{aLabel}</span><span><i class="b"></i>{bLabel}</span></div>
  <div class="plot" aria-hidden="true">
    {#each rows as r (r.label)}
      <div class="row">
        <span class="label">{r.label}</span>
        <div class="track" title="{r.label}: {aLabel} {fixed(r.a)}, {bLabel} {fixed(r.b)}">
          {#each ticks as t (t)}<span class="grid" style:left="{x(t)}%"></span>{/each}
          <span class="link" style:left="{Math.min(x(r.a), x(r.b))}%" style:width="{Math.abs(x(r.b) - x(r.a))}%"></span>
          <span class="dot a" style:left="{x(r.a)}%"></span>
          <span class="dot b" style:left="{x(r.b)}%"></span>
        </div>
        <span class="num d">{delta(r.b - r.a)}</span>
      </div>
    {/each}
    <div class="row axis">
      <span></span>
      <div class="ticks">{#each ticks as t (t)}<span class="num" style:left="{x(t)}%">{t.toFixed(3)}</span>{/each}</div>
      <span></span>
    </div>
  </div>
  {#if caption}<figcaption>{caption}</figcaption>{/if}
  <div class="sr-only"><table>
    <caption>{caption}</caption>
    <thead><tr><th scope="col">Run</th><th scope="col">{aLabel}</th><th scope="col">{bLabel}</th><th scope="col">Change</th></tr></thead>
    <tbody>{#each rows as r (r.label)}<tr><th scope="row">{r.label}</th><td>{fixed(r.a)}</td><td>{fixed(r.b)}</td><td>{delta(r.b - r.a)}</td></tr>{/each}</tbody>
  </table></div>
</figure>

<style>
  .dots { margin: 0; }
  .legend { display: flex; gap: 18px; font-size: 0.82rem; color: var(--muted); margin-bottom: 10px; }
  .legend span { display: flex; align-items: center; gap: 7px; }
  .legend i { width: 10px; height: 10px; border-radius: 50%; }
  .row { display: grid; grid-template-columns: 64px 1fr 56px; gap: 14px; align-items: center; height: 32px; }
  .label { font-size: 0.86rem; }
  .track { position: relative; height: 100%; }
  .grid { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--line); }
  .link { position: absolute; top: 50%; height: 2px; margin-top: -1px; background: var(--muted); }
  .dot { position: absolute; top: 50%; width: 12px; height: 12px; margin: -6px 0 0 -6px; border-radius: 50%; box-shadow: 0 0 0 2px var(--bg); }
  .a { background: var(--wubrg-c); }
  .b { background: var(--ink); }
  .d { font-size: 0.8rem; text-align: right; }
  .axis { height: 18px; align-items: start; }
  .ticks { position: relative; height: 100%; font-size: 0.7rem; color: var(--muted); }
  .ticks span { position: absolute; transform: translateX(-50%); }
  figcaption { margin-top: 12px; font-size: 0.84rem; color: var(--muted); max-width: 68ch; }
  @media (max-width: 560px) {
    .row { grid-template-columns: 52px 1fr 48px; gap: 8px; }
    .ticks span:nth-child(even) { visibility: hidden; }
  }
</style>

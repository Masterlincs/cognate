<script>
  import { delta, fixed, scaleLinear } from './format.js';

  // rows: [{label, a, b}] drawn as a dot at a, a dot at b and an arrow between them
  let { rows, aLabel, bLabel, caption = '' } = $props();
  const all = $derived(rows.flatMap((r) => [r.a, r.b]));
  const lo = $derived(Math.floor((Math.min(...all) - 0.002) * 200) / 200);
  const hi = $derived(Math.ceil((Math.max(...all) + 0.002) * 200) / 200);
  const x = $derived(scaleLinear([lo, hi], [0, 100]));
  const ticks = $derived(Array.from({ length: Math.round((hi - lo) / 0.005) + 1 }, (_, i) => lo + i * 0.005));
</script>

<figure class="dots">
  <div class="legend" aria-hidden="true"><span><i class="a"></i>{aLabel}</span><span><i class="b"></i>{bLabel}</span></div>
  {#each rows as r (r.label)}
    <div class="row">
      <span class="label">{r.label}</span>
      <div class="track">
        {#each ticks as t (t)}<span class="grid" style:left="{x(t)}%"></span>{/each}
        <span class="link" style:left="{Math.min(x(r.a), x(r.b))}%" style:width="{Math.abs(x(r.b) - x(r.a))}%"></span>
        <span class="dot a" style:left="{x(r.a)}%" title="{aLabel}: {fixed(r.a)}"></span>
        <span class="dot b" style:left="{x(r.b)}%" title="{bLabel}: {fixed(r.b)}"></span>
      </div>
      <span class="num d">{delta(r.b - r.a)}</span>
    </div>
  {/each}
  <div class="row axis" aria-hidden="true">
    <span></span>
    <div class="ticks">{#each ticks as t (t)}<span style:left="{x(t)}%">{t.toFixed(3)}</span>{/each}</div>
    <span></span>
  </div>
  {#if caption}<figcaption>{caption}</figcaption>{/if}
  <table class="sr-only">
    <caption>{caption}</caption>
    <thead><tr><th>Run</th><th>{aLabel}</th><th>{bLabel}</th></tr></thead>
    <tbody>{#each rows as r (r.label)}<tr><th>{r.label}</th><td>{fixed(r.a)}</td><td>{fixed(r.b)}</td></tr>{/each}</tbody>
  </table>
</figure>

<style>
  .dots { margin: 0; }
  .legend { display: flex; gap: 16px; font-size: 0.82rem; color: var(--muted); margin-bottom: 10px; }
  .legend span { display: flex; align-items: center; gap: 6px; }
  .legend i { width: 10px; height: 10px; border-radius: 50%; }
  .row { display: grid; grid-template-columns: 64px 1fr 56px; gap: 12px; align-items: center; height: 30px; }
  .label { font-size: 0.84rem; }
  .track { position: relative; height: 100%; }
  .grid { position: absolute; top: 0; bottom: 0; width: 1px; background: var(--line); }
  .link { position: absolute; top: 50%; height: 2px; margin-top: -1px; background: var(--muted); }
  .dot { position: absolute; top: 50%; width: 12px; height: 12px; margin: -6px 0 0 -6px; border-radius: 50%; border: 2px solid var(--surface); }
  .a { background: var(--wubrg-c); }
  .b { background: var(--accent); }
  .d { font-size: 0.78rem; text-align: right; color: var(--muted); }
  .axis { height: 18px; }
  .ticks { position: relative; height: 100%; font: 0.68rem var(--mono); color: var(--muted); }
  .ticks span { position: absolute; transform: translateX(-50%); }
  figcaption { margin-top: 10px; font-size: 0.84rem; color: var(--muted); }
</style>

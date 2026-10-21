<script>
  import { onMount } from 'svelte';
  import { asset, resolve } from '$app/paths';
  import Search from './Search.svelte';
  import { getCard, getNeighbours, getPretrained, loadRelease, loadSiteData, setRoot } from './data.js';
  import { resolveSelection } from './explorer.js';

  // showcase: [{id, name, why}]; data loads when the section scrolls into view
  let { showcase = [] } = $props();
  const TOP = 8;
  let root;
  let names = $state(null);
  let ids = null;
  let phase = $state('idle');
  let current = $state(null);
  let cols = $state(null);
  let error = $state('');
  let token = 0;

  const firstLine = (c) => (c.faces.map((f) => f.oracle_text).join(' // ').split('\n')[0] ?? '');

  async function show(index) {
    const t = ++token;
    error = '';
    try {
      const [card, tuned, pre] = await Promise.all([getCard(index), getNeighbours(index), getPretrained(index)]);
      if (pre === null) { phase = 'missing'; return; }
      if (!card.has_text) { if (t === token) { current = card; cols = null; } return; }
      const load = (list) => Promise.all(list.slice(0, TOP).map(([i, s]) => getCard(i).then((c) => ({ i, c, s }))));
      const [a, b] = await Promise.all([load(pre), load(tuned)]);
      if (t !== token) return;
      current = card;
      const inA = new Set(a.map((r) => r.i)), inB = new Set(b.map((r) => r.i));
      cols = { pre: a.map((r) => ({ ...r, both: inB.has(r.i) })), tuned: b.map((r) => ({ ...r, both: inA.has(r.i) })), overlap: a.filter((r) => inB.has(r.i)).length };
    } catch (e) { if (t === token) error = e.message; }
  }

  const byId = (id) => resolveSelection(ids, { card: id }).index;

  async function start() {
    if (phase !== 'idle') return;
    phase = 'loading';
    try {
      setRoot(asset(''));
      const r = await loadRelease(fetch);
      names = r.names;
      const site = await loadSiteData(r.release.release_id);
      ids = await r.idsReady;
      if (!site) { phase = 'missing'; return; }
      phase = 'ready';
      const first = showcase.map((s) => byId(s.id)).find((i) => i >= 0);
      if (first !== undefined) show(first);
    } catch (e) { error = e.message; phase = 'error'; }
  }

  onMount(() => {
    const io = new IntersectionObserver((entries) => { if (entries.some((e) => e.isIntersecting)) { io.disconnect(); start(); } }, { rootMargin: '300px' });
    io.observe(root);
    return () => io.disconnect();
  });
</script>

<div class="ba" bind:this={root}>
  <div class="pick">
    <div class="chips" role="group" aria-label="Example cards">
      {#each showcase as s (s.id)}
        <button type="button" class="chip" class:on={current?.id === s.id} disabled={phase !== 'ready'} onclick={() => show(byId(s.id))}>{s.name}</button>
      {/each}
    </div>
    <div class="search"><Search {names} id="ba-search" label="Compare any card" placeholder="Or type any card…" onpick={(i) => show(i)} /></div>
  </div>

  {#if phase === 'missing'}
    <p class="notice">The before-and-after comparison is not available for this release.</p>
  {:else if error}
    <p class="notice" role="alert">Could not load the comparison: {error}</p>
  {:else if current && !current.has_text}
    <p class="notice">{current.name} has no rules text, so neither model has anything to compare.</p>
  {:else if cols && current}
    <p class="summary">
      <strong>{current.name}</strong>: {showcase.find((s) => s.id === current.id)?.why ?? 'a card you picked.'}
      <span class="muted">{cols.overlap} of the top {TOP} appear in both lists.</span>
      <a href="{resolve('/')}?card={current.id}">Open in the explorer →</a>
    </p>
    <div class="cols">
      {#each [['Before', 'pretrained e5-base-v2', cols.pre], ['After', 'fine-tuned on card tags', cols.tuned]] as [title, sub, rows] (title)}
        <section class:after={title === 'After'}>
          <h3>{title} <span>{sub}</span></h3>
          <ol>
            {#each rows as r (r.i)}
              <li class:both={r.both}>
                <span class="name">{r.c.name}</span>
                <span class="line">{firstLine(r.c)}</span>
              </li>
            {/each}
          </ol>
        </section>
      {/each}
    </div>
    <p class="key muted"><span class="both-key"></span> in both lists</p>
  {:else}
    <div class="cols" aria-hidden="true"><section class="ghost"></section><section class="ghost"></section></div>
  {/if}
</div>

<style>
  .pick { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between; margin-bottom: 16px; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip { border: 1px solid var(--line); background: var(--surface); color: var(--ink); padding: 5px 11px; font-size: 0.84rem; }
  .chip.on { border-color: var(--accent); background: var(--accent-soft); }
  .chip:disabled { opacity: 0.6; cursor: default; }
  .search { width: min(100%, 300px); }
  .search :global(input) { height: 40px; box-shadow: none; }
  .summary { font-size: 0.95rem; }
  .summary a { white-space: nowrap; }
  .cols { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }
  section { padding: 14px 16px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); min-width: 0; }
  section.after { border-color: color-mix(in srgb, var(--accent) 50%, var(--line)); }
  section.ghost { min-height: 420px; background: var(--sunken); border: 0; }
  h3 { font-size: 1.1rem; margin-bottom: 8px; }
  h3 span { display: block; font: 400 0.78rem var(--body); color: var(--muted); }
  ol { margin: 0; padding-left: 1.4em; }
  li { padding: 6px 0; border-top: 1px solid var(--line); font-size: 0.86rem; }
  li::marker { color: var(--muted); font: 0.75rem var(--mono); }
  .name { display: block; font-weight: 600; }
  .line { display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; line-clamp: 2; overflow: hidden; color: var(--muted); }
  li.both .name { box-shadow: inset 0 -0.45em var(--mark); }
  .key { font-size: 0.8rem; margin-top: 8px; display: flex; align-items: center; gap: 6px; }
  .both-key { display: inline-block; width: 22px; height: 8px; background: var(--mark); }
  @media (max-width: 640px) { .cols { grid-template-columns: minmax(0, 1fr); } }
</style>

<script>
  import { onMount } from 'svelte';
  import { asset, resolve } from '$app/paths';
  import Rules from './Rules.svelte';
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
      if (first !== undefined && token === 0) show(first);
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
        <button type="button" class="chip" aria-pressed={current?.id === s.id} disabled={phase !== 'ready'} onclick={() => show(byId(s.id))}>{s.name}</button>
      {/each}
    </div>
    <div class="search"><Search names={phase === 'ready' ? names : null} id="ba-search" label="Compare any card" placeholder="Or type any card…" onpick={(i) => show(i)} /></div>
  </div>

  {#if phase === 'missing'}
    <p class="notice">This release has no before-and-after comparison.</p>
  {:else if error}
    <p class="notice" role="alert">The comparison did not load: {error}</p>
  {:else if current && !current.has_text}
    <p class="notice">{current.name} has no rules text, so neither model has anything to compare.</p>
  {:else}
    <div class="summary" aria-live="polite">
      {#if cols && current}
        <p><strong>{current.name}</strong>: {showcase.find((s) => s.id === current.id)?.why ?? 'a card you picked.'}</p>
        <p class="muted">{cols.overlap} of the top {TOP} are in both lists. <a href="{resolve('/')}?card={current.id}">Open {current.name} in the explorer</a></p>
      {/if}
    </div>
    <div class="cols">
      {#each [['Before', 'pretrained e5-base-v2', cols?.pre], ['After', 'fine-tuned on card tags', cols?.tuned]] as [title, sub, rows] (title)}
        <section class:after={title === 'After'} aria-busy={!rows}>
          <h4>{title} <span>{sub}</span></h4>
          <ol>
            {#each rows ?? Array.from({ length: TOP }, (_, i) => ({ i, ghost: true })) as r (r.i)}
              <li class:both={r.both} class:ghost={r.ghost}>
                {#if !r.ghost}
                  <span class="name"><span class="nm">{r.c.name}</span>{#if r.both}<span class="tag">in both</span>{/if}</span>
                  <span class="line"><Rules text={firstLine(r.c)} /></span>
                {/if}
              </li>
            {/each}
          </ol>
        </section>
      {/each}
    </div>
  {/if}
</div>

<style>
  .pick { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; justify-content: space-between; margin-bottom: 14px; }
  .chips { display: flex; flex-wrap: wrap; gap: 6px; }
  .chip { border: 1px solid var(--line); background: var(--surface); color: var(--ink); padding: 6px 12px; font-size: 0.84rem; }
  .chip:hover:not(:disabled) { border-color: var(--ink); }
  .chip[aria-pressed='true'] { border-color: var(--ink); background: var(--ink); color: var(--bg); }
  .chip:disabled { opacity: 0.6; cursor: default; }
  .search { width: min(100%, 300px); }
  .search :global(input) { height: 40px; box-shadow: none; }
  /* fixed heights so the lists do not jump when the data arrives or another card is picked */
  .summary { min-height: 5.2em; font-size: 0.95rem; max-width: 76ch; }
  .summary p { margin: 0 0 4px; }
  .summary .muted { font-size: 0.86rem; }
  .cols { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; margin-top: 10px; }
  section { padding: 14px 16px 6px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); min-width: 0; }
  section.after { border-color: var(--ink); }
  h4 { margin: 0 0 8px; font-size: 1.05rem; font-weight: 650; }
  h4 span { display: block; font: 400 0.78rem var(--sans); color: var(--muted); margin-top: 2px; }
  ol { margin: 0; padding-left: 1.5em; }
  li { height: 4.6rem; padding: 7px 0; border-top: 1px solid var(--line); font-size: 0.86rem; line-height: 1.4; overflow: hidden; }
  li::marker { color: var(--muted); font: 0.75rem var(--sans); }
  li.ghost::marker { color: transparent; }
  li.ghost { background: linear-gradient(var(--sunken), var(--sunken)) 0 10px / 55% 0.9em no-repeat, linear-gradient(var(--sunken), var(--sunken)) 0 34px / 85% 0.8em no-repeat; }
  .name { display: flex; align-items: baseline; gap: 8px; font-weight: 600; }
  .nm { min-width: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .tag { flex: none; padding: 0 6px; border-radius: 3px; background: var(--mark); font-size: 0.7rem; font-weight: 500; }
  .line { font-family: var(--serif); display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; line-clamp: 2; overflow: hidden; color: var(--muted); }
  @media (max-width: 640px) { .cols { grid-template-columns: minmax(0, 1fr); } }
</style>

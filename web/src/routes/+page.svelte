<script>
  import { onMount } from 'svelte';
  import { base, resolve } from '$app/paths';
  import { pushState, replaceState } from '$app/navigation';
  import AtlasMap from '$lib/AtlasMap.svelte';
  import CardText from '$lib/CardText.svelte';
  import CardTile from '$lib/CardTile.svelte';
  import Compare from '$lib/Compare.svelte';
  import Rules from '$lib/Rules.svelte';
  import Search from '$lib/Search.svelte';
  import { parseUrl, resolveAbilityIndex, serializeUrl } from '$lib/bundle.js';
  import { getAbilities, getCard, getNeighbours, getTagLabels, loadRelease, loadSiteData, setRoot } from '$lib/data.js';
  import { activeFilters, resolveSelection, visibleNeighbours } from '$lib/explorer.js';
  import { decodeMap } from '$lib/map.js';
  import metrics from '$lib/generated/metrics.json';

  const PAGE = 12;
  const NO_FILTERS = { color: '', minMv: null, maxMv: null, creature: '', type: '' };
  let release = $state(null);
  let names = $state(null);
  let ids = null;
  let map = $state(null);
  let regions = $state([]);
  let siteMissing = $state(false);
  let error = $state('');
  let notice = $state('');
  let index = $state(-1);
  let card = $state(null);
  let abilities = $state([]);
  let tagLabels = $state([]);
  let whole = $state([]);
  let ability = $state(null);
  let filters = $state({ ...NO_FILTERS });
  let shown = $state(PAGE);
  let byIndex = $state({});
  let compare = $state(null);
  let tab = $state('results');
  let busy = $state(false);
  let token = 0;

  const saved = $derived(ability === null ? whole : abilities[ability]?.neighbours ?? []);
  const filtering = $derived(activeFilters(filters));
  const visible = $derived(visibleNeighbours(saved, (i) => byIndex[i], filters));
  const pageRows = $derived(visible.slice(0, shown));
  const pending = $derived((filtering ? saved : saved.slice(0, shown)).some(([i]) => !byIndex[i]));
  const showcase = metrics.showcase ?? [];

  // load the cards the list needs: the shown page, or every saved neighbour while filtering
  $effect(() => {
    const need = (filtering ? saved : saved.slice(0, shown)).map(([i]) => i).filter((i) => !byIndex[i]);
    if (!need.length) return;
    const t = token;
    Promise.all(need.map((i) => getCard(i).then((c) => [i, c]))).then((rows) => {
      if (t !== token) return;
      byIndex = { ...byIndex, ...Object.fromEntries(rows) };
    }).catch((e) => { error = e.message; });
  });

  function updateUrl(push) {
    const qs = serializeUrl({ card: card?.id ?? '', ability, ...filters });
    (push ? pushState : replaceState)(qs ? `?${qs}` : location.pathname, {});
  }

  async function select(i, { push = true, abilityIndex = null } = {}) {
    const t = ++token;
    busy = true;
    notice = '';
    compare = null;
    shown = PAGE;
    try {
      const c = await getCard(i);
      const [n, ab, labels] = c.has_text ? await Promise.all([getNeighbours(i), getAbilities(i), getTagLabels()]) : [[], [], await getTagLabels()];
      if (t !== token) return;
      index = i; card = c; whole = n; abilities = ab; tagLabels = labels;
      ability = resolveAbilityIndex(abilityIndex, ab);
      updateUrl(push);
      if (push) document.getElementById('results')?.scrollIntoView({ block: 'start', behavior: 'smooth' });
    } catch (e) {
      if (t === token) error = e.message;
    } finally {
      if (t === token) busy = false;
    }
  }

  function clear() {
    token += 1;
    index = -1; card = null; whole = []; abilities = []; ability = null; compare = null;
  }

  function apply(state) {
    filters = { color: state.color, minMv: state.minMv, maxMv: state.maxMv, creature: state.creature, type: state.type };
    const r = resolveSelection(ids, state);
    if (r.unknown) { clear(); notice = 'That link points to a card that is not in this release.'; return; }
    if (r.index < 0) { clear(); return; }
    select(r.index, { push: false, abilityIndex: state.ability });
  }

  function setAbility(k) {
    ability = k;
    shown = PAGE;
    updateUrl(true);
  }

  function setFilter(key, value) {
    filters = { ...filters, [key]: value };
    updateUrl(false);
  }

  const numberOrNull = (v) => (v === '' || v === null || !Number.isFinite(Number(v)) ? null : Number(v));
  const byId = (id) => (ids ? resolveSelection(ids, { card: id }).index : -1);
  const pick = (c) => { const i = byId(c.id); if (i >= 0) select(i); };

  onMount(() => {
    setRoot(base);
    const onPop = () => ids && apply(parseUrl(location.search));
    addEventListener('popstate', onPop);
    (async () => {
      try {
        const r = await loadRelease(fetch);
        release = r.release; names = r.names;
        loadSiteData(release.release_id).then((s) => {
          if (s) { map = decodeMap(s.map); regions = s.regions; } else siteMissing = true;
        }).catch(() => { siteMissing = true; });
        ids = await r.idsReady;
        apply(parseUrl(location.search));
      } catch (e) { error = e.message; }
    })();
    return () => removeEventListener('popstate', onPop);
  });
</script>

<svelte:head>
  <title>{card ? `${card.name} · Card Atlas` : 'Card Atlas · find Magic cards by what they do'}</title>
  <meta name="description" content="Pick a Magic: The Gathering card and see the cards whose rules do the most similar thing, ranked by a fine-tuned language model, on a map of every card." />
</svelte:head>

<div class="explorer" class:has-card={card}>
  <div class="tabs" role="tablist" aria-label="View">
    <button role="tab" aria-selected={tab === 'results'} onclick={() => (tab = 'results')}>Results</button>
    <button role="tab" aria-selected={tab === 'map'} onclick={() => (tab = 'map')}>Map</button>
  </div>

  <section class="map-pane" class:hidden={tab !== 'map'} aria-label="Card map">
    {#if map && names}
      <AtlasMap {map} {regions} {names} selected={card?.has_text ? index : -1} neighbours={pageRows.map(([i]) => i)} onselect={(i) => select(i)} />
      {#if card}
        <div class="map-bar"><span>{card.name}</span><button type="button" class="btn" onclick={() => (tab = 'results')}>See results</button></div>
      {/if}
    {:else if siteMissing}
      <p class="notice">The map is not available for this release.</p>
    {:else}
      <div class="map-loading" aria-hidden="true"></div>
    {/if}
  </section>

  <section class="results-pane" class:hidden={tab !== 'results'}>
    {#if error}<p class="notice" role="alert">Something went wrong loading the card data: {error}</p>{/if}
    {#if notice}<p class="notice" role="status">{notice}</p>{/if}

    {#if !card}
      <div class="intro">
        <p class="eyebrow">Magic: The Gathering · {release ? release.n_cards.toLocaleString('en') : '…'} cards</p>
        <h1>Find cards by <em>what they do</em>.</h1>
        <p class="lede">Pick a card and Card Atlas lists the cards whose rules do the most similar thing, ranked by a language model fine-tuned on card text. The map places every card by the same similarity, so related effects sit together.</p>
        <Search {names} onpick={(i) => select(i)} />
        {#if showcase.length}
          <div class="examples">
            <span class="muted">Try</span>
            {#each showcase as ex (ex.id)}
              <a class="chip" href="?card={ex.id}" onclick={(e) => { if (byId(ex.id) >= 0) { e.preventDefault(); select(byId(ex.id)); } }}>{ex.name}</a>
            {/each}
          </div>
        {/if}
        <p class="how"><a href={resolve('/how-it-works')}>How the model was built and how well it works →</a></p>
      </div>
    {:else}
      <div class="top-search"><Search {names} id="card-search-top" onpick={(i) => select(i)} /></div>
      <article class="selected" aria-labelledby="selected-name">
        {#if card.image}<img src={card.image} alt="" width="488" height="680" />{/if}
        <div class="selected-body">
          <p class="eyebrow">Selected card</p>
          <h1 id="selected-name">{card.name} {#if card.faces.length === 1 && card.faces[0].mana_cost}<span class="cost"><Rules text={card.faces[0].mana_cost} /></span>{/if}</h1>
          <CardText {card} />
          {#if card.tags?.length}
            <div class="tags" title={release.tags_status}>
              <span class="muted">Predicted tags</span>
              {#each card.tags.slice(0, 10) as t (t)}<span class="chip">{tagLabels[t]}</span>{/each}
            </div>
          {/if}
          <p class="links"><a href={card.scryfall_uri} rel="noopener">View on Scryfall</a></p>
        </div>
      </article>

      {#if !card.has_text}
        <p class="notice">This card has no rules text, so the model has nothing to compare it on.</p>
      {:else}
        <div class="similar-by" id="results">
          <h2>Similar cards</h2>
          {#if abilities.length > 1}
            <div class="by" role="group" aria-label="Match on">
              <button type="button" class:on={ability === null} aria-pressed={ability === null} onclick={() => setAbility(null)}>Whole card</button>
              {#each abilities as a, k (k)}
                <button type="button" class:on={ability === k} aria-pressed={ability === k} onclick={() => setAbility(k)}><Rules text={a.text} /></button>
              {/each}
            </div>
            <p class="by-note muted">{ability === null ? 'Ranked on the whole card. Pick a line to match cards that do that one thing.' : 'Matching one line: each card scores by its closest line. Exploratory: line matches were not evaluated.'}</p>
          {/if}
        </div>

        <details class="filters" open={filtering}>
          <summary>Filter results{filtering ? ` · ${visible.length} of ${saved.length} match` : ''}</summary>
          <div class="filter-row">
            <label>Colour
              <select value={filters.color} onchange={(e) => setFilter('color', e.currentTarget.value)}>
                <option value="">Any</option><option value="W">White</option><option value="U">Blue</option><option value="B">Black</option><option value="R">Red</option><option value="G">Green</option><option value="C">Colourless</option>
              </select>
            </label>
            <label>Mana value
              <span class="mv"><input type="number" min="0" max="20" inputmode="numeric" placeholder="min" value={filters.minMv ?? ''} onchange={(e) => setFilter('minMv', numberOrNull(e.currentTarget.value))} aria-label="Minimum mana value" />–<input type="number" min="0" max="20" inputmode="numeric" placeholder="max" value={filters.maxMv ?? ''} onchange={(e) => setFilter('maxMv', numberOrNull(e.currentTarget.value))} aria-label="Maximum mana value" /></span>
            </label>
            <label>Type
              <input type="text" placeholder="e.g. instant" value={filters.type} onchange={(e) => setFilter('type', e.currentTarget.value)} />
            </label>
            <label>Creatures
              <select value={filters.creature} onchange={(e) => setFilter('creature', e.currentTarget.value)}>
                <option value="">Any</option><option value="creature">Only creatures</option><option value="noncreature">No creatures</option>
              </select>
            </label>
            {#if filtering}<button type="button" class="btn" onclick={() => { filters = { ...NO_FILTERS }; updateUrl(false); }}>Clear</button>{/if}
          </div>
        </details>

        <ol class="grid" aria-busy={busy || pending}>
          {#each pageRows as [i, score], r (i)}
            <li><CardTile card={byIndex[i]} {score} rank={r + 1} oncompare={(c) => (compare = { card: c, score })} onselect={() => select(i)} /></li>
          {/each}
          {#if pending}{#each { length: Math.max(0, Math.min(shown, saved.length) - pageRows.length) } as _, k (k)}<li class="ghost" aria-hidden="true"></li>{/each}{/if}
        </ol>
        {#if !pending && filtering && !visible.length}<p class="notice">No similar card passes these filters.</p>{/if}
        {#if visible.length > shown}<button type="button" class="btn more" onclick={() => (shown += PAGE)}>Show more</button>{/if}
        <p class="model-note muted">Scores are cosine similarities from {release.model.label}. Rankings are the model's judgement; check the cards before relying on them.</p>
      {/if}
    {/if}
  </section>
</div>

<Compare a={card} b={compare?.card ?? null} score={compare?.score ?? null} onclose={() => (compare = null)} onselect={pick} />

<style>
  .explorer { display: grid; grid-template-columns: minmax(0, 46fr) minmax(0, 54fr); gap: 24px; padding: 18px clamp(14px, 3vw, 28px) 32px; max-width: 1600px; margin: 0 auto; }
  .tabs { display: none; }
  .map-pane { position: sticky; top: 74px; height: calc(100dvh - 92px); min-height: 420px; }
  .map-loading { height: 100%; border-radius: var(--radius); background: var(--sunken); }
  .map-bar { display: none; }
  .results-pane { min-width: 0; }
  .intro { padding: clamp(8px, 6vh, 72px) 0 24px; max-width: 620px; }
  .intro h1 { margin: 10px 0 14px; }
  .intro h1 em { font-style: italic; color: var(--accent); }
  .lede { font-size: 1.08rem; color: var(--muted); margin-bottom: 22px; }
  .examples { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin-top: 14px; font-size: 0.9rem; }
  .examples .chip { text-decoration: none; color: var(--ink); background: var(--surface); border: 1px solid var(--line); }
  .examples .chip:hover { border-color: var(--accent); color: var(--accent); }
  .how { margin-top: 26px; font-weight: 500; }
  .top-search { margin-bottom: 18px; }
  .selected { display: grid; grid-template-columns: 200px minmax(0, 1fr); gap: 20px; padding: 18px; background: var(--surface); border: 1px solid var(--line); border-radius: 14px; }
  .selected img { width: 200px; height: auto; aspect-ratio: 488 / 680; border-radius: 10px; background: var(--sunken); }
  .selected-body { min-width: 0; }
  .selected h1 { font-size: clamp(1.5rem, 2.6vw, 2rem); margin: 4px 0 10px; }
  .cost { font-size: 1rem; vertical-align: middle; white-space: nowrap; }
  .tags { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; margin-top: 12px; font-size: 0.82rem; }
  .links { margin: 12px 0 0; font-size: 0.88rem; }
  .similar-by { margin: 28px 0 10px; scroll-margin-top: 70px; }
  .similar-by h2 { margin-bottom: 10px; }
  .by { display: flex; flex-wrap: wrap; gap: 6px; }
  .by button { max-width: 100%; padding: 6px 12px; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); font-size: 0.84rem; text-align: left; overflow-wrap: anywhere; }
  .by button.on { border-color: var(--accent); background: var(--accent-soft); }
  .by-note { font-size: 0.84rem; margin: 8px 0 0; }
  .filters { margin: 12px 0 14px; font-size: 0.9rem; }
  .filters summary { cursor: pointer; color: var(--muted); }
  .filter-row { display: flex; flex-wrap: wrap; gap: 12px; align-items: end; margin-top: 10px; }
  .filter-row label { display: grid; gap: 4px; font-size: 0.8rem; color: var(--muted); }
  .filter-row select, .filter-row input { height: 34px; padding: 0 8px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }
  .filter-row input[type='text'] { width: 130px; }
  .mv { display: flex; align-items: center; gap: 4px; }
  .mv input { width: 64px; }
  .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(min(100%, 330px), 1fr)); gap: 12px; margin: 0; padding: 0; list-style: none; }
  .grid > li { display: grid; min-width: 0; }
  .ghost { min-height: 150px; border-radius: var(--radius); background: var(--sunken); }
  .more { margin: 16px auto 0; display: flex; }
  .model-note { font-size: 0.8rem; margin-top: 20px; }
  @media (max-width: 900px) {
    .explorer { grid-template-columns: minmax(0, 1fr); gap: 14px; padding-top: 12px; }
    .tabs { display: flex; gap: 4px; padding: 4px; background: var(--sunken); border-radius: 10px; }
    .tabs button { flex: 1; padding: 7px; border: 0; border-radius: 8px; background: none; font-weight: 500; color: var(--muted); }
    .tabs button[aria-selected='true'] { background: var(--surface); color: var(--ink); box-shadow: var(--shadow); }
    .explorer:not(.has-card) .tabs { display: none; }
    .explorer:not(.has-card) .map-pane { display: block; position: static; height: 46dvh; min-height: 280px; order: 2; }
    .map-pane { position: relative; top: 0; height: 72dvh; }
    .hidden { display: none; }
    .explorer:not(.has-card) .results-pane { display: block; }
    .map-bar { display: flex; position: absolute; left: 10px; right: 52px; top: 10px; align-items: center; justify-content: space-between; gap: 8px; padding: 6px 6px 6px 12px; border-radius: 10px; background: var(--surface); box-shadow: var(--shadow); font-weight: 600; font-size: 0.9rem; }
    .map-bar span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .selected { grid-template-columns: 110px minmax(0, 1fr); gap: 14px; padding: 14px; }
    .selected img { width: 110px; }
    .intro { padding-top: 8px; }
  }
  @media (max-width: 480px) {
    .selected { grid-template-columns: minmax(0, 1fr); }
    .selected img { width: 150px; }
  }
</style>

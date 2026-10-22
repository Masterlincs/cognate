<script>
  import { onMount } from 'svelte';
  import { asset, resolve } from '$app/paths';
  import { afterNavigate, pushState, replaceState } from '$app/navigation';
  import AtlasMap from '#lib/AtlasMap.svelte';
  import CardText from '#lib/CardText.svelte';
  import CardGridItem from '#lib/CardGridItem.svelte';
  import CardTile from '#lib/CardTile.svelte';
  import ColourFilter from '#lib/ColourFilter.svelte';
  import Compare from '#lib/Compare.svelte';
  import Rules from '#lib/Rules.svelte';
  import Search from '#lib/Search.svelte';
  import Segmented from '#lib/Segmented.svelte';
  import { isV4Plus, isV5Plus, isV6, parseUrl, resolveAbilityIndex, serializeUrl } from '#lib/bundle.js';
  import { getAbilities, getCard, getFaces, getLines, getNeighbours, getTagLabels, loadRelease, loadSiteData, setRoot } from '#lib/data.js';
  import { activeFilters, hasMore, resolveSelection, savedList, visibleNeighbours } from '#lib/explorer.js';
  import { decodeMap, regionLabel } from '#lib/map.js';
  import metrics from '#lib/generated/metrics.json';

  const PAGE = 12;
  const NO_FILTERS = { color: '', colorMode: 'include', colorBy: 'identity', minMv: null, maxMv: null, creature: '', type: '' };
  const PARAM_NAMES = { color: 'colour', cm: 'colour match', cb: 'colour or identity', min: 'minimum mana value', max: 'maximum mana value', creature: 'creatures' };
  let release = $state(null);
  let names = $state(null);
  let ids = null;
  let idsReady = null;
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
  let rank = $state('whole');
  let lineRows = $state(null);
  let faceRows = $state(null);
  let filters = $state({ ...NO_FILTERS });
  let shown = $state(PAGE);
  let byIndex = $state({});
  let compare = $state(null);
  let tab = $state('results');
  let view = $state('list');
  let busy = $state(false);
  let token = 0;

  const hasLines = $derived(isV4Plus(release));
  const hasFaces = $derived(isV6(release));
  const hasColours = $derived(isV5Plus(release));
  const byLine = $derived(ability === null && rank === 'line');
  const byFace = $derived(ability === null && rank === 'face');
  const saved = $derived(savedList({ ability, rank, whole, lines: lineRows, faces: faceRows, abilities }));
  const filtering = $derived(activeFilters(filters));
  const visible = $derived(visibleNeighbours(saved, (i) => byIndex[i], filters));
  const pageRows = $derived(visible.slice(0, shown));
  const pending = $derived((byLine && lineRows === null) || (byFace && faceRows === null) || (filtering ? saved : saved.slice(0, shown)).some(([i]) => !byIndex[i]));
  const showcase = metrics.showcase ?? [];

  // load the cards the list needs: the shown page, or every saved neighbour while filtering.
  // Not while a selection loads: the list still belongs to the old card but the token is new.
  $effect(() => {
    if (busy) return;
    const need = (filtering ? saved : saved.slice(0, shown)).map(([i]) => i).filter((i) => !byIndex[i]);
    if (!need.length) return;
    const t = token;
    Promise.all(need.map((i) => getCard(i).then((c) => [i, c]))).then((rows) => {
      if (t !== token) return;
      byIndex = { ...byIndex, ...Object.fromEntries(rows) };
    }).catch((e) => { if (t === token) error = e.message; });
  });

  // fetch candidate-side face matches only while Face match is selected
  $effect(() => {
    if (busy || !byFace || !card?.has_text || faceRows !== null) return;
    const t = token;
    getFaces(index).then((rows) => { if (t === token) faceRows = rows; }).catch((e) => { if (t === token) error = e.message; });
  });

  // line-match rows for the shown card, once any selection has settled; this also covers
  // switching to Line match while another card is still loading
  $effect(() => {
    if (busy || !byLine || !card?.has_text || lineRows !== null) return;
    const t = token;
    getLines(index).then((rows) => { if (t === token) lineRows = rows; }).catch((e) => { if (t === token) error = e.message; });
  });

  function updateUrl(push) {
    const qs = serializeUrl({ card: card?.id ?? '', ability, rank, view, ...filters });
    (push ? pushState : replaceState)(qs ? `?${qs}` : location.pathname, {});
  }

  async function select(i, { push = true, abilityIndex = null } = {}) {
    const t = ++token;
    busy = true;
    notice = '';
    error = '';
    compare = null;
    shown = PAGE;
    try {
      // card shards are checked against ids.json, so wait for it before the first load
      if (!ids) await idsReady;
      const c = await getCard(i);
      const [n, ab, ln, fc, labels] = c.has_text ? await Promise.all([getNeighbours(i), getAbilities(i), rank === 'line' ? getLines(i) : null, rank === 'face' && hasFaces ? getFaces(i) : null, getTagLabels()]) : [[], [], [], [], await getTagLabels()];
      if (t !== token) return;
      index = i; card = c; whole = n; abilities = ab; lineRows = ln; faceRows = fc; tagLabels = labels;
      ability = resolveAbilityIndex(abilityIndex, ab);
      updateUrl(push);
      const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
      if (push) document.getElementById('results')?.scrollIntoView({ block: 'start', behavior: still ? 'auto' : 'smooth' });
    } catch (e) {
      if (t === token) error = e.message;
    } finally {
      if (t === token) busy = false;
    }
  }

  function clear() {
    token += 1;
    index = -1; card = null; whole = []; abilities = []; lineRows = null; faceRows = null; ability = null; compare = null;
  }

  function apply(state) {
    filters = { color: state.color, colorMode: state.colorMode, colorBy: hasColours ? state.colorBy : 'identity', minMv: state.minMv, maxMv: state.maxMv, creature: state.creature, type: state.type };
    rank = hasFaces ? state.rank : hasLines ? (state.rank === 'face' ? 'whole' : state.rank) : 'whole';
    view = state.view;
    const r = resolveSelection(ids, state);
    if (r.unknown) clear();
    else if (r.index < 0) clear();
    else select(r.index, { push: false, abilityIndex: state.ability });
    // select() clears the notice before its first await, so these land after it
    const dropped = state.dropped.length ? `This link has a value the explorer does not recognise for ${state.dropped.map((k) => PARAM_NAMES[k]).join(', ')}, so ${state.dropped.length > 1 ? 'those filters are' : 'that filter is'} off.` : '';
    const unavailableFace = state.rank === 'face' && !hasFaces ? 'Face match is unavailable in this release, so Whole card is selected.' : '';
    notice = [r.unknown ? 'That link points to a card that is not in this release.' : '', dropped, unavailableFace].filter(Boolean).join(' ');
  }

  function setAbility(k) {
    ability = k;
    rank = 'whole';
    shown = PAGE;
    updateUrl(true);
  }

  function setRank(value) {
    ability = null;
    rank = value;
    shown = PAGE;
    updateUrl(true);
  }

  // a filter change replaces any link-repair message, which would otherwise describe the old filters
  function setFilter(key, value) {
    filters = { ...filters, [key]: value };
    notice = '';
    updateUrl(false);
  }

  function setView(value) {
    view = value;
    updateUrl(false);
  }

  function setColour(color, colorMode, colorBy) {
    filters = { ...filters, color, colorMode, colorBy };
    notice = '';
    updateUrl(false);
  }

  const numberOrNull = (v) => (v === '' || v === null || !(Number(v) >= 0) || !Number.isFinite(Number(v)) ? null : Number(v));
  // a rejected value (e.g. -1) leaves the filter unchanged, so clear it from the box too
  const setMv = (key, el) => { const v = numberOrNull(el.value); if (v === null) el.value = ''; setFilter(key, v); };
  const byId = (id) => (ids ? resolveSelection(ids, { card: id }).index : -1);
  const pick = (c) => { const i = byId(c.id); if (i >= 0) select(i); };

  // a plain link to this page (header logo, Explore) keeps the component, so re-read the URL
  afterNavigate(({ type }) => {
    if (ids && (type === 'link' || type === 'goto')) apply(parseUrl(location.search));
  });

  onMount(() => {
    setRoot(asset(''));
    const onPop = () => ids && apply(parseUrl(location.search));
    addEventListener('popstate', onPop);
    (async () => {
      try {
        const r = await loadRelease(fetch);
        release = r.release; names = r.names; idsReady = r.idsReady;
        loadSiteData(release.release_id).then((s) => {
          if (s) { map = decodeMap(s.map); regions = s.regions.map((r) => ({ ...r, label: regionLabel(r.label) })); } else siteMissing = true;
        }).catch(() => { siteMissing = true; });
        const before = token;
        ids = await idsReady;
        // a card picked from search while ids loaded wins over the starting URL
        if (token === before) apply(parseUrl(location.search));
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
        <h1>Find Magic cards by what they do.</h1>
        <p class="lede">Pick a card to see the cards whose rules do the most similar thing, out of {release ? release.n_cards.toLocaleString('en') : 'all'} cards. A language model fine-tuned on rules text does the ranking. The map places every card by the same similarity, so related effects sit together.</p>
        <Search {names} onpick={(i) => select(i)} />
        {#if showcase.length}
          <div class="examples">
            <span class="muted">Try</span>
            {#each showcase as ex (ex.id)}
              <a class="chip" href="?card={ex.id}" onclick={(e) => { if (byId(ex.id) >= 0) { e.preventDefault(); select(byId(ex.id)); } }}>{ex.name}</a>
            {/each}
          </div>
        {/if}
        <p class="how"><a href={resolve('/how-it-works')}>How the model was built, and how well it works</a></p>
      </div>
    {:else}
      <div class="top-search"><Search {names} id="card-search-top" onpick={(i) => select(i)} /></div>
      <article class="selected" aria-labelledby="selected-name">
        {#if card.image}<img src={card.image} alt="" width="488" height="680" />{/if}
        <div class="selected-body">
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
          {#if hasLines || abilities.length > 1}
            {#if hasLines}<Segmented label="Rank by" options={[{ value: 'whole', label: 'Whole card' }, { value: 'line', label: 'Line match' }, ...(hasFaces ? [{ value: 'face', label: 'Face match' }] : [])]} value={ability === null ? rank : null} onchange={setRank} />{/if}
            {#if !hasLines}<button type="button" class="btn" class:on={ability === null} aria-pressed={ability === null} onclick={() => setRank('whole')}>Whole card</button>{/if}
            {#if abilities.length > 1}
              <p class="by-label muted">Or match one line:</p>
              <div class="by" role="group" aria-label="Match one line">
                {#each abilities as a, k (k)}
                  <button type="button" aria-pressed={ability === k} onclick={() => setAbility(k)}><Rules text={a.text} /></button>
                {/each}
              </div>
            {/if}
            <p class="by-note muted">{#if ability !== null}Matching one line: each card scores by its closest line. Exploratory: line matches were not evaluated.{:else if byFace}Face match: a card with two faces also counts as each face on its own, so a card whose back face does what this card does can rank as high as a one-faced card. Exploratory: on the development benchmark it scored about the same as the whole-card ranking.{:else if byLine}Line match: each line of this card is paired with the closest line on the other card, and lines that many cards share count for less. Exploratory: on the earlier model it ranked below the whole-card score on the tag benchmark.{:else}Ranked on the whole card.{#if abilities.length > 1}{' '}Pick a line to match cards that do that one thing.{/if}{/if}</p>
          {/if}
        </div>

        <div class="filters" role="group" aria-label="Filter results">
          <div class="f"><span class="fl">Colour</span><ColourFilter color={filters.color} mode={filters.colorMode} basis={filters.colorBy} showBasis={hasColours} onchange={setColour} /></div>
          <div class="f"><span class="fl">Type</span>
            <Segmented label="Creatures" options={[{ value: '', label: 'Any' }, { value: 'creature', label: 'Creatures' }, { value: 'noncreature', label: 'Non-creatures' }]} value={filters.creature} onchange={(v) => setFilter('creature', v)} />
            <input type="text" class="type" placeholder="Type line has…" aria-label="Type line contains" value={filters.type} onchange={(e) => setFilter('type', e.currentTarget.value)} />
          </div>
          <div class="f"><span class="fl">Mana value</span>
            <input type="number" min="0" max="20" inputmode="numeric" placeholder="min" value={filters.minMv ?? ''} onchange={(e) => setMv('minMv', e.currentTarget)} aria-label="Minimum mana value" /><span class="muted">to</span><input type="number" min="0" max="20" inputmode="numeric" placeholder="max" value={filters.maxMv ?? ''} onchange={(e) => setMv('maxMv', e.currentTarget)} aria-label="Maximum mana value" />
          </div>
          <div class="f"><span class="fl">View</span><Segmented label="View" options={[{ value: 'list', label: 'List' }, { value: 'grid', label: 'Grid' }]} value={view} onchange={setView} /></div>
          <p class="fstatus muted" aria-live="polite">{#if filtering}{visible.length} of the {saved.length} most similar cards match. <button type="button" class="linkish" onclick={() => { filters = { ...NO_FILTERS }; notice = ''; updateUrl(false); }}>Clear filters</button>{:else}Filters narrow the {saved.length} most similar cards; they do not search the whole catalogue.{/if}</p>
        </div>

        <ol class="results" class:grid={view === 'grid'} aria-busy={busy || pending}>
          {#each pageRows as [i, score], r (i)}
            {@const Item = view === 'grid' ? CardGridItem : CardTile}
            <li><Item card={byIndex[i]} {score} rank={r + 1} oncompare={(c) => (compare = { card: c, score })} onselect={() => select(i)} /></li>
          {/each}
          {#if pending}{#each { length: Math.max(0, Math.min(shown, saved.length) - pageRows.length) } as _, k (k)}<li class="ghost" aria-hidden="true"></li>{/each}{/if}
        </ol>
        {#if !pending && filtering && !visible.length}<p class="notice">None of the {saved.length} most similar cards passes these filters.</p>{/if}
        {#if hasMore({ saved, visible, shown, filtering })}<button type="button" class="btn more" onclick={() => (shown += PAGE)}>Show more</button>{/if}
        <p class="model-note muted">Scores are {byLine ? 'weighted line-match scores' : 'cosine similarities'} from {release.model.label}. Rankings are the model's judgement; check the cards before relying on them.</p>
      {/if}
    {/if}
  </section>
</div>

<Compare a={card} b={compare?.card ?? null} score={compare?.score ?? null} onclose={() => (compare = null)} onselect={pick} />

<style>
  .explorer { display: grid; grid-template-columns: minmax(0, 46fr) minmax(0, 54fr); gap: 28px; padding: 18px clamp(14px, 3vw, 28px) 32px; max-width: 1600px; margin: 0 auto; }
  .tabs { display: none; }
  .map-pane { position: sticky; top: 74px; height: calc(100dvh - 92px); min-height: 420px; }
  .map-loading { height: 100%; border-radius: var(--radius); background: var(--sunken); }
  .map-bar { display: none; }
  .results-pane { min-width: 0; }
  .intro { padding: clamp(8px, 7vh, 88px) 0 24px; max-width: 600px; }
  .intro h1 { margin: 0 0 16px; }
  .lede { font-size: 1.06rem; color: var(--muted); margin-bottom: 24px; max-width: 54ch; }
  .examples { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 14px; margin-top: 14px; font-size: 0.92rem; }
  .examples .chip { padding: 0; background: none; border-radius: 0; color: var(--ink); font-size: inherit; text-decoration: underline; text-decoration-color: var(--line); text-underline-offset: 0.25em; }
  .examples .chip:hover { text-decoration-color: currentColor; }
  .how { margin-top: 32px; font-size: 0.95rem; }
  .top-search { margin-bottom: 22px; }
  .selected { display: grid; grid-template-columns: 180px minmax(0, 1fr); gap: 22px; padding-bottom: 24px; border-bottom: 1px solid var(--line); }
  .selected img { width: 180px; height: auto; aspect-ratio: 488 / 680; border-radius: 9px; background: var(--sunken); }
  .selected-body { min-width: 0; }
  .selected h1 { font-size: clamp(1.6rem, 2.6vw, 2.1rem); margin: 2px 0 12px; }
  .selected :global(.card-text) { font-size: 1.02rem; }
  .cost { font-size: 1.05rem; vertical-align: 0.2em; white-space: nowrap; letter-spacing: 0; }
  .tags { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; margin-top: 14px; font-size: 0.82rem; }
  .links { margin: 14px 0 0; font-size: 0.9rem; }
  .similar-by { margin: 26px 0 0; scroll-margin-top: 70px; }
  .similar-by h2 { margin-bottom: 12px; }
  .by { display: flex; flex-direction: column; gap: 2px; }
  .by button { max-width: 100%; padding: 6px 10px; border: 0; border-left: 2px solid var(--line); background: none; font-size: 0.88rem; text-align: left; overflow-wrap: anywhere; color: var(--muted); }
  .by button:hover { color: var(--ink); }
  .by button[aria-pressed='true'] { border-left-color: var(--ink); color: var(--ink); background: var(--surface); }
  .btn.on { border-color: var(--ink); }
  .by-note { font-size: 0.84rem; margin: 10px 0 0; max-width: 70ch; }
  .by-label { font-size: 0.84rem; margin: 14px 0 6px; }
  .filters { display: flex; flex-wrap: wrap; align-items: center; gap: 12px 26px; margin: 22px 0 6px; padding: 14px 0; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
  .f { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  .fl { font-size: 0.84rem; font-weight: 600; }
  .filters input { height: 32px; padding: 0 8px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); font-size: 0.88rem; }
  .filters input[type='number'] { width: 66px; }
  .filters input.type { width: 140px; }
  .fstatus { flex-basis: 100%; margin: 0; font-size: 0.82rem; }
  .linkish { padding: 0; border: 0; background: none; font-size: inherit; color: var(--ink); text-decoration: underline; text-underline-offset: 0.2em; }
  .results { margin: 0; padding: 0; list-style: none; }
  .results > li { display: grid; min-width: 0; border-bottom: 1px solid var(--line); }
  .results.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 22px 16px; padding-top: 14px; }
  .results.grid > li { border-bottom: 0; }
  .results.grid .ghost { min-height: 0; aspect-ratio: 488 / 680; margin: 0; }
  .ghost { min-height: 120px; margin: 8px 0; border-radius: var(--radius); background: var(--sunken); }
  .more { margin: 18px auto 0; display: flex; }
  .model-note { font-size: 0.8rem; margin-top: 22px; }
  @media (max-width: 900px) {
    .explorer { grid-template-columns: minmax(0, 1fr); gap: 14px; padding-top: 12px; }
    .tabs { display: flex; border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }
    .tabs button { flex: 1; padding: 7px; border: 0; background: var(--surface); font-weight: 500; color: var(--muted); }
    .tabs button + button { border-left: 1px solid var(--line); }
    .tabs button[aria-selected='true'] { background: var(--ink); color: var(--bg); }
    .explorer:not(.has-card) .tabs { display: none; }
    .explorer:not(.has-card) .map-pane { display: block; position: static; height: 46dvh; min-height: 280px; order: 2; }
    .map-pane { position: relative; top: 0; height: 72dvh; }
    .hidden { display: none; }
    .explorer:not(.has-card) .results-pane { display: block; }
    .map-bar { display: flex; position: absolute; left: 10px; right: 52px; top: 10px; align-items: center; justify-content: space-between; gap: 8px; padding: 6px 6px 6px 12px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); font-weight: 600; font-size: 0.9rem; }
    .map-bar span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .selected { grid-template-columns: 110px minmax(0, 1fr); gap: 14px; }
    .selected img { width: 110px; }
    .intro { padding-top: 8px; }
  }
  @media (max-width: 480px) {
    .selected { grid-template-columns: minmax(0, 1fr); }
    .selected img { width: 150px; }
  }
</style>

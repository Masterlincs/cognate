<script>
  import { onMount, tick, untrack } from 'svelte';
  import { asset, resolve } from '$app/paths';
  import { afterNavigate, beforeNavigate, pushState, replaceState } from '$app/navigation';
  import { page } from '$app/state';
  import CardMap from '#lib/CardMap.svelte';
  import CardText from '#lib/CardText.svelte';
  import CardGridItem from '#lib/CardGridItem.svelte';
  import CardTile from '#lib/CardTile.svelte';
  import ColourFilter from '#lib/ColourFilter.svelte';
  import Compare from '#lib/Compare.svelte';
  import Examples from '#lib/Examples.svelte';
  import Rules from '#lib/Rules.svelte';
  import Search from '#lib/Search.svelte';
  import Segmented from '#lib/Segmented.svelte';
  import { filterNeighbours, isV4Plus, isV5Plus, isV6, parseUrl, resolveAbilityIndex, serializeUrl } from '#lib/bundle.js';
  import { entryKey, newEntryId } from '#lib/history.js';
  import { getAbilities, getCard, getFaces, getLines, getNeighbours, getTagLabels, loadRelease, loadSiteData, setRoot } from '#lib/data.js';
  import { activeFilters, clearFilter, filterChips, hasMore, randomIndex, resolveSelection, savedList } from '#lib/explorer.js';
  import { loadFonts } from '#lib/load-fonts.js';
  import { remember } from '#lib/search.js';
  import { modelLabel } from '#lib/wording.js';
  import { decodeMap, regionLabel, TEXTLESS } from '#lib/map.js';
  import metrics from '#lib/generated/metrics.json';

  const PAGE = 12;
  // the ranking modes, with what each does and how far it was tested (the caveats are the evaluation's, keep them)
  const MODES = {
    whole: { label: 'Whole card', hint: 'Compare all the rules text on both cards', about: 'Compares all the rules text on each card at once.' },
    line: { label: 'Line match', hint: 'Match each line to the closest line on the other card', about: 'Matches each line to the closest line on the other card. One shared ability can count, even when the rest of the card is different.', note: 'Exploratory. On an earlier model’s tag benchmark, it ranked below Whole card.' },
    face: { label: 'Face match', hint: 'Also score each face of a two-faced card on its own', about: 'Also scores each face of a two-faced card on its own. A card can match on only its front or only its back.', note: 'Exploratory. On the development benchmark, it scored about the same as Whole card.' },
  };
  const ICON_LIST = 'M5.5 4h8M5.5 8h8M5.5 12h8M2.5 4h.01M2.5 8h.01M2.5 12h.01';
  const ICON_GRID = 'M2.5 2.5h4.5v4.5H2.5zM9 2.5h4.5v4.5H9zM2.5 9h4.5v4.5H2.5zM9 9h4.5v4.5H9z';
  const NO_FILTERS = { color: '', colorMode: 'include', colorBy: 'identity', minMv: null, maxMv: null, creature: '', type: '' };
  const PARAM_NAMES = { color: 'colour', cm: 'colour match', cb: 'colour or identity', min: 'minimum mana value', max: 'maximum mana value', creature: 'creatures' };
  let release = $state.raw(null);
  let names = $state.raw(null); // raw, not a deep proxy: proxied, the first search took about 0.5 s on the throttled mobile profile
  let ids = null;
  let idsLoaded = $state(false); // ids stays a plain array; this flag lets the recent list react to it
  let idsReady = null;
  let map = $state(null);
  let regions = $state([]);
  let hidden = $state.raw(new Set()); // keyword-only cards: off the map and out of the lists
  let siteMissing = $state(false);
  let error = $state('');
  let rowsError = $state({}); // mode -> message: a failed Line or Face match load, shown only in that mode; clearing it retries
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
  let filtersOpen = $state(false); // narrow screens fold the filter fields behind one button
  let token = 0;
  let recent = $state.raw([]); // [{ id, name }], newest first, kept in localStorage
  let exampleCards = $state.raw({}); // showcase id -> card, for the landing page's example tiles
  let hoverIndex = $state(-1); // the result row under the pointer, ringed on the map
  let copied = $state(false);
  let resultsList = $state(null);
  let restoreY = $state(null); // back/forward: the scroll position to return to once that list has loaded
  const positions = new Map(); // history entry key (entryKey) -> { y, shown } when the page left that entry
  let entry = null; // the current history entry's id in page.state, or null for one made before this page ran
  let here = ''; // the current entry's key, under which keep() saves its place
  let popped = false; // Back or Forward: show the entry landed on once page.state is its state
  const tabY = { results: 0 };
  const RECENT_KEY = 'recent-cards';

  const hasLines = $derived(isV4Plus(release));
  const hasFaces = $derived(isV6(release));
  const hasColours = $derived(isV5Plus(release));
  const byLine = $derived(ability === null && rank === 'line');
  const byFace = $derived(ability === null && rank === 'face');
  const saved = $derived(savedList({ ability, rank, whole, lines: lineRows, faces: faceRows, abilities }).filter(([i]) => !hidden.has(i)));
  const filtering = $derived(activeFilters(filters));
  const chips = $derived(filterChips(filters));
  const visible = $derived(filterNeighbours(saved, (i) => byIndex[i], filters));
  const pageRows = $derived(visible.slice(0, shown));
  const shownError = $derived(error || (ability === null && rowsError[rank]) || '');
  const pending = $derived((byLine && lineRows === null) || (byFace && faceRows === null) || (filtering ? saved : saved.slice(0, shown)).some(([i]) => !byIndex[i]));
  const showcase = (metrics.showcase ?? []).filter((ex) => !ex.unseen);
  const recentOptions = $derived(names && idsLoaded ? recent.map((r) => ({ index: byId(r.id), name: r.name })).filter((r) => r.index >= 0 && r.index !== index) : []);
  const examples = $derived(showcase.map((ex) => ({ id: ex.id, name: ex.name, card: exampleCards[ex.id] ?? null })));
  const rankNote = $derived(ability !== null ? { about: 'Each card gets the score of its line that is closest to the line you picked.', note: 'Exploratory. One-line matches have not been evaluated.' } : MODES[rank] ?? MODES.whole);

  // the result row under the pointer, ringed on the map (one listener on the list, not one per row)
  $effect(() => {
    const list = resultsList;
    if (!list) return;
    const over = (e) => { const li = e.target.closest?.('li[data-index]'); hoverIndex = li ? Number(li.dataset.index) : -1; };
    const out = () => { hoverIndex = -1; };
    list.addEventListener('pointerover', over);
    list.addEventListener('pointerleave', out);
    return () => { list.removeEventListener('pointerover', over); list.removeEventListener('pointerleave', out); hoverIndex = -1; };
  });

  // the landing page's example tiles: their cards once ids.json is in (a failed shard leaves the name only)
  $effect(() => {
    if (!idsLoaded || card) return;
    untrack(() => {
      for (const ex of showcase) {
        const i = byId(ex.id);
        if (i >= 0 && !exampleCards[ex.id]) getCard(i).then((c) => { exampleCards = { ...exampleCards, [ex.id]: c }; }).catch(() => {});
      }
    });
  });

  // Back/Forward: SvelteKit sets page.state for the entry landed on after the popstate event, so
  // read its id here rather than in the event
  $effect(() => {
    const { e } = page.state;
    untrack(() => {
      if (!popped) return;
      popped = false;
      land(e);
      if (ids) apply(parseUrl(location.search), positions.get(here) ?? null);
    });
  });

  // back/forward: once the restored list's cards are in, return to where the page was
  $effect(() => {
    if (restoreY === null || busy || pending) return;
    const y = restoreY;
    restoreY = null;
    tick().then(() => scrollTo(0, y));
  });

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
    if (busy || !byFace || !card?.has_text || faceRows !== null || rowsError.face) return;
    const t = token;
    getFaces(index).then((rows) => { if (t === token) { faceRows = rows; } }).catch((e) => { if (t === token) rowsError.face = e.message; });
  });

  // line-match rows for the shown card, once any selection has settled; this also covers
  // switching to Line match while another card is still loading
  $effect(() => {
    if (busy || !byLine || !card?.has_text || lineRows !== null || rowsError.line) return;
    const t = token;
    getLines(index).then((rows) => { if (t === token) { lineRows = rows; } }).catch((e) => { if (t === token) rowsError.line = e.message; });
  });

  function updateUrl(push) {
    const qs = serializeUrl({ card: card?.id ?? '', ability, rank, view, ...filters });
    // a new entry gets a new id; a replaced one keeps its id, or gets one if it was made before this page ran
    if (push || !entry) entry = newEntryId();
    here = entry;
    (push ? pushState : replaceState)(qs ? `?${qs}` : location.pathname, { e: entry });
  }

  // the scroll position and list length to come back to on Back
  const keep = () => {
    positions.set(here, { y: scrollY, shown });
    // entries dropped from browser history are never read again, so keep only the most recent ones
    if (positions.size > 200) positions.delete(positions.keys().next().value);
  };
  // e: the id in the page.state of the entry now shown
  const land = (e) => { entry = e ?? null; here = entryKey(entry, location.search); };

  function rememberCard(c) {
    recent = remember(recent, { id: c.id, name: c.name });
    try { localStorage.setItem(RECENT_KEY, JSON.stringify(recent)); } catch { /* private mode */ }
  }

  // push: a visitor's pick (adds history, then shows the new card from the top and focuses its name);
  // focus: false keeps focus where it is (a map pick); back: a saved { y, shown } to restore
  async function select(i, { push = true, abilityIndex = null, focus = true, back = null } = {}) {
    if (push) keep();
    const t = ++token;
    busy = true;
    notice = '';
    error = ''; rowsError = {};
    compare = null;
    shown = PAGE;
    hoverIndex = -1;
    try {
      // card shards are checked against ids.json, so wait for it before the first load
      if (!ids) await idsReady;
      const c = await getCard(i);
      // Line and Face match rows load in their effects, so their failures stay with their mode
      const [n, ab, labels] = c.has_text ? await Promise.all([getNeighbours(i), getAbilities(i), getTagLabels()]) : [[], [], await getTagLabels()];
      if (t !== token) return;
      const rows = c.has_text ? null : [];
      index = i; card = c; whole = n; abilities = ab; lineRows = rows; faceRows = rows; tagLabels = labels;
      ability = resolveAbilityIndex(abilityIndex, ab);
      updateUrl(push);
      rememberCard(c);
      tabY.results = 0;
      if (back) { shown = back.shown; restoreY = back.y; }
      if (push && tab === 'results') {
        await tick();
        if (focus) document.getElementById('selected-name')?.focus({ preventScroll: true });
        const still = matchMedia('(prefers-reduced-motion: reduce)').matches;
        if (scrollY > 0) scrollTo({ top: 0, behavior: still ? 'auto' : 'smooth' });
      }
    } catch (e) {
      if (t === token) error = e.message;
    } finally {
      if (t === token) busy = false;
    }
  }

  function clear() {
    token += 1;
    index = -1; card = null; whole = []; abilities = []; lineRows = null; faceRows = null; ability = null; compare = null; rowsError = {};
  }

  function apply(state, back = null) {
    filters = { color: state.color, colorMode: state.colorMode, colorBy: hasColours ? state.colorBy : 'identity', minMv: state.minMv, maxMv: state.maxMv, creature: state.creature, type: state.type };
    rank = hasFaces ? state.rank : hasLines ? (state.rank === 'face' ? 'whole' : state.rank) : 'whole';
    view = state.view;
    const r = resolveSelection(ids, state);
    if (r.unknown) clear();
    else if (r.index < 0) clear();
    else select(r.index, { push: false, abilityIndex: state.ability, back });
    // select() clears the notice before its first await, so these land after it
    const dropped = state.dropped.length ? `This link has an unknown value for ${state.dropped.map((k) => PARAM_NAMES[k]).join(', ')}, so ${state.dropped.length > 1 ? 'those filters are' : 'that filter is'} off.` : '';
    const unavailableFace = state.rank === 'face' && !hasFaces ? 'Face match is not available in this release.' : '';
    notice = [r.unknown ? 'That card is not in this release.' : '', dropped, unavailableFace].filter(Boolean).join(' ');
  }

  function setAbility(k) {
    keep();
    ability = k;
    rank = 'whole';
    shown = PAGE;
    updateUrl(true);
  }

  function setRank(value) {
    keep();
    ability = null;
    rank = value;
    rowsError[value] = ''; // its effect retries
    shown = PAGE;
    updateUrl(true);
  }

  // a filter change replaces any link-repair message, which would otherwise describe the old filters
  function setFilter(key, value) {
    filters = { ...filters, [key]: value };
    notice = '';
    updateUrl(false);
  }

  // mobile Results/Map tabs: the map opens at the top, the results where they were left
  async function setTab(next) {
    if (next === tab) return;
    if (tab === 'results') tabY.results = scrollY;
    tab = next;
    await tick();
    scrollTo(0, next === 'results' ? tabY.results : 0);
  }
  function tabKey(e) {
    if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
    e.preventDefault();
    setTab(tab === 'results' ? 'map' : 'results').then(() => document.getElementById(`tab-${tab}`)?.focus());
  }

  // "/" jumps to the search box, as on many search sites; "r" opens a random card
  function shortcut(e) {
    if (e.ctrlKey || e.metaKey || e.altKey || compare || e.target.closest?.('input, textarea, select, [contenteditable]')) return;
    if (e.key === '/') {
      const box = document.querySelector('.explorer [role=combobox]');
      if (box) { e.preventDefault(); box.focus(); }
    } else if ((e.key === 'r' || e.key === 'R') && !e.shiftKey && !e.repeat && ids) {
      e.preventDefault();
      random();
    }
  }

  // any card with rules text that the map and lists show, other than the one already open
  function random() {
    if (!ids) return;
    const i = randomIndex(ids.length, (k) => k !== index && !hidden.has(k) && (!map || map.x[k] !== TEXTLESS));
    if (i >= 0) select(i);
  }

  function setFilters(next) {
    filters = next;
    notice = '';
    updateUrl(false);
  }

  // the removed chip's button goes away, so keep focus in the bar: the next chip, else the view switch
  async function removeChip(key) {
    setFilters(clearFilter(filters, key));
    await tick();
    (document.querySelector('.fchip') ?? document.querySelector('.view [aria-pressed=true]'))?.focus();
  }

  async function copyLink() {
    try {
      await navigator.clipboard.writeText(location.href);
      copied = true;
      setTimeout(() => (copied = false), 1800);
    } catch { /* no clipboard permission: the address bar still has the link */ }
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
  // an example tile opens in place once ids.json is in; before that its link loads the page
  const pickExample = (id) => { const i = byId(id); if (i < 0) return false; select(i); return true; };

  // a plain link to this page (header logo, Explore) keeps the component, so re-read the URL
  beforeNavigate(({ type }) => { if (type === 'link') keep(); });
  afterNavigate(({ type }) => {
    if (type !== 'link' && type !== 'goto') return;
    land(page.state.e);
    if (ids) apply(parseUrl(location.search));
  });

  onMount(() => {
    setRoot(asset(''));
    land(page.state.e);
    // save the place of the entry being left; the $effect on page.state shows the one landed on.
    // A fragment link (Skip to content) also fires popstate, with no state yet; SvelteKit leaves
    // page.state alone then (its popstate listener in @sveltejs/kit/src/runtime/client/client.js)
    // While a pick is loading, select() has already saved the place it left and reset the list length,
    // so saving again would overwrite that snapshot with the reset one
    const onPop = (e) => { if (!e.state) return; if (!busy) keep(); popped = true; };
    try { recent = JSON.parse(localStorage.getItem(RECENT_KEY) ?? '[]').filter((r) => typeof r?.id === 'string' && typeof r?.name === 'string'); } catch { /* private mode or bad JSON */ }
    // a first visit on a desktop starts in the search box; on touch screens that would open the keyboard over the page
    if (!new URLSearchParams(location.search).get('card') && matchMedia('(hover: hover) and (pointer: fine)').matches) document.getElementById('card-search')?.focus();
    addEventListener('popstate', onPop);
    // web fonts wait for the search index (or its failure) so they do not slow it down, and cards
    // wait for the fonts, so a card never shows blank mana symbols while they load
    const starting = loadRelease(fetch);
    const fontsReady = starting.then(() => loadFonts(), () => loadFonts());
    (async () => {
      try {
        const r = await starting;
        release = r.release; names = r.names; idsReady = Promise.all([r.idsReady, fontsReady]).then(([v]) => v);
        loadSiteData(release.release_id).then((s) => {
          if (s) { hidden = s.hidden; map = decodeMap(s.map, hidden); regions = s.regions.map((r) => ({ ...r, label: regionLabel(r.label) })); } else siteMissing = true;
        }).catch(() => { siteMissing = true; });
        const before = token;
        ids = await idsReady;
        idsLoaded = true;
        // a card picked from search while ids loaded wins over the starting URL
        if (token === before) apply(parseUrl(location.search));
      } catch (e) { error = e.message; }
    })();
    return () => removeEventListener('popstate', onPop);
  });
</script>

<svelte:head>
  <title>{card ? `${card.name} · Cognate` : 'Cognate'}</title>
  <meta name="description" content="Pick a Magic card and find the cards that do a similar thing. A fine-tuned embedding model ranks them." />
</svelte:head>

{#snippet dice()}<svg class="ico" viewBox="0 0 16 16" aria-hidden="true"><rect x="2" y="2" width="12" height="12" rx="2.75" /><circle cx="5.4" cy="5.4" r="1" /><circle cx="8" cy="8" r="1" /><circle cx="10.6" cy="10.6" r="1" /></svg>{/snippet}

<svelte:window onkeydown={shortcut} />

<div class="explorer" class:has-card={card}>
  <div class="tabs" role="tablist" aria-label="View">
    <button role="tab" id="tab-results" aria-controls="results-pane" aria-selected={tab === 'results'} tabindex={tab === 'results' ? 0 : -1} onclick={() => setTab('results')} onkeydown={tabKey}>Results</button>
    <button role="tab" id="tab-map" aria-controls="map-pane" aria-selected={tab === 'map'} tabindex={tab === 'map' ? 0 : -1} onclick={() => setTab('map')} onkeydown={tabKey}>Map</button>
  </div>

  <section class="map-pane" id="map-pane" class:hidden={tab !== 'map'} aria-label="Card map">
    {#if map && names}
      <CardMap {map} {regions} {names} selected={card?.has_text ? index : -1} neighbours={pageRows.map(([i]) => i)} highlight={hoverIndex} onselect={(i) => select(i, { focus: false })} />
      {#if card}
        <div class="map-bar"><span>{card.name}</span><button type="button" class="btn" onclick={() => setTab('results')}>See results</button></div>
      {/if}
    {:else if siteMissing}
      <p class="notice">This release has no map.</p>
    {:else}
      <div class="map-loading" aria-hidden="true"></div>
    {/if}
  </section>

  <section class="results-pane" id="results-pane" class:hidden={tab !== 'results'} class:busy aria-label="Results">
    {#if shownError}<p class="notice" role="alert">The card data did not load: {shownError}</p>{/if}
    {#if notice}<p class="notice" role="status">{notice}</p>{/if}

    {#if !card}
      <div class="intro">
        <h1>Find Magic cards by what they do.</h1>
        <p class="lede">Pick a card. Cognate ranks other cards by how similar their rules text is in effect, using a model trained on community tags.</p>
        <div class="search-row">
          <Search {names} recent={recentOptions} onpick={(i) => select(i)} />
          <button type="button" class="btn random" onclick={random} disabled={!idsLoaded} aria-keyshortcuts="r" title="Open a random card (R)">{@render dice()}<span class="lbl">Random card</span></button>
        </div>
        <p class="keys muted"><kbd>/</kbd> search <kbd>R</kbd> random card <kbd>↑</kbd><kbd>↓</kbd> recent cards</p>

        {#if examples.length}
          <section class="block" aria-labelledby="examples-title">
            <h2 id="examples-title" class="kicker">Try a classic</h2>
            <Examples items={examples} onpick={pickExample} />
          </section>
        {/if}
      </div>
    {:else}
      <div class="top-search">
        <Search {names} recent={recentOptions} id="card-search-top" onpick={(i) => select(i)} />
        <button type="button" class="btn random square" onclick={random} aria-label="Random card" aria-keyshortcuts="r" title="Random card (R)">{@render dice()}</button>
      </div>
      <article class="selected" aria-labelledby="selected-name">
        <!-- keyed, so the last card's image never shows under the new card's name -->
        {#key card.id}{#if card.image}<img src={card.image} alt="" width="488" height="680" />{/if}{/key}
        <div class="selected-body">
          <h1 id="selected-name" tabindex="-1">{card.name} {#if card.faces.length === 1 && card.faces[0].mana_cost}<span class="cost"><Rules text={card.faces[0].mana_cost} /></span>{/if}</h1>
          <CardText {card} />
          {#if card.tags?.length}
            <div class="tags" title="The tags that Scryfall Tagger lists for this card">
              <span class="muted">Tagger tags</span>
              {#each card.tags.slice(0, 10) as t (t)}<span class="chip">{tagLabels[t]}</span>{/each}
            </div>
          {/if}
          <p class="links">
            <a href={card.scryfall_uri} rel="noopener">View on Scryfall</a>
            <button type="button" class="linkish" onclick={copyLink}>{copied ? 'Link copied' : 'Copy link'}</button>
            <span class="sr-only" role="status">{copied ? 'Link copied' : ''}</span>
          </p>
        </div>
      </article>

      {#if !card.has_text}
        <p class="notice">This card has no rules text, so there is nothing to compare.</p>
      {:else}
        <section class="similar" aria-labelledby="similar-title">
          <h2 id="similar-title">Similar cards</h2>
          {#if hasLines || abilities.length > 1}
            <div class="rank">
              <div class="rank-row">
                <span class="flabel" aria-hidden="true">Rank by</span>
                {#if hasLines}
                  <Segmented label="Rank by" describedby="rank-help" options={Object.entries(MODES).filter(([k]) => k !== 'face' || hasFaces).map(([value, m]) => ({ value, label: m.label, hint: m.hint }))} value={ability === null ? rank : null} onchange={setRank} />
                {:else}
                  <Segmented label="Rank by" describedby="rank-help" options={[{ value: 'whole', label: MODES.whole.label, hint: MODES.whole.hint }]} value={ability === null ? 'whole' : null} onchange={setRank} />
                {/if}
              </div>
              <p class="rank-help" id="rank-help">{rankNote.about}{#if rankNote.note} <span class="muted">{rankNote.note}</span>{/if}</p>
              {#if abilities.length > 1}
                <p class="flabel by-label" id="by-label">Or match one line</p>
                <div class="by" role="group" aria-labelledby="by-label">
                  {#each abilities as a, k (k)}
                    <button type="button" aria-pressed={ability === k} onclick={() => setAbility(k)}><Rules text={a.text} /></button>
                  {/each}
                </div>
              {/if}
            </div>
          {/if}

          <div class="controls">
            <div class="fields" id="filter-fields" class:open={filtersOpen} role="group" aria-label="Filter results">
              <div class="field"><span class="flabel" aria-hidden="true">Colour</span><ColourFilter color={filters.color} mode={filters.colorMode} basis={filters.colorBy} showBasis={hasColours} onchange={setColour} /></div>
              <div class="field"><span class="flabel" aria-hidden="true">Card type</span>
                <div class="ctl">
                  <Segmented label="Creatures" options={[{ value: '', label: 'Any' }, { value: 'creature', label: 'Creatures' }, { value: 'noncreature', label: 'Non-creatures' }]} value={filters.creature} onchange={(v) => setFilter('creature', v)} />
                  <input type="search" class="type" placeholder="Type line has…" aria-label="Type line contains" value={filters.type} oninput={(e) => setFilter('type', e.currentTarget.value)} />
                </div>
              </div>
              <div class="field"><span class="flabel" aria-hidden="true">Mana value</span>
                <div class="mv">
                  <input type="number" min="0" max="20" inputmode="numeric" placeholder="min" value={filters.minMv ?? ''} oninput={(e) => setFilter('minMv', numberOrNull(e.currentTarget.value))} onchange={(e) => setMv('minMv', e.currentTarget)} aria-label="Minimum mana value" />
                  <span aria-hidden="true">–</span>
                  <input type="number" min="0" max="20" inputmode="numeric" placeholder="max" value={filters.maxMv ?? ''} oninput={(e) => setFilter('maxMv', numberOrNull(e.currentTarget.value))} onchange={(e) => setMv('maxMv', e.currentTarget)} aria-label="Maximum mana value" />
                </div>
              </div>
            </div>

            <div class="list-bar">
              <button type="button" class="btn ftoggle" aria-expanded={filtersOpen} aria-controls="filter-fields" onclick={() => (filtersOpen = !filtersOpen)}>
                <svg class="ico" viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 3.5h11l-4.25 5v4l-2.5 1.25V8.5z" /></svg>Filters{#if chips.length}<span class="badge num">{chips.length}</span>{/if}
              </button>
              <p class="count" aria-live="polite">
                {#if (byLine && lineRows === null) || (byFace && faceRows === null)}Loading similar cards…{:else if filtering}<strong class="num">{visible.length}</strong> of the top {saved.length} match{:else}Top <strong class="num">{saved.length}</strong> by similarity{/if}
              </p>
              {#if chips.length}
                <ul class="active" aria-label="Active filters">
                  {#each chips as c (c.key)}
                    <li><button type="button" class="fchip" aria-label="Remove filter: {c.label}" onclick={() => removeChip(c.key)}>{c.label}<span aria-hidden="true">×</span></button></li>
                  {/each}
                  {#if chips.length > 1}<li><button type="button" class="linkish" onclick={() => setFilters({ ...NO_FILTERS })}>Clear all</button></li>{/if}
                </ul>
              {/if}
              <div class="view"><Segmented label="View" options={[{ value: 'list', label: 'List', icon: ICON_LIST }, { value: 'grid', label: 'Grid', icon: ICON_GRID }]} value={view} onchange={setView} /></div>
            </div>
          </div>

          <ol class="results" class:grid={view === 'grid'} aria-busy={busy || pending} bind:this={resultsList}>
            {#each pageRows as [i, score], r (i)}
              {@const Item = view === 'grid' ? CardGridItem : CardTile}
              <li data-index={i}><Item card={byIndex[i]} {score} rank={r + 1} scoreLabel={byLine ? 'Line-match score' : 'Cosine similarity to the selected card'} oncompare={(c) => (compare = { card: c, score })} onselect={() => select(i)} /></li>
            {/each}
            {#if pending}{#each { length: Math.max(0, Math.min(shown, saved.length) - pageRows.length) } as _, k (k)}<li class="ghost" aria-hidden="true"></li>{/each}{/if}
          </ol>
          {#if !pending && filtering && !visible.length}
            <div class="empty">
              <p>None of the top {saved.length} cards match these filters.</p>
              <button type="button" class="btn" onclick={() => setFilters({ ...NO_FILTERS })}>Clear filters</button>
            </div>
          {/if}
          {#if hasMore({ saved, visible, shown, filtering })}<button type="button" class="btn more" onclick={() => (shown += PAGE)}>Show more</button>{/if}
          <p class="model-note muted">Scores: {byLine ? 'line-match score' : 'cosine similarity'}. Model: {modelLabel(release.model, metrics.deployed)}.</p>
        </section>
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
  .ico { width: 16px; height: 16px; flex: none; fill: none; stroke: currentColor; stroke-width: 1.5; stroke-linejoin: round; }
  .ico circle { fill: currentColor; stroke: none; }
  .flabel { margin: 0; font-size: 0.72rem; font-weight: 650; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }
  .linkish { padding: 0; border: 0; background: none; font-size: inherit; color: var(--ink); text-decoration: underline; text-decoration-color: color-mix(in srgb, currentColor 35%, transparent); text-underline-offset: 0.2em; }
  .linkish:hover { text-decoration-color: currentColor; }

  /* landing */
  .intro { padding: clamp(8px, 5vh, 56px) 0 8px; max-width: 700px; }
  .intro h1 { margin: 0 0 12px; }
  .lede { max-width: 54ch; margin: 0 0 22px; font-size: 1.06rem; color: var(--muted); }
  .search-row, .top-search { display: flex; gap: 8px; }
  .search-row > :global(.search), .top-search > :global(.search) { flex: 1; min-width: 0; }
  .random { flex: none; height: 48px; padding: 0 16px; font-size: 0.92rem; }
  .random.square { width: 48px; padding: 0; justify-content: center; }
  .keys { display: flex; flex-wrap: wrap; align-items: center; gap: 5px; margin: 10px 0 0; font-size: 0.78rem; }
  .keys kbd:not(:first-child) { margin-left: 8px; }
  .keys kbd + kbd { margin-left: 0; }
  @media not ((hover: hover) and (pointer: fine)) { .keys { display: none; } }
  .block { margin-top: 36px; }
  .kicker { margin: 0 0 12px; font-size: 0.75rem; font-weight: 650; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }

  /* the selected card; the search stays in reach below the site header */
  .top-search { position: sticky; top: 56px; z-index: 15; margin: -8px 0 14px; padding: 8px 0; background: var(--bg); }
  .selected h1:focus { outline: none; }
  .busy :is(.selected, .similar) { opacity: 0.55; transition: opacity 0.2s 0.15s; }
  .selected { display: grid; grid-template-columns: 180px minmax(0, 1fr); gap: 22px; padding-bottom: 24px; border-bottom: 1px solid var(--line); }
  .selected img { width: 180px; height: auto; aspect-ratio: 488 / 680; border-radius: 9px; background: var(--sunken); }
  .selected-body { min-width: 0; }
  .selected h1 { font-size: clamp(1.6rem, 2.6vw, 2.1rem); margin: 2px 0 12px; }
  .selected :global(.card-text) { font-size: 1.02rem; }
  .cost { font-size: 1.05rem; vertical-align: 0.2em; white-space: nowrap; letter-spacing: 0; }
  .tags { display: flex; flex-wrap: wrap; gap: 5px; align-items: center; margin-top: 14px; font-size: 0.82rem; }
  .links { display: flex; flex-wrap: wrap; gap: 6px 18px; margin: 14px 0 0; font-size: 0.9rem; }

  /* ranking */
  .similar { margin-top: 26px; }
  .similar h2 { margin-bottom: 14px; }
  .rank { display: grid; gap: 8px; }
  .rank-row { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; }
  .rank-help { max-width: 70ch; margin: 0; font-size: 0.86rem; }
  .by-label { margin-top: 8px; }
  .by { display: flex; flex-direction: column; gap: 2px; }
  .by button { max-width: 100%; padding: 6px 10px; border: 0; border-left: 2px solid var(--line); background: none; font-size: 0.88rem; text-align: left; overflow-wrap: anywhere; color: var(--muted); }
  .by button:hover { color: var(--ink); border-left-color: var(--muted); }
  .by button[aria-pressed='true'] { border-left-color: var(--ink); color: var(--ink); background: var(--surface); }

  /* filters, then the bar over the list: count, active filters, view */
  .controls { display: flex; flex-direction: column; margin-top: 20px; }
  .fields { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 14px 28px; padding: 14px 16px; border: 1px solid var(--line); border-radius: calc(var(--radius) + 2px); background: var(--surface); }
  .field { display: grid; gap: 6px; }
  .ctl { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  .fields input { height: 34px; padding: 0 10px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--bg); font-size: 0.88rem; }
  .fields input.type { width: 150px; }
  .mv { display: inline-flex; align-items: center; height: 34px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--bg); color: var(--muted); }
  .mv:focus-within { border-color: var(--ink); }
  .fields .mv input { width: 62px; height: 32px; border: 0; background: none; text-align: center; }
  .list-bar { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 12px; margin-top: 14px; padding-bottom: 12px; border-bottom: 1px solid var(--line); }
  .count { margin: 0; font-size: 0.86rem; color: var(--muted); }
  .count strong { color: var(--ink); font-weight: 650; }
  .active { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 0; padding: 0; list-style: none; font-size: 0.8rem; }
  .fchip { display: inline-flex; align-items: center; gap: 6px; height: 26px; padding: 0 7px 0 10px; border: 1px solid var(--line); border-radius: 999px; background: var(--surface); font-size: 0.78rem; font-weight: 550; }
  .fchip span { font-size: 1rem; line-height: 1; color: var(--muted); }
  .fchip:hover { border-color: var(--ink); }
  .fchip:hover span { color: var(--ink); }
  .view { margin-left: auto; }
  .ftoggle { display: none; }
  .badge { display: inline-grid; place-items: center; min-width: 18px; height: 18px; padding: 0 5px; border-radius: 9px; background: var(--ink); color: var(--bg); font-size: 0.72rem; font-weight: 700; }

  /* results */
  .results { margin: 0; padding: 0; list-style: none; }
  .results > li { display: grid; min-width: 0; border-bottom: 1px solid var(--line); }
  .results.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr)); gap: 22px 16px; padding-top: 16px; }
  .results.grid > li { border-bottom: 0; }
  .results.grid .ghost { min-height: 0; aspect-ratio: 488 / 680; margin: 0; }
  .ghost { min-height: 120px; margin: 8px 0; border-radius: var(--radius); background: var(--sunken); }
  @media (prefers-reduced-motion: no-preference) { .ghost { animation: pulse 1.4s ease-in-out infinite; } }
  @keyframes pulse { 50% { opacity: 0.55; } }
  .empty { display: grid; justify-items: start; gap: 10px; padding: 26px 0 8px; }
  .empty p { margin: 0; color: var(--muted); }
  .more { margin: 18px auto 0; display: flex; }
  .model-note { font-size: 0.8rem; margin-top: 22px; }

  @media (max-width: 1100px) {
  }
  @media (max-width: 900px) {
    .explorer { grid-template-columns: minmax(0, 1fr); gap: 14px; padding-top: 12px; }
    /* at the foot of the screen, in thumb reach, while the explorer is in view */
    .tabs { display: flex; order: 3; position: sticky; bottom: max(10px, env(safe-area-inset-bottom)); z-index: 18; border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; box-shadow: 0 6px 20px rgb(0 0 0 / 18%); }
    .tabs button { flex: 1; padding: 11px 7px; border: 0; background: var(--surface); font-weight: 500; color: var(--muted); }
    .tabs button + button { border-left: 1px solid var(--line); }
    .tabs button[aria-selected='true'] { background: var(--ink); color: var(--bg); }
    .explorer:not(.has-card) .tabs { display: none; }
    .explorer:not(.has-card) .map-pane { display: block; position: static; height: 52dvh; min-height: 300px; order: 2; }
    .map-pane { position: relative; top: 0; height: calc(100dvh - 150px); min-height: 360px; }
    .hidden { display: none; }
    .explorer:not(.has-card) .results-pane { display: block; }
    .map-bar { display: flex; position: absolute; left: 10px; right: 52px; top: 10px; align-items: center; justify-content: space-between; gap: 8px; padding: 6px 6px 6px 12px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); font-weight: 600; font-size: 0.9rem; }
    .map-bar span { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
    .intro { padding-top: 8px; }
    .block { margin-top: 28px; }
    .ftoggle { display: inline-flex; height: 34px; }
    .ftoggle[aria-expanded='true'] { border-color: var(--ink); }
    .list-bar { order: 1; margin-top: 0; }
    .fields { display: none; order: 2; flex-direction: column; align-items: stretch; margin-top: 12px; }
    .fields.open { display: flex; }
    .view { order: 1; }
    .active { order: 2; flex-basis: 100%; }
    .count { flex-basis: 100%; order: 3; }
    /* on a phone the map follows the intro, so keep the intro short */
    .selected { grid-template-columns: 110px minmax(0, 1fr); gap: 14px; }
    .selected img { width: 110px; }
  }
  @media (max-width: 480px) {
    .selected { grid-template-columns: 92px minmax(0, 1fr); gap: 12px; }
    .selected img { width: 92px; border-radius: 5px; }
    .selected h1 { font-size: 1.45rem; }
    .random { width: 48px; padding: 0; justify-content: center; }
    .random .lbl { position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }
  }
</style>

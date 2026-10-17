<script>
  import { filterNeighbours, parseUrl, searchNames, serializeUrl, validateRelease, validateShard } from './lib.js';

  let release;
  let names;
  let selectedIndex = -1;
  let selectedCard;
  let savedNeighbours = [];
  let visibleNeighbours = [];
  let suggestions = [];
  let query = '';
  let activeSuggestion = -1;
  let loading = true;
  let error = '';
  let unknownId = false;
  let color = '';
  let minMv = '';
  let maxMv = '';
  let creature = '';
  let type = '';
  let imageFailed = false;
  let token = 0;
  let filterToken = 0;
  const cache = new Map();

  const filters = () => ({ color, minMv: minMv === '' ? null : Number(minMv), maxMv: maxMv === '' ? null : Number(maxMv), creature, type });
  const base = import.meta.env.BASE_URL;
  const fetchJson = async (url) => {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`Could not load ${url} (${response.status}).`);
    return response.json();
  };
  const shardFor = (index) => Math.floor(index / release.shard_size);
  const getShard = async (shard) => {
    if (!cache.has(shard)) cache.set(shard, Promise.all([
      fetchJson(`${base}data/${release.release_id}/cards/${String(shard).padStart(3, '0')}.json`),
      fetchJson(`${base}data/${release.release_id}/neighbours/${String(shard).padStart(3, '0')}.json`)
    ]).then(([cards, neighbours]) => {
      validateShard(cards, neighbours, release, shard);
      return { cards, neighbours };
    }).catch((reason) => { cache.delete(shard); throw reason; }));
    return cache.get(shard);
  };
  const getCard = async (index) => (await getShard(shardFor(index))).cards[index % release.shard_size];
  const updateUrl = (push = true, card = selectedCard?.id ?? parseUrl(location.search).card) => {
    const queryString = serializeUrl({ card, ...filters() });
    const next = `${location.pathname}${queryString ? `?${queryString}` : ''}`;
    history[push ? 'pushState' : 'replaceState']({}, '', next);
  };
  // urlMode: 'push' (filter change), 'none' (selection path; selectIndex writes the URL itself)
  const refreshFilters = async (urlMode = 'push', expectedToken = token) => {
    if (selectedIndex < 0 || !selectedCard || !release) return;
    const thisFilterToken = ++filterToken;
    // Neighbour cards may belong to other uncached shards, so load them before applying card filters.
    const shards = [...new Set(savedNeighbours.map(([index]) => shardFor(index)))];
    await Promise.all(shards.map((shard) => getShard(shard)));
    if (expectedToken !== token || thisFilterToken !== filterToken) return;
    const cardCache = new Map();
    for (const [index] of savedNeighbours) cardCache.set(index, await getCard(index));
    if (expectedToken !== token || thisFilterToken !== filterToken) return;
    visibleNeighbours = filterNeighbours(savedNeighbours, (index) => cardCache.get(index), filters());
    if (urlMode === 'push') updateUrl(true);
  };
  const selectIndex = async (index, push = true) => {
    const request = ++token;
    loading = true;
    error = '';
    unknownId = false;
    imageFailed = false;
    selectedIndex = index;
    try {
      const [card, shardData] = await Promise.all([getCard(index), getShard(shardFor(index))]);
      if (request !== token) return;
      selectedCard = card;
      savedNeighbours = shardData.neighbours[index % release.shard_size];
      visibleNeighbours = savedNeighbours;
      if (card.has_text) await refreshFilters('none', request);
      if (request !== token) return;
      loading = false;
      if (push) updateUrl(true, names.ids[index]);
    } catch (reason) {
      if (request !== token) return;
      error = reason.message;
      loading = false;
    }
  };
  const selectId = (id, push = true) => {
    const index = names?.ids.indexOf(id) ?? -1;
    if (index < 0) {
      token += 1;
      selectedIndex = -1;
      selectedCard = undefined;
      savedNeighbours = [];
      visibleNeighbours = [];
      unknownId = true;
      loading = false;
      error = '';
      if (push) updateUrl(true, id);
      return;
    }
    query = names.names[index];
    selectIndex(index, push);
  };
  const onSearch = (event) => {
    query = event.currentTarget.value;
    suggestions = searchNames(query, names ?? { ids: [], names: [], aliases: [] });
    activeSuggestion = suggestions.length ? 0 : -1;
  };
  const chooseSuggestion = (suggestion) => {
    suggestions = [];
    query = suggestion.name;
    selectIndex(suggestion.index);
  };
  const searchKeydown = (event) => {
    if (event.key === 'ArrowDown' && suggestions.length) { event.preventDefault(); activeSuggestion = (activeSuggestion + 1) % suggestions.length; }
    else if (event.key === 'ArrowUp' && suggestions.length) { event.preventDefault(); activeSuggestion = (activeSuggestion - 1 + suggestions.length) % suggestions.length; }
    else if (event.key === 'Enter' && suggestions.length) { event.preventDefault(); chooseSuggestion(suggestions[Math.max(activeSuggestion, 0)]); }
    else if (event.key === 'Escape') suggestions = [];
  };
  const onFilter = () => refreshFilters('push').catch((reason) => { error = reason.message; });
  const onHistory = () => {
    const state = parseUrl(location.search);
    color = state.color; minMv = state.minMv ?? ''; maxMv = state.maxMv ?? ''; creature = state.creature; type = state.type;
    if (state.card) selectId(state.card, false);
    else { selectedCard = undefined; selectedIndex = -1; }
  };

  (async () => {
    try {
      const current = await fetchJson(`${base}data/current.json`);
      if (!current.release_id || current.release_id.includes('/') || current.release_id.includes('..')) throw new Error('Invalid release pointer.');
      release = await fetchJson(`${base}data/${current.release_id}/release.json`);
      names = await fetchJson(`${base}data/${current.release_id}/names.json`);
      validateRelease(release, names);
      const state = parseUrl(location.search);
      color = state.color; minMv = state.minMv ?? ''; maxMv = state.maxMv ?? ''; creature = state.creature; type = state.type;
      loading = false;
      if (state.card) selectId(state.card, false);
    } catch (reason) { error = reason.message; loading = false; }
  })();
  addEventListener('popstate', onHistory);
</script>

<svelte:head>
  <title>Card Atlas — MTG effect discovery</title>
  <meta name="description" content="Explore ranked effect neighbours for a card." />
</svelte:head>

<header class="topbar">
  <a class="brand" href={base} aria-label="Card Atlas home"><span class="brand-mark">✳</span><span>Card Atlas<small>Effect discovery</small></span></a>
  <span class="topnote">A field guide to familiar spells</span>
</header>

<main>
  {#if release?.fixture}<div class="fixture-banner" role="status">FIXTURE DATA <span>Synthetic examples for interface testing</span></div>{/if}
  <section class="intro">
    <p class="eyebrow">THE CARD INDEX <span>·</span> SNAPSHOT {release?.snapshot_date ?? '—'}</p>
    <h1>Find cards by<br /><em>what they do.</em></h1>
    <p class="lede">Start with a card. Explore its closest effects, then narrow the saved list to what you need.</p>
    <div class="search-wrap">
      <label for="card-search">Search by card or face name</label>
      <div class="searchbox"><span aria-hidden="true">⌕</span><input id="card-search" role="combobox" aria-autocomplete="list" aria-controls="suggestions" aria-expanded={suggestions.length > 0} aria-activedescendant={activeSuggestion >= 0 ? `option-${activeSuggestion}` : undefined} placeholder="Try “Aster Fold”…" bind:value={query} oninput={onSearch} onkeydown={searchKeydown} /><kbd>↵</kbd></div>
      {#if suggestions.length}<ul id="suggestions" class="suggestions" role="listbox">{#each suggestions as option, i}<li id={`option-${i}`} role="option" aria-selected={i === activeSuggestion}><button type="button" onmousedown={(event) => event.preventDefault()} onclick={() => chooseSuggestion(option)}>{option.name}<span>{option.id}</span></button></li>{/each}</ul>{/if}
    </div>
  </section>

  {#if error}<div class="error" role="alert"><strong>Bundle could not be loaded</strong><span>{error}</span></div>{/if}
  {#if unknownId}<div class="notice" role="status"><strong>Card not found</strong><span>This Oracle ID is not in the current release.</span></div>{/if}
  {#if loading && !error}<p class="loading" role="status">Loading card data…</p>{/if}

  {#if selectedCard}
    <section class="workspace" aria-label="Card discovery results">
      <article class="card-panel">
        <div class="section-kicker"><span>SELECTED CARD</span><span class="index-id">{selectedCard.id}</span></div>
        {#if selectedCard.image && !imageFailed}<img class="card-image" src={selectedCard.image} alt={`Card image for ${selectedCard.name}`} onerror={() => imageFailed = true} />{:else}<div class="image-placeholder" aria-label="Card image unavailable"><span>✳</span><small>IMAGE UNAVAILABLE</small></div>{/if}
        <div class="card-content">
          {#each selectedCard.faces as face}<div class="face"><h2>{face.name}</h2><p class="mana">{face.mana_cost || '—'}</p><p class="type-line">{face.type_line}</p><p class="oracle">{face.oracle_text}</p></div>{/each}
          <div class="stats"><div><small>COLOUR IDENTITY</small><strong>{selectedCard.color_identity || 'Colourless'}</strong></div><div><small>MANA VALUE</small><strong>{selectedCard.mv.toFixed(1)}</strong></div></div>
          <a class="scryfall" href={selectedCard.scryfall_uri} target="_blank" rel="noreferrer">Open card reference <span>↗</span></a>
        </div>
      </article>

      <section class="results" aria-labelledby="neighbours-title">
        <div class="results-heading"><div><p class="eyebrow">RANKED BY EFFECT SIMILARITY</p><h2 id="neighbours-title">Nearby ideas <span>{savedNeighbours.length}</span></h2></div><div class="model-chip"><i></i>{release?.model?.status ?? 'unknown'}</div></div>
        {#if !selectedCard.has_text}<div class="notice empty"><strong>No neighbours for this card</strong><span>This card has no rules text, so no neighbours were computed.</span></div>
        {:else}
          <p class="filter-label">Filtering the {savedNeighbours.length} saved neighbours</p>
          <div class="filters">
            <label>Colour identity<select bind:value={color} onchange={onFilter}><option value="">Any colour</option><option value="W">White</option><option value="U">Blue</option><option value="B">Black</option><option value="R">Red</option><option value="G">Green</option><option value="C">Colourless</option></select></label>
            <label>Min MV<input type="number" min="0" step="0.5" bind:value={minMv} onchange={onFilter} /></label>
            <label>Max MV<input type="number" min="0" step="0.5" bind:value={maxMv} onchange={onFilter} /></label>
            <label>Card kind<select bind:value={creature} onchange={onFilter}><option value="">Any</option><option value="creature">Creature</option><option value="noncreature">Noncreature</option></select></label>
            <label class="type-filter">Type text<input type="search" placeholder="e.g. artifact" bind:value={type} oninput={onFilter} /></label>
          </div>
          {#if visibleNeighbours.length}
            <ol class="neighbour-list">{#each visibleNeighbours as [index, score], position}<li><button type="button" class="neighbour" onclick={() => selectIndex(index)}><span class="rank">{String(position + 1).padStart(2, '0')}</span><span class="neighbour-name">{names.names[index]}<small>{names.ids[index]}</small></span><span class="score"><strong>{score.toFixed(3)}</strong><small>cosine (ranking score, not a probability)</small></span><span class="arrow">↗</span></button></li>{/each}</ol>
          {:else}<div class="notice empty"><strong>No matches in this list</strong><span>Try changing or clearing one of the filters.</span></div>{/if}
        {/if}
      </section>
    </section>
  {:else if !loading && !error && !unknownId}
    <section class="empty-start"><div class="orbit">✳</div><h2>Your next discovery<br />starts with a card.</h2><p>Search the catalogue to see related effects, ranked and ready to explore.</p><div class="start-rule"></div><span>NO QUERY MODEL NEEDED</span></section>
  {/if}
</main>

<footer><span>Card Atlas <i>·</i> {release?.model?.label ?? 'Release metadata unavailable'}</span><span>{release?.attribution}</span></footer>

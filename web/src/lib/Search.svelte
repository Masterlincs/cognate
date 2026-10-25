<script>
  import { findCards } from './search.js';

  // names: names.json (null while loading); onpick(index); recent: [{ index, name }] offered while the box is empty
  let { names, onpick, onfocus = null, recent = [], id = 'card-search', label = 'Search for a card', placeholder = 'Card name' } = $props();
  let query = $state('');
  let active = $state(-1);
  let open = $state(false);
  let input;
  const matches = $derived(names && query.trim() ? findCards(query, names, 8) : query.trim() ? [] : recent);
  const options = $derived(open ? matches : []);
  const status = $derived(!open || !query.trim() ? '' : !names ? 'Loading card names…' : matches.length ? '' : 'No matching card.');

  function pick(option) {
    query = '';
    open = false;
    active = -1;
    onpick(option.index);
  }
  function clear() {
    query = '';
    active = -1;
    open = false;
    input.focus();
  }
  function keydown(e) {
    // ArrowDown also reopens the list after Escape, or shows recent cards in an empty box
    if (e.key === 'ArrowDown' && matches.length) { e.preventDefault(); active = open ? (active + 1) % matches.length : 0; open = true; }
    else if (e.key === 'ArrowUp' && options.length) { e.preventDefault(); active = (active - 1 + options.length) % options.length; }
    else if (e.key === 'Enter' && options.length && (query.trim() || active >= 0)) { e.preventDefault(); pick(options[Math.max(0, active)]); }
    else if (e.key === 'Escape') { e.preventDefault(); if (options.length || status) { open = false; active = -1; } else query = ''; }
  }
</script>

<div class="search">
  <label class="sr-only" for={id}>{label}</label>
  <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></svg>
  <input {id} bind:this={input} type="search" role="combobox" autocomplete="off" spellcheck="false" aria-autocomplete="list" aria-controls="{id}-list" aria-expanded={options.length > 0} aria-activedescendant={active >= 0 && options.length ? `${id}-${active}` : undefined}
    {placeholder} bind:value={query} onfocus={() => { open = Boolean(query.trim()); onfocus?.(); }} onclick={() => (open = true)} oninput={() => { open = true; active = -1; }} onkeydown={keydown} onblur={() => setTimeout(() => (open = false), 120)} />
  {#if query}<button type="button" class="clear" aria-label="Clear search" onclick={clear}>×</button>{/if}
  {#if options.length}
    <ul id="{id}-list" role="listbox" aria-label={query.trim() ? 'Matching cards' : 'Recent cards'}>
      {#each options as option, i (option.index)}
        <!-- svelte-ignore a11y_click_events_have_key_events (keyboard selection is on the combobox input) -->
        <li id="{id}-{i}" role="option" aria-selected={i === active} class:active={i === active} onmousedown={(e) => e.preventDefault()} onclick={() => pick(option)}>{option.name}</li>
      {/each}
    </ul>
  {:else if status}
    <p class="status" role="status">{status}</p>
  {/if}
</div>

<style>
  .search { position: relative; width: 100%; }
  svg { position: absolute; left: 14px; top: 50%; width: 18px; height: 18px; transform: translateY(-50%); fill: none; stroke: var(--muted); stroke-width: 2; pointer-events: none; }
  input { width: 100%; height: 48px; padding: 0 44px 0 42px; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); font-size: 1rem; }
  input:focus { outline: none; border-color: var(--ink); box-shadow: 0 0 0 1px var(--ink); }
  input::-webkit-search-cancel-button { display: none; }
  .clear { position: absolute; right: 6px; top: 50%; transform: translateY(-50%); width: 36px; height: 36px; border: 0; border-radius: var(--radius); background: none; font-size: 1.4rem; line-height: 1; color: var(--muted); }
  .clear:hover { color: var(--ink); }
  ul, .status { position: absolute; z-index: 30; left: 0; right: 0; top: calc(100% + 6px); margin: 0; padding: 6px; list-style: none; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); box-shadow: 0 8px 24px rgb(0 0 0 / 12%); }
  .status { padding: 10px 16px; color: var(--muted); font-size: 0.92rem; }
  li { padding: 7px 10px; border-radius: 4px; cursor: pointer; }
  li.active, li:hover { background: var(--accent-soft); }
</style>

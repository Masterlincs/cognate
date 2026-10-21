<script>
  import { searchNames } from './bundle.js';

  // names: names.json (null while loading); onpick(index)
  let { names, onpick, id = 'card-search', label = 'Search for a card', placeholder = 'Search any card, e.g. Lightning Bolt' } = $props();
  let query = $state('');
  let active = $state(-1);
  let open = $state(false);
  const options = $derived(names && open ? searchNames(query, names, null, 8) : []);

  function pick(option) {
    query = '';
    open = false;
    active = -1;
    onpick(option.index);
  }
  function keydown(e) {
    if (e.key === 'ArrowDown' && options.length) { e.preventDefault(); open = true; active = (active + 1) % options.length; }
    else if (e.key === 'ArrowUp' && options.length) { e.preventDefault(); active = (active - 1 + options.length) % options.length; }
    else if (e.key === 'Enter' && options.length) { e.preventDefault(); pick(options[Math.max(0, active)]); }
    else if (e.key === 'Escape') { open = false; active = -1; }
  }
</script>

<div class="search">
  <label class="sr-only" for={id}>{label}</label>
  <svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="11" cy="11" r="7" /><path d="m20 20-4-4" /></svg>
  <input {id} type="search" role="combobox" autocomplete="off" spellcheck="false" aria-autocomplete="list" aria-controls="{id}-list" aria-expanded={options.length > 0} aria-activedescendant={active >= 0 ? `${id}-${active}` : undefined}
    placeholder={names ? placeholder : 'Loading card names…'} disabled={!names} bind:value={query} oninput={() => { open = true; active = -1; }} onkeydown={keydown} onblur={() => setTimeout(() => (open = false), 120)} />
  {#if options.length}
    <ul id="{id}-list" role="listbox">
      {#each options as option, i (option.index)}
        <!-- svelte-ignore a11y_click_events_have_key_events (keyboard selection is on the combobox input) -->
        <li id="{id}-{i}" role="option" aria-selected={i === active} class:active={i === active} onmousedown={(e) => e.preventDefault()} onclick={() => pick(option)}>{option.name}</li>
      {/each}
    </ul>
  {/if}
</div>

<style>
  .search { position: relative; width: 100%; }
  svg { position: absolute; left: 14px; top: 50%; width: 18px; height: 18px; transform: translateY(-50%); fill: none; stroke: var(--muted); stroke-width: 2; pointer-events: none; }
  input { width: 100%; height: 48px; padding: 0 16px 0 42px; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); font-size: 1rem; box-shadow: var(--shadow); }
  input:focus { outline: none; border-color: var(--accent); box-shadow: 0 0 0 3px var(--accent-soft); }
  ul { position: absolute; z-index: 30; left: 0; right: 0; top: calc(100% + 6px); margin: 0; padding: 6px; list-style: none; background: var(--surface); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); }
  li { padding: 8px 10px; border-radius: 8px; cursor: pointer; }
  li.active, li:hover { background: var(--accent-soft); }
</style>

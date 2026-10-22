<script>
  // Pick any combination of WUBRG (or colourless) and how a card's colour identity must match it.
  // color: '' | 'C' | a WUBRG-ordered subset; mode: include | exact | within;
  // basis: identity (every mana symbol on the card) | color (the card's printed colours, v5 bundles)
  import Segmented from './Segmented.svelte';
  let { color, mode, basis = 'identity', showBasis = false, onchange } = $props();
  const BASES = [{ value: 'identity', label: 'Colour identity' }, { value: 'color', label: 'Colour' }];
  const COLOURS = [['W', 'White'], ['U', 'Blue'], ['B', 'Black'], ['R', 'Red'], ['G', 'Green']];
  const MODES = [{ value: 'include', label: 'Including' }, { value: 'exact', label: 'Exactly' }, { value: 'within', label: 'At most' }];

  function toggle(c) {
    if (c === 'C') return onchange(color === 'C' ? '' : 'C', mode, basis);
    const now = color === 'C' ? '' : color;
    onchange([...'WUBRG'].filter((x) => (x === c) !== now.includes(x)).join(''), mode, basis);
  }
</script>

<div class="colour">
  <div class="pips" role="group" aria-label="Colours">
    {#each COLOURS as [c, name] (c)}
      <button type="button" aria-pressed={color !== 'C' && color.includes(c)} title={name} onclick={() => toggle(c)}><i class="ms ms-cost ms-{c.toLowerCase()}" aria-hidden="true"></i><span class="sr-only">{name}</span></button>
    {/each}
    <button type="button" aria-pressed={color === 'C'} title="Colourless" onclick={() => toggle('C')}><i class="ms ms-cost ms-c" aria-hidden="true"></i><span class="sr-only">Colourless</span></button>
  </div>
  {#if color && color !== 'C'}
    <Segmented label="How colours match" options={MODES} value={mode} onchange={(m) => onchange(color, m, basis)} />
  {/if}
  {#if color && showBasis}
    <Segmented label="Match against" options={BASES} value={basis} onchange={(b) => onchange(color, mode, b)} />
  {/if}
</div>

<style>
  .colour { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  .pips { display: flex; gap: 2px; }
  button { display: grid; place-items: center; width: 32px; height: 32px; padding: 0; border: 0; border-radius: 50%; background: none; font-size: 1.15rem; }
  /* glyphs stay full colour for contrast; the ring alone marks a picked colour */
  button:hover .ms { box-shadow: 0 0 0 2px var(--bg), 0 0 0 4px var(--line); }
  button[aria-pressed='true'] .ms { box-shadow: 0 0 0 2px var(--bg), 0 0 0 4px var(--ink); }
</style>

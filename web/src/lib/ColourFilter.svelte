<script>
  // Pick any combination of WUBRG (or colourless) and how a card's colour identity must match it.
  // color: '' | 'C' | a WUBRG-ordered subset; mode: include | exact | within
  import Segmented from './Segmented.svelte';
  let { color, mode, onchange } = $props();
  const COLOURS = [['W', 'White'], ['U', 'Blue'], ['B', 'Black'], ['R', 'Red'], ['G', 'Green']];
  const MODES = [{ value: 'include', label: 'Including' }, { value: 'exact', label: 'Exactly' }, { value: 'within', label: 'At most' }];

  function toggle(c) {
    if (c === 'C') return onchange(color === 'C' ? '' : 'C', mode);
    const now = color === 'C' ? '' : color;
    onchange([...'WUBRG'].filter((x) => (x === c) !== now.includes(x)).join(''), mode);
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
    <Segmented label="How colours match" options={MODES} value={mode} onchange={(m) => onchange(color, m)} />
  {/if}
</div>

<style>
  .colour { display: flex; flex-wrap: wrap; align-items: center; gap: 8px; }
  .pips { display: flex; gap: 2px; }
  button { display: grid; place-items: center; width: 32px; height: 32px; padding: 0; border: 0; border-radius: 50%; background: none; font-size: 1.15rem; }
  button .ms { opacity: 0.5; filter: grayscale(0.6); transition: opacity 0.12s; }
  button:hover .ms { opacity: 0.8; }
  button[aria-pressed='true'] .ms { opacity: 1; filter: none; box-shadow: 0 0 0 2px var(--bg), 0 0 0 4px var(--ink); }
</style>

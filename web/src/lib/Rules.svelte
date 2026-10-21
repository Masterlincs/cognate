<script>
  // One rules-text line with {X} symbols drawn as small mana pills (hybrid ones split in two).
  import { manaSymbol } from './text.js';
  let { text } = $props();
  const parts = $derived(text.split(/(\{[^}]+\})/).filter(Boolean));
  const fill = (cs) => cs.length === 2 ? `linear-gradient(135deg, var(--wubrg-${cs[0]}) 50%, var(--wubrg-${cs[1]}) 50%)` : cs.length ? `var(--wubrg-${cs[0]})` : null;
</script>

{#each parts as part, i (i)}{#if /^\{[^}]+\}$/.test(part)}{@const s = manaSymbol(part.slice(1, -1))}<abbr class="sym" class:n={!s.colours.length} class:light={s.colours[0] === 'w' && s.colours.length === 1} title={part} style:background={fill(s.colours)}>{s.text}<span class="sr-only">{s.text ? '' : part}</span></abbr>{:else}{part}{/if}{/each}

<style>
  .sym { display: inline-grid; place-items: center; min-width: 1.25em; height: 1.25em; margin: 0 0.06em; padding: 0 0.2em; border-radius: 999px; font: 600 0.72em/1 var(--mono); text-decoration: none; vertical-align: 0.08em; color: #fff; }
  .light { color: #2a2418; }
  .n { background: var(--sunken); color: var(--ink); box-shadow: inset 0 0 0 1px var(--line); }
</style>

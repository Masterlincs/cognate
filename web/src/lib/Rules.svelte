<script>
  // One rules-text line with {X} symbols drawn in the Mana font; a symbol it lacks stays as text.
  import { manaClass } from './text.js';
  let { text } = $props();
  const parts = $derived(text.split(/(\{[^}]+\})/).filter(Boolean));
</script>

{#each parts as part, i (i)}{#if /^\{[^}]+\}$/.test(part)}{@const c = manaClass(part.slice(1, -1))}{#if c}<i class="ms ms-cost ms-{c}" title={part} aria-hidden="true"></i><span class="sr-only">{part}</span>{:else}{part}{/if}{:else}{part}{/if}{/each}

<style>
  .ms { margin: 0 0.05em; font-size: 0.82em; vertical-align: 0.04em; }
</style>

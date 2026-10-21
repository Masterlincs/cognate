<script>
  // One rules-text line with {X} symbols drawn as small mana pills.
  let { text } = $props();
  const parts = $derived(text.split(/(\{[^}]+\})/).filter(Boolean));
  const cls = (sym) => /^[WUBRGC]$/.test(sym) ? sym.toLowerCase() : sym === 'T' ? 'tap' : 'n';
</script>

{#each parts as part, i (i)}{#if /^\{[^}]+\}$/.test(part)}<abbr class="sym {cls(part.slice(1, -1))}" title={part}>{part.slice(1, -1) === 'T' ? '↷' : part.slice(1, -1)}</abbr>{:else}{part}{/if}{/each}

<style>
  .sym { display: inline-grid; place-items: center; min-width: 1.25em; height: 1.25em; margin: 0 0.06em; padding: 0 0.2em; border-radius: 999px; font: 600 0.72em/1 var(--mono); text-decoration: none; vertical-align: 0.08em; background: var(--wubrg-c); color: #fff; }
  .w { background: var(--wubrg-w); color: #2a2418; }
  .u { background: var(--wubrg-u); }
  .b { background: var(--wubrg-b); }
  .r { background: var(--wubrg-r); }
  .g { background: var(--wubrg-g); }
  .n, .tap { background: var(--sunken); color: var(--ink); box-shadow: inset 0 0 0 1px var(--line); }
</style>

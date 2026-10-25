<script>
  // A row of mutually exclusive buttons on a sunken track. options: [{ value, label, hint?, icon? }]:
  // hint is the button's tooltip; icon (an SVG path in a 16-unit box) shows instead of the label,
  // which stays as the accessible name
  let { label, options, value, onchange, describedby = undefined, full = false } = $props();
</script>

<div class="seg" class:full role="group" aria-label={label} aria-describedby={describedby}>
  {#each options as o (o.value)}
    <button type="button" class:icon={o.icon} aria-pressed={value === o.value} title={o.hint ?? (o.icon ? o.label : undefined)} onclick={() => onchange(o.value)}>
      {#if o.icon}<svg viewBox="0 0 16 16" aria-hidden="true"><path d={o.icon} /></svg><span class="sr-only">{o.label}</span>{:else}{o.label}{/if}
    </button>
  {/each}
</div>

<style>
  .seg { display: inline-flex; gap: 2px; padding: 3px; border-radius: calc(var(--radius) + 3px); background: var(--sunken); box-shadow: inset 0 0 0 1px var(--line); }
  .full { display: flex; }
  .full button { flex: 1; }
  button { display: inline-flex; align-items: center; justify-content: center; height: 28px; padding: 0 11px; border: 0; border-radius: var(--radius); background: none; font-size: 0.84rem; font-weight: 550; color: var(--muted); white-space: nowrap; }
  button.icon { width: 32px; padding: 0; }
  svg { width: 15px; height: 15px; fill: none; stroke: currentColor; stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round; }
  button:hover { color: var(--ink); }
  button[aria-pressed='true'] { background: var(--raised); color: var(--ink); box-shadow: 0 0 0 1px var(--line), 0 1px 3px rgb(0 0 0 / 0.12); }
  button:focus-visible { outline-offset: 0; }
</style>

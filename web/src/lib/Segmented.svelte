<script>
  // A row of mutually exclusive buttons. options: [{ value, label }]
  let { label, options, value, onchange } = $props();
</script>

<div class="seg" role="group" aria-label={label}>
  {#each options as o (o.value)}
    <button type="button" aria-pressed={value === o.value} onclick={() => onchange(o.value)}>{o.label}</button>
  {/each}
</div>

<style>
  .seg { display: inline-flex; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); overflow: hidden; }
  button { padding: 5px 11px; border: 0; background: none; font-size: 0.84rem; color: var(--muted); white-space: nowrap; }
  button + button { border-left: 1px solid var(--line); }
  button:hover { color: var(--ink); }
  button[aria-pressed='true'] { background: var(--ink); color: var(--bg); }
  button:focus-visible { outline-offset: -2px; }
  /* the accent ring vanishes on the pressed button's ink fill; use the page colour there */
  button[aria-pressed='true']:focus-visible { outline-color: var(--bg); outline-offset: -4px; }
</style>

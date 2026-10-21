<script>
  import CardText from './CardText.svelte';
  import { sharedLines } from './text.js';

  // a: selected card, b: neighbour (or null when closed)
  let { a, b = null, score = null, onclose, onselect } = $props();
  let dialog;
  const shared = $derived(a && b ? sharedLines(a, b) : new Set());

  $effect(() => {
    if (b && !dialog.open) dialog.showModal();
    if (!b && dialog.open) dialog.close();
  });
</script>

<dialog bind:this={dialog} onclose={() => onclose?.()} onclick={(e) => { if (e.target === dialog) dialog.close(); }} aria-labelledby="compare-title">
  {#if a && b}
    <header>
      <h2 id="compare-title">{a.name} <span class="muted">vs</span> {b.name}</h2>
      <button type="button" class="close" onclick={() => dialog.close()} aria-label="Close">×</button>
    </header>
    <p class="meta">
      {#if score !== null}Similarity <span class="num">{score.toFixed(3)}</span>. {/if}
      {shared.size ? `${shared.size} ${shared.size > 1 ? 'lines appear' : 'line appears'} on both cards (highlighted; numbers and card names ignored).` : 'No line appears on both cards, so the similarity does not come from identical wording.'}
    </p>
    <div class="cols">
      {#each [a, b] as card, i (i)}
        <section>
          {#if card.image}<img src={card.image} alt="" loading="lazy" width="488" height="680" />{/if}
          <h3>{card.name}</h3>
          <CardText {card} highlight={shared} />
        </section>
      {/each}
    </div>
    <footer><button type="button" class="btn primary" onclick={() => { onselect?.(b); dialog.close(); }}>Explore {b.name}</button></footer>
  {/if}
</dialog>

<style>
  dialog { width: min(920px, calc(100vw - 24px)); max-height: calc(100dvh - 24px); padding: 20px; border: 1px solid var(--line); border-radius: 10px; background: var(--bg); color: var(--ink); box-shadow: 0 20px 60px rgb(0 0 0 / 30%); }
  dialog::backdrop { background: rgb(10 12 16 / 55%); }
  header { display: flex; justify-content: space-between; gap: 12px; align-items: start; }
  h2 { font-size: 1.4rem; }
  .close { border: 0; background: none; font-size: 1.6rem; line-height: 1; padding: 0 4px; color: var(--muted); }
  .meta { color: var(--muted); font-size: 0.88rem; margin: 8px 0 16px; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  section { min-width: 0; padding: 14px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); }
  img { width: min(220px, 100%); aspect-ratio: 488 / 680; height: auto; border-radius: 9px; margin: 0 auto 12px; }
  h3 { margin-bottom: 6px; }
  footer { display: flex; justify-content: flex-end; margin-top: 16px; }
  @media (max-width: 620px) {
    .cols { grid-template-columns: 1fr; }
    img { width: 140px; }
  }
</style>

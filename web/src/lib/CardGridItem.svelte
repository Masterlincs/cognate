<script>
  import CardText from './CardText.svelte';
  import { faceName, flipKind } from './faces.js';

  let { card, score = null, rank = null, oncompare = null, onselect = null } = $props();
  let flipped = $state(false);
  const kind = $derived(flipKind(card));
  const src = $derived(flipped && kind === 'back' ? card.image_back : card.image);
  // the list reuses a cell for another card when the order changes; start it face up
  $effect(() => { card.id; flipped = false; });
</script>

<article class="cell">
  <div class="frame">
    <button type="button" class="pick" onclick={() => onselect?.(card)} aria-label={card.name}>
      {#if src}
        <img {src} alt={faceName(card, flipped)} class:turned={flipped && kind === 'rotate'} loading="lazy" decoding="async" width="488" height="680" />
      {:else}
        <span class="blank"><strong>{card.name}</strong><CardText {card} clamp={8} /></span>
      {/if}
    </button>
    {#if kind && src}
      <button type="button" class="flip" aria-pressed={flipped} aria-label="Show other face of {card.name}" title="Show other face" onclick={() => (flipped = !flipped)}>Flip</button>
    {/if}
  </div>
  <div class="meta">
    {#if rank !== null}<span class="rank num">{rank}</span>{/if}
    {#if score !== null}<span class="score num" title="Similarity to the selected card">{score.toFixed(2)}</span>{/if}
    {#if oncompare}<button type="button" class="btn" onclick={() => oncompare(card)}>Compare</button>{/if}
  </div>
</article>

<style>
  .cell { min-width: 0; }
  .frame { position: relative; }
  .pick { display: block; width: 100%; padding: 0; border: 0; background: none; cursor: pointer; border-radius: 4.75% / 3.4%; }
  .pick:focus-visible { outline: 2px solid var(--ink); outline-offset: 3px; }
  img, .blank { display: block; width: 100%; height: auto; aspect-ratio: 488 / 680; border-radius: 4.75% / 3.4%; background: var(--sunken); }
  img.turned { transform: rotate(180deg); }
  .blank { display: flex; flex-direction: column; gap: 6px; padding: 10px; overflow: hidden; text-align: left; font-size: 0.8rem; border: 1px solid var(--line); }
  .flip { position: absolute; right: 6px; bottom: 6px; padding: 2px 8px; font-size: 0.75rem; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); color: var(--ink); }
  .flip[aria-pressed='true'] { background: var(--ink); color: var(--surface); border-color: var(--ink); }
  .meta { display: flex; align-items: center; gap: 8px; margin-top: 6px; min-width: 0; }
  .rank { color: var(--muted); font-size: 0.8rem; }
  .score { font-weight: 600; font-size: 0.9rem; }
  .meta .btn { margin-left: auto; padding: 2px 8px; font-size: 0.75rem; }
  @media (prefers-reduced-motion: no-preference) { img { transition: transform 0.3s ease; } }
</style>

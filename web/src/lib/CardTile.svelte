<script>
  import CardText from './CardText.svelte';
  import Rules from './Rules.svelte';
  import { cardImage } from './faces.js';

  // scoreLabel: what the score is (the tooltip on it)
  let { card, score = null, rank = null, highlight = null, scoreLabel = 'Similarity to the selected card', oncompare = null, onselect = null } = $props();
  const cost = $derived(card.faces[0]?.mana_cost ?? '');
</script>

<article class="row">
  <!-- a mouse shortcut beside the name button, which keyboards use; hovering it previews the card -->
  <button type="button" class="art" tabindex="-1" aria-hidden="true" onclick={() => onselect?.(card)}>
    {#if card.image}
      <img src={cardImage(card.image, 'small')} alt="" loading="lazy" decoding="async" width="146" height="204" />
      <img class="peek" src={card.image} alt="" loading="lazy" decoding="async" width="488" height="680" />
    {/if}
  </button>
  <div class="body">
    <header>
      {#if rank !== null}<span class="rank num">{rank}</span>{/if}
      <h3><button type="button" class="name" onclick={() => onselect?.(card)}>{card.name}</button></h3>
      {#if cost && card.faces.length === 1}<span class="cost"><Rules text={cost} /></span>{/if}
    </header>
    <CardText {card} {highlight} clamp={5} />
  </div>
  <div class="side">
    {#if score !== null}
      <span class="score num" title="{scoreLabel}: {score.toFixed(4)}. Differences in the third decimal place are not meaningful.">{score.toFixed(2)}</span>
    {/if}
    {#if oncompare}<button type="button" class="btn ghost compare" onclick={() => oncompare(card)} aria-label="Compare {card.name} with the selected card">Compare</button>{/if}
  </div>
</article>

<style>
  .row { position: relative; display: grid; grid-template-columns: 64px minmax(0, 1fr) auto; gap: 14px; padding: 14px 6px 14px 0; min-width: 0; }
  .art { position: relative; z-index: 1; width: 64px; aspect-ratio: 146 / 204; padding: 0; border: 0; border-radius: 4px; background: var(--sunken); }
  .art > img:first-child { width: 100%; height: 100%; object-fit: cover; border-radius: 4px; }
  .peek { display: none; position: absolute; left: calc(100% + 12px); top: -10px; width: 230px; max-width: none; height: auto; border-radius: 11px; box-shadow: 0 14px 40px rgb(0 0 0 / 0.35); pointer-events: none; }
  @media (hover: hover) and (pointer: fine) { .art:hover .peek { display: block; } }
  .body { min-width: 0; }
  header { display: flex; align-items: baseline; gap: 8px; margin-bottom: 2px; }
  .rank { color: var(--muted); font-size: 0.8rem; min-width: 1.4em; }
  h3 { min-width: 0; }
  .name { all: unset; cursor: pointer; overflow-wrap: anywhere; }
  /* the name button covers the row, so the whole row opens the card; Compare sits above it */
  .name::after { content: ''; position: absolute; inset: 0; }
  .row:hover .name { text-decoration: underline; text-decoration-color: color-mix(in srgb, currentColor 40%, transparent); text-underline-offset: 0.2em; }
  .name:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
  .cost { white-space: nowrap; font-size: 0.92rem; }
  .side { display: flex; flex-direction: column; align-items: flex-end; justify-content: space-between; gap: 8px; }
  .score { font-size: 0.95rem; font-weight: 650; }
  .compare { position: relative; padding: 3px 9px; font-size: 0.78rem; }
  .row:hover .compare { border-color: var(--line); color: var(--ink); }
  @media (max-width: 520px) {
    .row { grid-template-columns: 52px minmax(0, 1fr); }
    .art { width: 52px; }
    .side { grid-column: 2; flex-direction: row; align-items: center; }
  }
</style>

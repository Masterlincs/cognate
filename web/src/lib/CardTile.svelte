<script>
  import CardText from './CardText.svelte';
  import Rules from './Rules.svelte';

  let { card, score = null, rank = null, highlight = null, oncompare = null, onselect = null } = $props();
  const small = (url) => url?.replace('/normal/', '/small/');
  const cost = $derived(card.faces[0]?.mana_cost ?? '');
</script>

<article class="row">
  <div class="art">
    {#if card.image}<img src={small(card.image)} alt="" loading="lazy" decoding="async" width="146" height="204" />{/if}
  </div>
  <div class="body">
    <header>
      {#if rank !== null}<span class="rank num">{rank}</span>{/if}
      <h3><button type="button" class="name" onclick={() => onselect?.(card)}>{card.name}</button></h3>
      {#if cost && card.faces.length === 1}<span class="cost"><Rules text={cost} /></span>{/if}
    </header>
    <CardText {card} {highlight} clamp={5} />
  </div>
  <div class="side">
    {#if score !== null}<span class="score num" title="Similarity to the selected card">{score.toFixed(2)}</span>{/if}
    {#if oncompare}<button type="button" class="btn" onclick={() => oncompare(card)}>Compare</button>{/if}
  </div>
</article>

<style>
  .row { display: grid; grid-template-columns: 64px minmax(0, 1fr) auto; gap: 14px; padding: 14px 0; min-width: 0; }
  .art { width: 64px; aspect-ratio: 146 / 204; border-radius: 4px; overflow: hidden; background: var(--sunken); }
  .art img { width: 100%; height: 100%; object-fit: cover; }
  .body { min-width: 0; }
  header { display: flex; align-items: baseline; gap: 8px; margin-bottom: 2px; }
  .rank { color: var(--muted); font-size: 0.8rem; min-width: 1.4em; }
  h3 { min-width: 0; }
  .name { all: unset; cursor: pointer; overflow-wrap: anywhere; }
  .name:hover { text-decoration: underline; text-underline-offset: 0.2em; }
  .name:focus-visible { outline: 2px solid var(--ink); outline-offset: 2px; }
  .cost { white-space: nowrap; font-size: 0.92rem; }
  .side { display: flex; flex-direction: column; align-items: flex-end; justify-content: space-between; gap: 8px; }
  .score { font-size: 0.95rem; font-weight: 600; }
  .btn { padding: 3px 10px; font-size: 0.8rem; }
  @media (max-width: 520px) {
    .row { grid-template-columns: 52px minmax(0, 1fr); }
    .art { width: 52px; }
    .side { grid-column: 2; flex-direction: row; align-items: center; }
  }
</style>

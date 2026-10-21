<script>
  import CardText from './CardText.svelte';
  import Rules from './Rules.svelte';

  let { card, score = null, rank = null, highlight = null, oncompare = null, onselect = null } = $props();
  const small = (url) => url?.replace('/normal/', '/small/');
  const cost = $derived(card.faces[0]?.mana_cost ?? '');
</script>

<article class="tile">
  <div class="art">
    {#if card.image}<img src={small(card.image)} alt="" loading="lazy" decoding="async" width="146" height="204" />{/if}
  </div>
  <div class="body">
    <header>
      {#if rank !== null}<span class="rank num">{rank}</span>{/if}
      <h3><button type="button" class="name" onclick={() => onselect?.(card)}>{card.name}</button></h3>
      {#if cost && card.faces.length === 1}<span class="cost"><Rules text={cost} /></span>{/if}
    </header>
    <CardText {card} {highlight} clamp={6} />
    <footer>
      {#if score !== null}
        <span class="score" title="Similarity to the selected card"><span class="bar" style:--s={Math.max(0, score)}></span><span class="num">{score.toFixed(2)}</span></span>
      {/if}
      {#if oncompare}<button type="button" class="btn" onclick={() => oncompare(card)}>Compare</button>{/if}
    </footer>
  </div>
</article>

<style>
  .tile { display: grid; grid-template-columns: 88px 1fr; gap: 12px; padding: 12px; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); min-width: 0; }
  .art { width: 88px; aspect-ratio: 146 / 204; border-radius: 5px; overflow: hidden; background: var(--sunken); }
  .art img { width: 100%; height: 100%; object-fit: cover; }
  .body { min-width: 0; display: flex; flex-direction: column; gap: 6px; }
  header { display: flex; align-items: baseline; gap: 8px; }
  .rank { color: var(--muted); font-size: 0.78rem; }
  h3 { flex: 1; min-width: 0; font-size: 1.02rem; }
  .name { all: unset; cursor: pointer; overflow-wrap: anywhere; }
  .name:hover { color: var(--accent); }
  .name:focus-visible { outline: 2px solid var(--accent); }
  .cost { white-space: nowrap; font-size: 0.9rem; }
  footer { margin-top: auto; display: flex; align-items: center; justify-content: space-between; gap: 8px; }
  .score { display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: 0.78rem; }
  .bar { width: 48px; height: 4px; border-radius: 2px; background: linear-gradient(90deg, var(--accent) calc(var(--s) * 100%), var(--sunken) 0); }
  .btn { padding: 4px 11px; font-size: 0.8rem; }
</style>

<script>
  import Rules from './Rules.svelte';
  import { cardImage } from './faces.js';

  // items: [{ id, name, card }], card null until its shard loads; onpick(id) opens one in place
  let { items, onpick } = $props();
  const firstLine = (c) => (c.faces[0]?.oracle_text ?? '').split('\n').find((t) => t.trim()) ?? '';
</script>

<ul class="examples">
  {#each items as { id, name, card }, n (id)}
    <li>
      <a class="tile" href="?card={id}" onclick={(e) => { if (!e.ctrlKey && !e.metaKey && !e.shiftKey && onpick(id)) e.preventDefault(); }}>
        <span class="art">{#if card?.image}<img src={cardImage(card.image, 'small')} alt="" loading={n < 4 ? 'eager' : 'lazy'} fetchpriority={n < 4 ? 'high' : 'auto'} decoding="async" width="146" height="204" />{/if}</span>
        <span class="body">
          <span class="head"><strong>{name}</strong>{#if card?.faces[0]?.mana_cost}<span class="cost"><Rules text={card.faces[0].mana_cost} /></span>{/if}</span>
          {#if card}
            <span class="type">{card.type_line}</span>
            <span class="text"><Rules text={firstLine(card)} /></span>
          {:else}
            <span class="bone" aria-hidden="true"></span><span class="bone short" aria-hidden="true"></span>
          {/if}
        </span>
      </a>
    </li>
  {/each}
</ul>

<style>
  .examples { display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 12px; margin: 0; padding: 0; list-style: none; }
  li { display: grid; min-width: 0; }
  .tile { display: flex; flex-direction: column; min-width: 0; overflow: hidden; border: 1px solid var(--line); border-radius: calc(var(--radius) + 2px); background: var(--surface); text-decoration: none; }
  .tile:hover { border-color: color-mix(in srgb, var(--ink) 45%, var(--line)); }
  .tile:hover img { transform: scale(1.04); }
  .tile:focus-visible { outline-offset: 2px; }
  .art { display: block; aspect-ratio: 5 / 7; overflow: hidden; background: var(--sunken); padding: 10px 14px 0; }
  img { display: block; width: 100%; height: auto; aspect-ratio: 488 / 680; border-radius: 4.8% / 3.4%; }
  .body { display: flex; flex-direction: column; gap: 2px; padding: 9px 11px 11px; min-width: 0; }
  .head { display: flex; align-items: baseline; justify-content: space-between; gap: 6px; }
  strong { font-weight: 650; font-size: 0.92rem; line-height: 1.25; }
  .cost { flex: none; font-size: 0.8rem; white-space: nowrap; }
  .type { color: var(--muted); font-size: 0.76rem; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .text { display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: 2; line-clamp: 2; overflow: hidden; margin-top: 3px; font-family: var(--serif); font-size: 0.82rem; line-height: 1.35; color: var(--ink); }
  .bone { display: block; height: 0.7rem; margin-top: 6px; border-radius: 3px; background: var(--sunken); }
  .bone.short { width: 60%; }
  @media (prefers-reduced-motion: no-preference) { img { transition: transform 0.4s ease; } }
  @media (max-width: 520px) {
    .examples { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .text { display: none; }
  }
</style>

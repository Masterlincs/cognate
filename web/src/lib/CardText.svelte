<script>
  import Rules from './Rules.svelte';
  import { lineKey } from './text.js';

  // highlight: Set of lineKey()s to mark (lines shared with another card)
  let { card, highlight = null, clamp = 0 } = $props();
  const multi = $derived(card.faces.length > 1);
</script>

<div class="card-text" style:--clamp={clamp || null} class:clamped={clamp > 0}>
  {#each card.faces as face, f (f)}
    <div class="face">
      {#if multi}<p class="face-name">{face.name} {#if face.mana_cost}<Rules text={face.mana_cost} />{/if}</p>{/if}
      <p class="type">{face.type_line}</p>
      {#each (face.oracle_text ?? '').split('\n').filter((t) => t.trim()) as line, l (l)}
        {@const shared = highlight?.has(lineKey(card, line.trim()))}
        <p class="line" class:shared>{#if shared}<span class="sr-only">Shared line: </span>{/if}<Rules text={line} /></p>
      {/each}
      {#if !(face.oracle_text ?? '').trim()}<p class="line muted">No rules text.</p>{/if}
    </div>
  {/each}
</div>

<style>
  .card-text { font-size: 0.9rem; line-height: 1.5; }
  .clamped { display: -webkit-box; -webkit-box-orient: vertical; -webkit-line-clamp: var(--clamp); line-clamp: var(--clamp); overflow: hidden; }
  .face + .face { margin-top: 0.6em; padding-top: 0.6em; border-top: 1px dashed var(--line); }
  p { margin: 0 0 0.3em; }
  .face-name { font-weight: 600; }
  .type { color: var(--muted); font-size: 0.82rem; }
  .shared { background: var(--mark); box-shadow: -4px 0 0 var(--mark), 4px 0 0 var(--mark); border-radius: 2px; }
</style>

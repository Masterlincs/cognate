<script>
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { resolve } from '$app/paths';

  // the app loads this page's code at start-up, so its fonts load only when it is shown
  onMount(() => { import('#lib/fonts.js'); });
</script>

<svelte:head><title>{page.status} · Card Atlas</title></svelte:head>

<div class="error">
  <h1>{page.status === 404 ? 'Page not found' : 'Something went wrong'}</h1>
  <p class="muted">Error {page.status}</p>
  <p><a href={resolve('/')}>Back to the explorer</a></p>
</div>

<style>
  .error { max-width: 760px; margin: 0 auto; padding: 80px clamp(16px, 4vw, 32px); }
  h1 { font-size: clamp(2rem, 5vw, 3rem); margin: 8px 0 16px; }
</style>

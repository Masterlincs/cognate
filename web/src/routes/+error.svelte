<script>
  import { page } from '$app/state';
  import { resolve } from '$app/paths';
</script>

<svelte:head><title>{page.status} · Card Atlas</title></svelte:head>

<div class="error">
  <p class="eyebrow">Error {page.status}</p>
  <h1>{page.status === 404 ? 'Page not found' : 'Something went wrong'}</h1>
  <p><a href={resolve('/')}>Back to the explorer</a></p>
</div>

<style>
  .error { max-width: 760px; margin: 0 auto; padding: 80px clamp(16px, 4vw, 32px); }
  h1 { font-size: clamp(2rem, 5vw, 3rem); margin: 8px 0 16px; }
</style>

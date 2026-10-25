<script>
  import '../app.css';
  import { onMount } from 'svelte';
  import { page } from '$app/state';
  import { resolve } from '$app/paths';

  let { children } = $props();
  const REPO = 'https://github.com/Masterlincs/mtg-embeddings';
  let dark = $state(false);
  onMount(() => { dark = document.documentElement.dataset.theme === 'dark'; });

  function toggleTheme() {
    const next = dark ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    dark = next === 'dark';
    try { localStorage.setItem('theme', next); } catch { /* private mode */ }
  }
</script>

<a class="skip" href="#main">Skip to content</a>
<header class="site-header">
  <a class="wordmark" href={resolve('/')}>Card Atlas</a>
  <nav aria-label="Main">
    <a href={resolve('/')} aria-current={page.url.pathname.endsWith('/how-it-works') ? undefined : 'page'}>Explore</a>
    <a href={resolve('/how-it-works')} aria-current={page.url.pathname.endsWith('/how-it-works') ? 'page' : undefined}>How it works</a>
    <a href={REPO} rel="noopener">GitHub</a>
    <button class="theme" type="button" onclick={toggleTheme} aria-label="Toggle dark mode" aria-pressed={dark}>
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3a9 9 0 1 0 9 9A7 7 0 0 1 12 3z" /></svg>
    </button>
  </nav>
</header>

<main id="main">{@render children()}</main>

<footer class="site-footer">
  <p class="fine">Card Atlas is unofficial Fan Content permitted under the <a href="https://company.wizards.com/en/legal/fancontentpolicy" rel="noopener">Fan Content Policy</a>. Not approved/endorsed by Wizards. Portions of the materials used are property of Wizards of the Coast. ©Wizards of the Coast LLC. Card data and images from <a href="https://scryfall.com" rel="noopener">Scryfall</a>, tags from Scryfall Tagger; not affiliated with Scryfall.</p>
</footer>

<script>
  import '@fontsource-variable/schibsted-grotesk';
  import '@fontsource-variable/source-serif-4';
  import '#lib/vendor/mana.css';
  import '../app.css';
  import { page } from '$app/state';
  import { resolve } from '$app/paths';

  let { children } = $props();
  const REPO = 'https://github.com/Masterlincs/mtg-embeddings';

  function toggleTheme() {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
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
    <button class="theme" type="button" onclick={toggleTheme} aria-label="Toggle dark mode">
      <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 3a9 9 0 1 0 9 9A7 7 0 0 1 12 3z" /></svg>
    </button>
  </nav>
</header>

<main id="main">{@render children()}</main>

<footer class="site-footer">
  <p>Card Atlas finds Magic: The Gathering cards that do similar things, using a sentence-embedding model fine-tuned on card rules text. <a href={REPO} rel="noopener">Source and write-up on GitHub</a>.</p>
  <p class="fine">Card data and images © Wizards of the Coast; data from <a href="https://scryfall.com" rel="noopener">Scryfall</a>. Tags from Scryfall Tagger. Not affiliated with or endorsed by Wizards of the Coast or Scryfall.</p>
</footer>

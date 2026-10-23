// The site's web fonts and the mana symbol font. The explorer imports this after its search index
// has loaded, so they do not compete with it on a slow connection; nothing on its start
// screen uses mana symbols. Other pages import it directly.
import '@fontsource-variable/schibsted-grotesk';
import '@fontsource-variable/source-serif-4';
import './vendor/mana.css';

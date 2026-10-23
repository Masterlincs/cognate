// Load the web fonts and mana symbols after start-up. Cards wait for this, so it settles after
// at most `wait` ms. If the fonts fail or are still loading then, the page gets the class no-mana,
// and app.css shows each mana symbol's text ({R}, {T}) in place of a blank icon until they arrive.
export function loadFonts(load = () => import('./fonts.js'), root = document.documentElement, wait = 5000) {
  let settled = false;
  const loading = load().then(
    () => { settled = true; root.classList.remove('no-mana'); },
    () => { settled = true; root.classList.add('no-mana'); }
  );
  const timeout = new Promise((done) => setTimeout(done, wait)).then(() => { if (!settled) root.classList.add('no-mana'); });
  return Promise.race([loading, timeout]);
}

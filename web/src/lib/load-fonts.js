// Load the web fonts and mana symbols after start-up. If they fail, the page gets the class
// no-mana, and app.css then shows each mana symbol's text ({R}, {T}) in place of a blank icon.
export const loadFonts = (load = () => import('./fonts.js'), root = document.documentElement) =>
  load().then(() => {}, () => { root.classList.add('no-mana'); });

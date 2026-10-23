// Start the explorer's start-up requests from the page itself, before the app's code has loaded
//. The build (scripts/prune-build.mjs) bakes in the deployed release id. data.js takes these
// responses once; it fetches again if one is missing or failed, or if current.json names another release.
export function prefetch(win, fetchFn, id) {
  const early = (win.__atlasPrefetch = {});
  for (const path of ['current.json', `${id}/release.json`, `${id}/names.json`]) {
    const request = fetchFn(`data/${path}`);
    request.catch(() => {});
    early[path] = request;
  }
}

// first in <head>: an inline script after a stylesheet waits for that stylesheet to load
export function withPrefetch(html, id) {
  if (!/^[\w.-]+$/.test(id) || id.includes('..')) throw new Error(`Invalid release id ${id}.`);
  if (!html.includes('<head>')) throw new Error('No <head> to add the start-up prefetch to.');
  return html.replace('<head>', `<head><script>(${prefetch})(window, fetch, ${JSON.stringify(id)})</script>`);
}

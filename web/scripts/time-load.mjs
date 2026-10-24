// P9 load timing: headless Chrome over CDP on a throttled mobile profile. Times how long until typing
// "Lightning Bolt" into the search box lists options (usable lookup), with the cache disabled.
// usage: node scripts/time-load.mjs <url> [slow4g|fast4g|none] [runs]
// Env: CHROME=<path to chrome>; GAP_MS=<pause between runs, default 6000> so one run's downloads
// do not spill into the next; TIMELINE=1 adds each resource's start and end times.
import { spawn } from 'node:child_process';
import { mkdtempSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const [url, profileName = 'slow4g', runs = '3'] = process.argv.slice(2);
// slow4g is Lighthouse's mobile throttling; fast4g is the Chrome DevTools "Fast 4G" preset
const PROFILES = {
  slow4g: { latency: 150, down: 1638.4, up: 675, cpu: 4 },
  fast4g: { latency: 165, down: 9000, up: 1500, cpu: 4 },
  none: null,
};
if (!url || !(profileName in PROFILES)) throw new Error('usage: node scripts/time-load.mjs <url> [slow4g|fast4g|none] [runs]');
const profile = PROFILES[profileName];
const chrome = process.env.CHROME ?? 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const port = 9300 + Math.floor(Math.random() * 500);
const browser = spawn(chrome, ['--headless=new', `--remote-debugging-port=${port}`,
  `--user-data-dir=${mkdtempSync(join(tmpdir(), 'time-load-'))}`, '--no-first-run', 'about:blank'], { stdio: 'ignore' });
const sleep = (ms) => new Promise((done) => setTimeout(done, ms));

let target;
for (let i = 0; i < 50 && !target; i++) {
  await sleep(200);
  try { target = (await (await fetch(`http://127.0.0.1:${port}/json`)).json()).find((t) => t.type === 'page'); } catch {}
}
if (!target) throw new Error(`Chrome did not start (${chrome}).`);
const ws = new WebSocket(target.webSocketDebuggerUrl);
await new Promise((done) => ws.addEventListener('open', done));
let id = 0;
const pending = new Map();
const logs = [];
ws.addEventListener('message', (e) => {
  const m = JSON.parse(e.data);
  if (pending.has(m.id)) { pending.get(m.id)(m); pending.delete(m.id); }
  if (m.method === 'Runtime.exceptionThrown') logs.push(['exception', m.params.exceptionDetails.exception?.description ?? m.params.exceptionDetails.text]);
});
const send = (method, params = {}) => new Promise((done) => { pending.set(++id, done); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async (expression) => (await send('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true })).result?.result?.value;

await send('Network.enable');
await send('Runtime.enable');
await send('Network.setCacheDisabled', { cacheDisabled: true });
await send('Emulation.setDeviceMetricsOverride', { width: 375, height: 812, deviceScaleFactor: 3, mobile: true });
if (profile) {
  await send('Network.emulateNetworkConditions', { offline: false, latency: profile.latency,
    downloadThroughput: profile.down * 1024 / 8, uploadThroughput: profile.up * 1024 / 8 });
  await send('Emulation.setCPUThrottlingRate', { rate: profile.cpu });
}
const usable = `(() => {
  const input = document.querySelector('[role=combobox]');
  if (!input || input.disabled) return false;
  Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value').set.call(input, 'Lightning Bolt');
  input.dispatchEvent(new Event('input', { bubbles: true }));
  return document.querySelectorAll('[role=option]').length > 0;
})()`;

const results = [];
for (let run = 1; run <= Number(runs); run++) {
  await send('Page.navigate', { url: 'about:blank' });
  await sleep(Number(process.env.GAP_MS ?? 6000));
  const t0 = Date.now();
  await send('Page.navigate', { url });
  let ok = false;
  while (!ok && Date.now() - t0 < 60000) { await sleep(50); ok = await evaluate(usable).catch(() => false); }
  const ms = Date.now() - t0;
  const transfer = await evaluate(`performance.getEntriesByType('resource').reduce((s, e) => s + e.transferSize, 0) + (performance.getEntriesByType('navigation')[0]?.transferSize || 0)`);
  const timeline = process.env.TIMELINE ? await evaluate(`performance.getEntriesByType('resource').map((e) => [e.name.split('/').pop(), Math.round(e.startTime), Math.round(e.responseEnd), e.transferSize])`) : undefined;
  results.push({ run, usable_lookup_ms: ok ? ms : null, transfer_bytes: transfer, timeline, logs: logs.splice(0) });
}
console.log(JSON.stringify({ url, profile: profileName, settings: profile, viewport: '375x812 mobile', cache: 'disabled', results }));
ws.close();
browser.kill();

// Serve build/ as a static host would, with gzip, for load timing. `vite preview` serves
// the Kit prerender output, not build/. usage: node scripts/serve-build.mjs [port]
// Env: NOPREFETCH=1 strips the start-up prefetch (baseline); BLOCK=<regex> answers matching paths
// with 404 and STALL=<regex> never answers them (failure checks).
import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize } from 'node:path';
import { gzipSync } from 'node:zlib';

const dir = new URL('../build/', import.meta.url).pathname.replace(/^\/(\w:)/, '$1');
const port = Number(process.argv[2] ?? 4318);
const types = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.json': 'application/json', '.woff2': 'font/woff2', '.bin': 'application/octet-stream', '.svg': 'image/svg+xml', '.png': 'image/png' };
// the `/*` rules from build/_headers, so the CSP is exercised here as a host would apply it
const hostText = await readFile(join(dir, '_headers'), 'utf8').catch(() => '');
const hostHeaders = Object.fromEntries((hostText.split(/\n(?=\S)/).find((b) => b.startsWith('/*\n')) ?? '').split('\n').slice(1).map((l) => l.trim().split(/:\s+/)).filter((p) => p.length === 2));
const block = process.env.BLOCK && new RegExp(process.env.BLOCK);
const stall = process.env.STALL && new RegExp(process.env.STALL);

createServer(async (req, res) => {
  let p;
  try { p = normalize(decodeURIComponent(new URL(req.url, 'http://x').pathname)).replace(/^[\\/]+/, ''); } catch { return res.writeHead(400).end(); }
  if (p.includes('..')) return res.writeHead(400).end();
  if (stall?.test(p)) return;
  if (block?.test(p)) return res.writeHead(404).end();
  let file = join(dir, p);
  try { if ((await stat(file)).isDirectory()) file = join(file, 'index.html'); } catch { file += '.html'; }
  try {
    let body = await readFile(file);
    if (process.env.NOPREFETCH && file.endsWith('.html')) body = Buffer.from(body.toString().replace(/<script>\(function prefetch[\s\S]*?<\/script>/, ''));
    const type = types[extname(file)] ?? 'application/octet-stream';
    const headers = { ...hostHeaders, 'Content-Type': type, 'Cache-Control': 'no-cache' };
    if (/html|javascript|css|json|svg/.test(type) && /gzip/.test(req.headers['accept-encoding'] ?? '')) {
      body = gzipSync(body);
      headers['Content-Encoding'] = 'gzip';
    }
    res.writeHead(200, headers).end(body);
  } catch {
    res.writeHead(404).end();
  }
}).listen(port, '127.0.0.1', () => console.log(`serving build/ on http://127.0.0.1:${port}/`));

<script>
  import { onMount } from 'svelte';
  import { buildGrid, colorKey, hitTest, TEXTLESS, toScreen, toWorld } from './map.js';
  import { dataBounds, densityGrid, EXTENT, inkOn, labelBudget, labelStyle, LEGEND, PALETTE, pointRadius } from './map-style.js';

  // map: decodeMap() result; regions: [{x, y, label, n}]; names: names.json; selected: index or -1;
  // highlight: a card to ring (the result row under the pointer), or -1
  let { map, regions = [], names, selected = -1, neighbours = [], highlight = -1, onselect } = $props();

  const TAU = Math.PI * 2;
  const SHADE = EXTENT * 1.15; // the density wash reaches past the outermost cards so its blur fades out
  let wrap, canvas, ctx, shade, controls, legend;
  let caption = $state(null); // the caption over the map, which labels keep clear of
  let width = 0, height = 0, dpr = 1;
  let v = { scale: 1, tx: 0, ty: 0 };
  let fit = 1;
  let tip = $state(null);
  let ready = $state(false);
  let palette = $state.raw(PALETTE.light);
  let hoverKey = ''; // a colour hovered in the legend
  let pinned = $state(''); // a colour picked in the legend; the others fade
  let focusKey = '';
  let hint = $state(true); // the how-to-read caption, until it is closed or a card is picked
  let frame = 0;
  let anim = 0;
  const grid = $derived(buildGrid(map.x, map.y, 1024));
  const keys = $derived(Array.from(map.color, colorKey));
  // cards dealt into a few passes, each bucketed by colour: a pass is one fill per colour,
  // and alternating passes keeps any one colour from always landing on top
  const passes = $derived.by(() => {
    const out = Array.from({ length: 4 }, () => ({}));
    for (let i = 0; i < map.n; i += 1) if (map.x[i] !== TEXTLESS) (out[i % 4][keys[i]] ??= []).push(i);
    return out;
  });
  const density = $derived(densityGrid(map.x, map.y, 160, SHADE, TEXTLESS));
  const bounds = $derived(dataBounds(map.x, map.y, TEXTLESS));
  const placed = $derived.by(() => { let k = 0; for (let i = 0; i < map.n; i += 1) if (map.x[i] !== TEXTLESS) k += 1; return k; });
  // regions by size, largest first: the order labels are placed in, and their rank for labelStyle
  const ranked = $derived([...regions].sort((a, b) => b.n - a.n));

  let colors = {};
  function readColors() {
    const css = getComputedStyle(document.documentElement);
    for (const k of ['ink', 'surface']) colors[k] = css.getPropertyValue(`--${k}`).trim();
    palette = PALETTE[document.documentElement.dataset.theme === 'dark' ? 'dark' : 'light'];
    paintShade();
  }

  // faint ink wash where cards are dense, so the shape of the map reads before any dot does;
  // a small canvas under the dots, scaled by a CSS transform so the compositor does the work
  function paintShade() {
    const n = Math.sqrt(density.length);
    shade.width = shade.height = n;
    const g = shade.getContext('2d'), img = g.createImageData(n, n);
    const [r, gr, b] = [1, 3, 5].map((k) => parseInt(colors.ink.slice(k, k + 2), 16));
    for (let i = 0; i < density.length; i += 1) {
      img.data[i * 4] = r; img.data[i * 4 + 1] = gr; img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = Math.round(255 * Math.sqrt(density[i]));
    }
    g.putImageData(img, 0, 0);
  }

  // frame the cards themselves, clear of the controls above and the legend below
  function fitAll() {
    const b = bounds ?? { x0: -EXTENT, y0: -EXTENT, x1: EXTENT, y1: EXTENT };
    const top = 20, bottom = 52, side = 20;
    fit = Math.max(1e-6, Math.min((width - 2 * side) / Math.max(1, b.x1 - b.x0), (height - top - bottom) / Math.max(1, b.y1 - b.y0)));
    v = { scale: fit, tx: width / 2 - ((b.x0 + b.x1) / 2) * fit, ty: top + (height - top - bottom) / 2 - ((b.y0 + b.y1) / 2) * fit };
  }

  // The density wash and the dots are rendered into an offscreen layer a margin larger than the
  // view. Frames reuse it, shifted while panning and scaled while zooming, and it is rendered
  // again when the view leaves it, zooms well past it, or has rested for a moment.
  const MARGIN = 0.25;
  let layer = null, layerV = null, layerKey = '', settle = 0;

  function renderLayer(dim, key) {
    const mw = width * MARGIN, mh = height * MARGIN;
    layer ??= document.createElement('canvas');
    const lw = Math.round((width + 2 * mw) * dpr), lh = Math.round((height + 2 * mh) * dpr);
    if (layer.width !== lw || layer.height !== lh) { layer.width = lw; layer.height = lh; }
    const g = layer.getContext('2d');
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.clearRect(0, 0, lw, lh);
    g.setTransform(dpr, 0, 0, dpr, mw * dpr, mh * dpr); // screen coordinates of the current view
    const r = pointRadius(v.scale, fit, map.n);
    // smaller dots behind a selection, so its neighbours stand out; from a few pixels across,
    // each dot gets a thin surface-coloured rim so neighbours stay separate instead of merging
    const pr = dim ? Math.max(1.1, r * 0.7) : r, rim = pr >= 2.5 ? Math.min(1, pr * 0.2) : 0;
    const base = dim ? 0.36 : 0.9, faded = base * 0.14;
    const pad = pr + rim, x0 = -mw - pad, y0 = -mh - pad, x1 = width + mw + pad, y1 = height + mh + pad;
    const path = (list, rad) => {
      g.beginPath();
      for (const i of list) {
        const sx = v.tx + map.x[i] * v.scale, sy = v.ty + map.y[i] * v.scale;
        if (sx < x0 || sy < y0 || sx > x1 || sy > y1) continue;
        g.moveTo(sx + rad, sy);
        g.arc(sx, sy, rad, 0, TAU);
      }
    };
    for (const part of passes) {
      if (rim) {
        g.globalAlpha = 1;
        g.fillStyle = colors.surface;
        for (const list of Object.values(part)) { path(list, pr + rim); g.fill(); }
      }
      for (const [k, list] of Object.entries(part)) {
        g.globalAlpha = focusKey && k !== focusKey ? faded : base;
        g.fillStyle = palette[k];
        path(list, pr);
        g.fill();
      }
    }
    layerV = { ...v, mw, mh, lw: lw / dpr, lh: lh / dpr };
    layerKey = key;
  }

  function draw() {
    frame = 0;
    if (!ctx || !width) return;
    const zoom = v.scale / fit;
    const dim = selected >= 0 && map.x[selected] !== TEXTLESS;
    const key = [dim, focusKey, palette.w, width, height, dpr].join();
    // where the layer lands in the current view
    let k = layerV ? v.scale / layerV.scale : 0;
    let dx = layerV ? v.tx + (-layerV.mw - layerV.tx) * k : 0, dy = layerV ? v.ty + (-layerV.mh - layerV.ty) * k : 0;
    const covers = layerV && dx <= 0 && dy <= 0 && dx + layerV.lw * k >= width && dy + layerV.lh * k >= height;
    if (key !== layerKey || !covers || k > 2) {
      renderLayer(dim, key);
      k = 1; dx = -layerV.mw; dy = -layerV.mh;
    }
    clearTimeout(settle);
    if (k !== 1 || dx !== -layerV.mw || dy !== -layerV.mh) settle = setTimeout(() => { layerKey = ''; redraw(); }, 150);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(layer, dx, dy, layerV.lw * k, layerV.lh * k);
    const wash = (dim ? 0.05 : 0.09) * Math.max(0, Math.min(1, 1.4 - zoom / 8));
    const [ox, oy] = toScreen(v, -SHADE, -SHADE);
    shade.style.opacity = wash;
    shade.style.display = wash > 0 ? '' : 'none';
    if (wash > 0) shade.style.transform = `translate(${ox}px, ${oy}px) scale(${(2 * SHADE * v.scale) / shade.width})`;
    drawLabels(zoom, dim);
    const r = pointRadius(v.scale, fit, map.n);
    if (dim) drawSelection(r);
    if (tip) {
      const [hx, hy] = toScreen(v, map.x[tip.i], map.y[tip.i]);
      ring(hx, hy, Math.max(r, 3) + 3.5, colors.ink, 1.5);
    }
    if (highlight >= 0 && highlight !== selected && map.x[highlight] !== TEXTLESS) {
      const [hx, hy] = toScreen(v, map.x[highlight], map.y[highlight]), hr = Math.max(5, r + 2.5);
      dot(hx, hy, hr, palette[keys[highlight]], 2);
      ring(hx, hy, hr + 4, colors.ink, 2);
    }
  }

  const ring = (x, y, rad, stroke, lineWidth) => {
    ctx.beginPath(); ctx.arc(x, y, rad, 0, TAU);
    ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.stroke();
  };
  const dot = (x, y, rad, fill, rim) => {
    ctx.beginPath(); ctx.arc(x, y, rad, 0, TAU);
    ctx.fillStyle = fill; ctx.fill();
    ctx.strokeStyle = colors.surface; ctx.lineWidth = rim; ctx.stroke();
  };

  // spokes to each shown neighbour, fading with rank, then the neighbours (best on top), then
  // the card itself: a dot in its colour inside an ink ring, which stays visible over a tight cluster
  function drawSelection(r) {
    const [cx, cy] = toScreen(v, map.x[selected], map.y[selected]);
    const pts = neighbours.filter((i) => i !== selected && map.x[i] !== TEXTLESS).map((i) => [i, ...toScreen(v, map.x[i], map.y[i])]);
    ctx.strokeStyle = colors.ink;
    ctx.lineWidth = 1;
    pts.forEach(([, sx, sy], k) => {
      ctx.globalAlpha = 0.5 - 0.3 * (k / Math.max(1, pts.length - 1));
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(sx, sy); ctx.stroke();
    });
    ctx.beginPath(); ctx.arc(cx, cy, 16, 0, TAU);
    ctx.fillStyle = colors.ink; ctx.globalAlpha = 0.08; ctx.fill(); ctx.globalAlpha = 1;
    const nr = Math.max(4, r + 1.5);
    for (let k = pts.length - 1; k >= 0; k -= 1) dot(pts[k][1], pts[k][2], nr, palette[keys[pts[k][0]]], 1.5);
    dot(cx, cy, 4.5, palette[keys[selected]], 1.5);
    ring(cx, cy, 9.5, colors.surface, 4.5);
    ring(cx, cy, 9.5, colors.ink, 2.25);
  }

  // larger regions first, and drawn larger (labelStyle); a label is skipped when it would crowd one
  // already drawn or sit under the controls, caption or legend, and a small or zoomed-out map gets
  // fewer labels (labelBudget)
  function drawLabels(zoom, dim) {
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.lineJoin = 'round';
    const placed = [controls, legend, caption].filter(Boolean).map((el) => [el.offsetLeft, el.offsetTop, el.offsetLeft + el.offsetWidth, el.offsetTop + el.offsetHeight]);
    if (dim) {
      const [cx, cy] = toScreen(v, map.x[selected], map.y[selected]);
      placed.push([cx - 18, cy - 18, cx + 18, cy + 18]);
    }
    const most = labelBudget(width, height, zoom) + placed.length;
    for (const [rank, reg] of ranked.entries()) {
      if (placed.length >= most) break;
      const st = labelStyle(rank, zoom);
      ctx.font = `${st.weight} ${st.size}px 'Schibsted Grotesk Variable', system-ui, sans-serif`;
      const [sx, sy] = toScreen(v, reg.x, reg.y);
      const w = ctx.measureText(reg.label).width + 14, h = st.size + 10;
      const box = [sx - w / 2, sy - h / 2, sx + w / 2, sy + h / 2];
      if (box[0] < 4 || box[1] < 4 || box[2] > width - 4 || box[3] > height - 4) continue;
      if (placed.some((p) => box[0] < p[2] && box[2] > p[0] && box[1] < p[3] && box[3] > p[1])) continue;
      placed.push(box);
      ctx.globalAlpha = st.alpha * (dim || focusKey ? 0.6 : 1);
      ctx.lineWidth = 4;
      ctx.strokeStyle = colors.surface;
      ctx.strokeText(reg.label, sx, sy);
      ctx.fillStyle = colors.ink;
      ctx.fillText(reg.label, sx, sy);
    }
    ctx.globalAlpha = 1;
  }

  const redraw = () => { if (!frame) frame = requestAnimationFrame(draw); };

  function zoomAt(sx, sy, factor, smooth = false) {
    const scale = Math.min(fit * 60, Math.max(fit * 0.8, v.scale * factor));
    const [wx, wy] = toWorld(v, sx, sy);
    const target = { scale, tx: sx - wx * scale, ty: sy - wy * scale };
    if (smooth) flyTo(target, 220);
    else { v = target; redraw(); }
  }

  // ease the view to a target, eased in log-scale so zooming feels even
  function flyTo(target, ms = 420) {
    cancelAnimationFrame(anim);
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) { v = target; draw(); return; }
    const from = { ...v }, t0 = performance.now();
    // keep the world point under the screen centre moving in a straight line while scale eases
    const c0 = toWorld(from, width / 2, height / 2), c1 = toWorld(target, width / 2, height / 2);
    const step = (t) => {
      const k = Math.min(1, (t - t0) / ms), e = 1 - (1 - k) ** 3;
      const scale = from.scale * (target.scale / from.scale) ** e;
      const wx = c0[0] + (c1[0] - c0[0]) * e, wy = c0[1] + (c1[1] - c0[1]) * e;
      v = k < 1 ? { scale, tx: width / 2 - wx * scale, ty: height / 2 - wy * scale } : target;
      draw();
      if (k < 1) anim = requestAnimationFrame(step);
    };
    anim = requestAnimationFrame(step);
  }

  // centre the selected card, zoomed so its shown neighbours fit
  function frameSelection() {
    if (selected < 0 || map.x[selected] === TEXTLESS || !width) return;
    const cx = map.x[selected], cy = map.y[selected];
    let reach = 300;
    for (const i of neighbours) if (map.x[i] !== TEXTLESS) reach = Math.max(reach, Math.abs(map.x[i] - cx), Math.abs(map.y[i] - cy));
    const scale = Math.min(fit * 12, Math.max(fit * 1.5, (Math.min(width, height) * 0.36) / reach));
    flyTo({ scale, tx: width / 2 - cx * scale, ty: height / 2 - cy * scale });
  }

  $effect(() => { void selected; void neighbours; frameSelection(); redraw(); });
  $effect(() => { void highlight; redraw(); });
  // a picked card replaces the how-to-read caption for good
  $effect(() => { if (selected >= 0) hint = false; });

  // pointer handling: drag to pan, two pointers to pinch, a click without movement selects
  const pointers = new Map();
  let moved = false, pinch = 0;
  function down(e) {
    canvas.setPointerCapture(e.pointerId);
    cancelAnimationFrame(anim);
    pointers.set(e.pointerId, [e.offsetX, e.offsetY]);
    moved = false;
    if (pointers.size === 2) pinch = dist();
  }
  const dist = () => { const [a, b] = [...pointers.values()]; return Math.hypot(a[0] - b[0], a[1] - b[1]); };
  function move(e) {
    if (!pointers.has(e.pointerId)) { hover(e.offsetX, e.offsetY); return; }
    const [px, py] = pointers.get(e.pointerId);
    pointers.set(e.pointerId, [e.offsetX, e.offsetY]);
    if (pointers.size === 2) {
      const d = dist();
      const [a, b] = [...pointers.values()];
      if (pinch) zoomAt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, d / pinch);
      pinch = d;
      moved = true;
      return;
    }
    const dx = e.offsetX - px, dy = e.offsetY - py;
    if (Math.abs(dx) + Math.abs(dy) > 1) moved = true;
    v = { ...v, tx: v.tx + dx, ty: v.ty + dy };
    tip = null;
    redraw();
  }
  function up(e) {
    pointers.delete(e.pointerId);
    pinch = 0;
    if (!moved && pointers.size === 0) {
      const [wx, wy] = toWorld(v, e.offsetX, e.offsetY);
      const i = hitTest(grid, map.x, map.y, wx, wy, 8 / v.scale);
      if (i >= 0) onselect?.(i);
    }
  }
  function hover(sx, sy) {
    const [wx, wy] = toWorld(v, sx, sy);
    const i = hitTest(grid, map.x, map.y, wx, wy, 7 / v.scale);
    if (i !== (tip?.i ?? -1)) redraw();
    // the tooltip's anchor slides along it with the pointer, so it never runs off either side
    tip = i >= 0 ? { i, x: sx, y: sy, along: (-100 * sx) / width } : null;
    canvas.style.cursor = i >= 0 ? 'pointer' : 'grab';
  }
  function leave() {
    if (tip) redraw();
    tip = null;
  }
  function wheel(e) {
    e.preventDefault();
    cancelAnimationFrame(anim);
    zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0015)));
  }
  const showAll = () => { const t = { ...v }; fitAll(); const target = v; v = t; flyTo(target); };
  const setFocus = () => { focusKey = hoverKey || pinned; redraw(); };
  const previewColour = (k) => { hoverKey = k; setFocus(); };
  const pinColour = (k) => { pinned = pinned === k ? '' : k; setFocus(); };

  onMount(() => {
    ctx = canvas.getContext('2d');
    readColors();
    const ro = new ResizeObserver(() => {
      const box = wrap.getBoundingClientRect();
      const first = !width;
      width = box.width; height = box.height; dpr = window.devicePixelRatio || 1;
      canvas.width = Math.round(width * dpr); canvas.height = Math.round(height * dpr);
      if (first) { fitAll(); frameSelection(); }
      draw();
      ready = true;
    });
    ro.observe(wrap);
    const mo = new MutationObserver(() => { readColors(); redraw(); });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    canvas.addEventListener('wheel', wheel, { passive: false });
    return () => { ro.disconnect(); mo.disconnect(); cancelAnimationFrame(frame); cancelAnimationFrame(anim); clearTimeout(settle); };
  });
</script>

<div class="map-view" bind:this={wrap}>
  <p class="sr-only">Map of every card, placed so that cards the model finds similar sit close together. Clicking a dot selects a card; the search box does the same from the keyboard.</p>
  <canvas bind:this={shade} class="shade" aria-hidden="true"></canvas>
  <canvas bind:this={canvas} class:ready aria-hidden="true" onpointerdown={down} onpointermove={move} onpointerup={up} onpointercancel={up} onpointerleave={leave}></canvas>
  {#if tip}
    <div class="tip" class:below={tip.y < 48} style:left="{tip.x}px" style:top="{tip.y}px" style:--along="{tip.along}%">
      <i style:background={palette[keys[tip.i]]}></i><span>{names.names[tip.i]}</span>
    </div>
  {/if}
  <div class="controls" bind:this={controls}>
    <div class="zoom">
      <button type="button" aria-label="Zoom in" onclick={() => zoomAt(width / 2, height / 2, 1.8, true)}><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 3v10M3 8h10" /></svg></button>
      <button type="button" aria-label="Zoom out" onclick={() => zoomAt(width / 2, height / 2, 1 / 1.8, true)}><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M3 8h10" /></svg></button>
    </div>
    <button type="button" aria-label="Show the whole map" onclick={showAll}><svg viewBox="0 0 16 16" aria-hidden="true"><path d="M2.5 6V2.5H6M10 2.5h3.5V6M13.5 10v3.5H10M6 13.5H2.5V10" /></svg></button>
  </div>
  {#if selected >= 0 && map.x[selected] !== TEXTLESS}
    <div class="caption picked" bind:this={caption}>
      <i style:background={palette[keys[selected]]}></i><span class="name">{names.names[selected]}</span>
      <span>and its {neighbours.length} closest</span>
      <button type="button" class="link" onclick={showAll}>Whole map</button>
    </div>
  {:else if hint}
    <div class="caption" bind:this={caption}>
      <p><strong>{placed.toLocaleString('en')} cards</strong>, placed by what they do. Nearby dots have similar effects: hover for a name, click one to see its matches.</p>
      <button type="button" class="close" aria-label="Hide this note" onclick={() => { hint = false; redraw(); }}>×</button>
    </div>
  {/if}
  <div class="legend" role="group" aria-label="Highlight one colour identity" bind:this={legend}>
    {#each LEGEND as [k, label] (k)}
      <button type="button" title={pinned === k ? 'Show every colour' : `Highlight ${label.toLowerCase()} cards`} aria-pressed={pinned === k} class:faded={pinned && pinned !== k} onpointerenter={() => previewColour(k)} onpointerleave={() => previewColour('')} onclick={() => pinColour(k)}>
        <span class="swatch" style:background={palette[k]} style:color={inkOn(palette[k])}>{#if k !== 'm'}<i class="ms ms-{k}" aria-hidden="true"></i>{/if}</span>
        {#if k === 'm'}<span class="mlabel">{label}</span>{:else}<span class="sr-only">{label}</span>{/if}
      </button>
    {/each}
  </div>
</div>

<style>
  .map-view { position: relative; width: 100%; height: 100%; min-height: 320px; overflow: hidden; background: var(--surface); border: 1px solid var(--line); border-radius: var(--radius); touch-action: none; }
  canvas { position: relative; display: block; width: 100%; height: 100%; cursor: grab; opacity: 0; transition: opacity 0.35s ease-out; }
  canvas.ready { opacity: 1; }
  .shade { position: absolute; left: 0; top: 0; width: auto; height: auto; opacity: 0; transform-origin: 0 0; pointer-events: none; transition: none; }
  canvas:active { cursor: grabbing; }
  .tip { position: absolute; display: flex; align-items: center; gap: 6px; max-width: min(260px, 80%); transform: translate(var(--along), calc(-100% - 14px)); padding: 5px 9px; border-radius: 6px; background: var(--ink); color: var(--surface); font-size: 0.8rem; font-weight: 500; line-height: 1.2; white-space: nowrap; pointer-events: none; box-shadow: 0 4px 14px rgb(0 0 0 / 0.18); }
  .tip.below { transform: translate(var(--along), 16px); }
  .tip span { overflow: hidden; text-overflow: ellipsis; }
  .tip i { flex: none; width: 8px; height: 8px; border-radius: 50%; box-shadow: 0 0 0 1.5px var(--surface); }
  .controls { position: absolute; right: 10px; top: 10px; display: grid; gap: 6px; }
  .zoom { display: grid; border: 1px solid var(--line); border-radius: var(--radius); background: var(--surface); overflow: hidden; }
  .zoom button + button { border-top: 1px solid var(--line); }
  .controls button { display: grid; place-items: center; width: 32px; height: 32px; padding: 0; border: 0; background: var(--surface); color: var(--muted); }
  .controls > button { border: 1px solid var(--line); border-radius: var(--radius); }
  .controls button:hover { color: var(--ink); background: color-mix(in srgb, var(--ink) 6%, var(--surface)); }
  .controls svg { width: 15px; height: 15px; fill: none; stroke: currentColor; stroke-width: 1.6; stroke-linecap: round; stroke-linejoin: round; }
  .legend { position: absolute; left: 10px; bottom: 10px; display: flex; flex-wrap: wrap; align-items: center; gap: 1px; padding: 3px; border-radius: 999px; background: color-mix(in srgb, var(--surface) 92%, transparent); box-shadow: 0 0 0 1px var(--line); font-size: 0.75rem; color: var(--muted); }
  .legend button { display: flex; align-items: center; gap: 5px; height: 26px; padding: 0 5px; border: 0; border-radius: 999px; background: none; color: inherit; }
  .legend button:hover { background: var(--sunken); color: var(--ink); }
  .legend button[aria-pressed='true'] { background: var(--sunken); box-shadow: inset 0 0 0 1.5px var(--ink); color: var(--ink); }
  .legend button.faded .swatch { opacity: 0.35; }
  .legend .mlabel { padding-right: 4px; }
  .caption { position: absolute; left: 10px; top: 10px; max-width: min(340px, calc(100% - 64px)); display: flex; align-items: flex-start; gap: 6px; padding: 8px 6px 8px 12px; border-radius: var(--radius); background: color-mix(in srgb, var(--surface) 92%, transparent); box-shadow: 0 0 0 1px var(--line), 0 4px 16px rgb(0 0 0 / 0.08); font-size: 0.8rem; line-height: 1.4; color: var(--muted); }
  .caption p { margin: 0; }
  .caption strong { color: var(--ink); font-weight: 650; }
  .close { flex: none; width: 24px; height: 24px; margin: -3px 0; padding: 0; border: 0; border-radius: 4px; background: none; font-size: 1.1rem; line-height: 1; color: var(--muted); }
  .close:hover { color: var(--ink); background: var(--sunken); }
  .picked { max-width: calc(100% - 64px); align-items: center; gap: 7px; padding: 6px 8px 6px 11px; white-space: nowrap; }
  .picked i { flex: none; width: 9px; height: 9px; border-radius: 50%; box-shadow: 0 0 0 2px var(--surface), 0 0 0 3.5px var(--ink); margin-right: 2px; }
  .picked .name { min-width: 0; max-width: 16em; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; color: var(--ink); font-weight: 650; }
  .link { padding: 1px 5px; border: 0; border-radius: 4px; background: none; font-size: inherit; font-weight: 550; color: var(--ink); text-decoration: underline; text-decoration-color: var(--line); text-underline-offset: 0.2em; }
  .link:hover { text-decoration-color: currentColor; }
  /* narrow screens name the picked card in the page's own bar over the map */
  @media (max-width: 900px) { .picked { display: none; } }
  .swatch { display: grid; place-items: center; width: 16px; height: 16px; border-radius: 50%; }
  .swatch .ms { font-size: 9px; line-height: 1; }
  @media (prefers-reduced-motion: reduce) { canvas { transition: none; } }
</style>

<script>
  import { onMount } from 'svelte';
  import { buildGrid, colorKey, hitTest, TEXTLESS, toScreen, toWorld } from './map.js';

  // map: decodeMap() result; regions: [{x, y, label, n}]; names: names.json; selected: index or -1
  let { map, regions = [], names, selected = -1, neighbours = [], onselect } = $props();

  let wrap, canvas, ctx;
  let width = 0, height = 0, dpr = 1;
  let v = { scale: 1, tx: 0, ty: 0 };
  let fit = 1;
  let tip = $state(null);
  let frame = 0;
  let anim = 0;
  const grid = $derived(buildGrid(map.x, map.y, 1024));
  // point indices bucketed by colour so each colour is one fillStyle pass
  const buckets = $derived.by(() => {
    const out = {};
    for (let i = 0; i < map.n; i += 1) {
      if (map.x[i] === TEXTLESS) continue;
      (out[colorKey(map.color[i])] ??= []).push(i);
    }
    return out;
  });

  let colors = {};
  function readColors() {
    const css = getComputedStyle(document.documentElement);
    for (const k of ['w', 'u', 'b', 'r', 'g', 'm', 'c']) colors[k] = css.getPropertyValue(`--wubrg-${k}`).trim();
    for (const k of ['ink', 'muted', 'accent', 'surface', 'bg']) colors[k] = css.getPropertyValue(`--${k}`).trim();
  }

  function fitAll() {
    fit = Math.min(width, height) / 66000;
    v = { scale: fit, tx: width / 2, ty: height / 2 };
  }

  function draw() {
    frame = 0;
    if (!ctx || !width) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const zoom = v.scale / fit;
    const r = Math.min(4, 0.9 + 0.45 * Math.sqrt(zoom));
    const dim = selected >= 0;
    ctx.globalAlpha = dim ? 0.35 : 0.8;
    for (const [k, list] of Object.entries(buckets)) {
      ctx.fillStyle = colors[k];
      for (const i of list) {
        const sx = v.tx + map.x[i] * v.scale, sy = v.ty + map.y[i] * v.scale;
        if (sx < -4 || sy < -4 || sx > width + 4 || sy > height + 4) continue;
        ctx.fillRect(sx - r / 2, sy - r / 2, r, r);
      }
    }
    ctx.globalAlpha = 1;
    if (dim && map.x[selected] !== TEXTLESS) {
      const [cx, cy] = toScreen(v, map.x[selected], map.y[selected]);
      ctx.strokeStyle = colors.accent;
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.45;
      ctx.beginPath();
      for (const i of neighbours) {
        const [sx, sy] = toScreen(v, map.x[i], map.y[i]);
        ctx.moveTo(cx, cy);
        ctx.lineTo(sx, sy);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
      for (const i of neighbours) {
        const [sx, sy] = toScreen(v, map.x[i], map.y[i]);
        ctx.beginPath();
        ctx.arc(sx, sy, 4, 0, Math.PI * 2);
        ctx.fillStyle = colors[colorKey(map.color[i])];
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = colors.surface;
        ctx.stroke();
      }
      ctx.beginPath();
      ctx.arc(cx, cy, 7, 0, Math.PI * 2);
      ctx.fillStyle = colors.accent;
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = colors.surface;
      ctx.stroke();
    }
    drawLabels(zoom);
  }

  // larger regions first; a label is skipped when it would overlap one already drawn
  function drawLabels(zoom) {
    const size = 11 + Math.min(3, zoom / 3);
    ctx.font = `600 ${size}px 'Inter Variable', system-ui, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    const placed = [];
    for (const reg of [...regions].sort((a, b) => b.n - a.n)) {
      const [sx, sy] = toScreen(v, reg.x, reg.y);
      const w = ctx.measureText(reg.label).width + 10, h = size + 6;
      const box = [sx - w / 2, sy - h / 2, sx + w / 2, sy + h / 2];
      if (box[0] < 2 || box[1] < 2 || box[2] > width - 2 || box[3] > height - 2) continue;
      if (placed.some((p) => box[0] < p[2] && box[2] > p[0] && box[1] < p[3] && box[3] > p[1])) continue;
      placed.push(box);
      ctx.lineWidth = 4;
      ctx.strokeStyle = colors.bg;
      ctx.globalAlpha = 0.9;
      ctx.strokeText(reg.label, sx, sy);
      ctx.globalAlpha = 1;
      ctx.fillStyle = colors.ink;
      ctx.fillText(reg.label, sx, sy);
    }
  }

  const redraw = () => { if (!frame) frame = requestAnimationFrame(draw); };

  function zoomAt(sx, sy, factor) {
    const scale = Math.min(fit * 60, Math.max(fit * 0.8, v.scale * factor));
    const [wx, wy] = toWorld(v, sx, sy);
    v = { scale, tx: sx - wx * scale, ty: sy - wy * scale };
    redraw();
  }

  // ease the view to frame the selected card and its neighbours
  function flyTo(target) {
    cancelAnimationFrame(anim);
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) { v = target; draw(); return; }
    const from = { ...v }, t0 = performance.now();
    const step = (t) => {
      const k = Math.min(1, (t - t0) / 380), e = 1 - (1 - k) ** 3;
      v = { scale: from.scale + (target.scale - from.scale) * e, tx: from.tx + (target.tx - from.tx) * e, ty: from.ty + (target.ty - from.ty) * e };
      draw();
      if (k < 1) anim = requestAnimationFrame(step);
    };
    anim = requestAnimationFrame(step);
  }

  function frameSelection() {
    if (selected < 0 || map.x[selected] === TEXTLESS || !width) return;
    const pts = [selected, ...neighbours].filter((i) => map.x[i] !== TEXTLESS);
    const xs = pts.map((i) => map.x[i]), ys = pts.map((i) => map.y[i]);
    const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
    const span = Math.max(x1 - x0, y1 - y0, 1500);
    const scale = Math.min(fit * 3, Math.max(fit * 1.5, (Math.min(width, height) * 0.7) / span));
    const cx = map.x[selected], cy = map.y[selected];
    flyTo({ scale, tx: width / 2 - cx * scale, ty: height / 2 - cy * scale });
  }

  $effect(() => { void selected; void neighbours; frameSelection(); redraw(); });

  // pointer handling: drag to pan, two pointers to pinch, a click without movement selects
  const pointers = new Map();
  let moved = false, pinch = 0;
  function down(e) {
    canvas.setPointerCapture(e.pointerId);
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
    tip = i >= 0 ? { i, x: sx, y: sy } : null;
    canvas.style.cursor = i >= 0 ? 'pointer' : 'grab';
  }
  function wheel(e) {
    e.preventDefault();
    zoomAt(e.offsetX, e.offsetY, Math.exp(-e.deltaY * (e.deltaMode ? 0.05 : 0.0015)));
  }

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
    });
    ro.observe(wrap);
    const mo = new MutationObserver(() => { readColors(); redraw(); });
    mo.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    canvas.addEventListener('wheel', wheel, { passive: false });
    return () => { ro.disconnect(); mo.disconnect(); cancelAnimationFrame(frame); cancelAnimationFrame(anim); };
  });
</script>

<div class="atlas" bind:this={wrap}>
  <p class="sr-only">Map of every card, placed so that cards the model finds similar sit close together. Clicking a dot selects a card; the search box does the same from the keyboard.</p>
  <canvas bind:this={canvas} aria-hidden="true" onpointerdown={down} onpointermove={move} onpointerup={up} onpointercancel={up} onpointerleave={() => (tip = null)}></canvas>
  {#if tip}<div class="tip" style:left="{tip.x}px" style:top="{tip.y}px">{names.names[tip.i]}</div>{/if}
  <div class="controls">
    <button type="button" aria-label="Zoom in" onclick={() => zoomAt(width / 2, height / 2, 1.6)}>+</button>
    <button type="button" aria-label="Zoom out" onclick={() => zoomAt(width / 2, height / 2, 1 / 1.6)}>−</button>
    <button type="button" aria-label="Show the whole map" onclick={() => { const t = { ...v }; fitAll(); const target = v; v = t; flyTo(target); }}>⤢</button>
  </div>
  <ul class="legend" aria-label="Colour identity">
    {#each [['w', 'White'], ['u', 'Blue'], ['b', 'Black'], ['r', 'Red'], ['g', 'Green'], ['m', 'Multi'], ['c', 'Colourless']] as [k, label]}
      <li><span style:background="var(--wubrg-{k})"></span>{label}</li>
    {/each}
  </ul>
</div>

<style>
  .atlas { position: relative; width: 100%; height: 100%; min-height: 320px; overflow: hidden; background: var(--sunken); border-radius: var(--radius); touch-action: none; }
  canvas { width: 100%; height: 100%; cursor: grab; }
  canvas:active { cursor: grabbing; }
  .tip { position: absolute; transform: translate(-50%, calc(-100% - 10px)); padding: 4px 8px; border-radius: 6px; background: var(--ink); color: var(--bg); font-size: 0.8rem; white-space: nowrap; pointer-events: none; }
  .controls { position: absolute; right: 10px; top: 10px; display: grid; gap: 4px; }
  .controls button { width: 32px; height: 32px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); font-size: 1.05rem; line-height: 1; }
  .controls button:hover { border-color: var(--accent); }
  .legend { position: absolute; left: 10px; bottom: 8px; display: flex; flex-wrap: wrap; gap: 4px 10px; margin: 0; padding: 5px 9px; list-style: none; border-radius: 8px; background: color-mix(in srgb, var(--surface) 85%, transparent); font-size: 0.72rem; color: var(--muted); }
  .legend li { display: flex; align-items: center; gap: 4px; }
  .legend span { width: 8px; height: 8px; border-radius: 2px; }
</style>

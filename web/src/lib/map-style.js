// Drawing parameters for the card map.

// Dot colours by colour identity, one set per theme, checked against the map surface
// (--surface) with the dataviz palette validator on all pairs: protan/deutan ΔE >= 10,
// normal-vision ΔE >= 17 (OKLab x100). White in the light theme is a pale mark by nature (2.2:1).
export const PALETTE = {
  light: { w: '#c6af00', u: '#366096', b: '#5b205a', r: '#ba002f', g: '#1d956d', m: '#b46d09', c: '#a0a3b3' },
  dark: { w: '#f2e5ad', u: '#58a7ff', b: '#b85eb7', r: '#ee5a50', g: '#29c38f', m: '#ea9127', c: '#8a9fa0' }
};

export const LEGEND = [['w', 'White'], ['u', 'Blue'], ['b', 'Black'], ['r', 'Red'], ['g', 'Green'], ['c', 'Colourless'], ['m', 'Multicolour']];

// Half-width of the laid-out map in world units (coordinates are int16).
export const EXTENT = 32000;

// Dot radius in CSS pixels: about a third of the mean gap between cards at this scale,
// growing with the square root of zoom so dots do not swell into blobs when zoomed in.
export function pointRadius(scale, fit, n) {
  const gap = 2 * EXTENT * fit * Math.sqrt(Math.PI / 4 / Math.max(1, n));
  return Math.min(5, Math.max(1.1, 0.4 * gap * Math.sqrt(scale / fit)));
}

// Card density on a size x size grid over [-span, span] in both axes, box-blurred three times (close to a
// Gaussian) and scaled to 0..1. Drawn faintly under the dots so clusters read at a glance.
export function densityGrid(x, y, size, span, skip) {
  let a = new Float32Array(size * size), b = new Float32Array(size * size);
  for (let i = 0; i < x.length; i += 1) {
    if (x[i] === skip) continue;
    const gx = Math.floor(((x[i] + span) / (2 * span)) * size), gy = Math.floor(((y[i] + span) / (2 * span)) * size);
    if (gx >= 0 && gy >= 0 && gx < size && gy < size) a[gy * size + gx] += 1;
  }
  const rad = Math.max(1, Math.round(size / 96));
  for (let pass = 0; pass < 3; pass += 1) {
    for (const horizontal of [true, false]) {
      for (let row = 0; row < size; row += 1) {
        let sum = 0;
        const at = (k) => (horizontal ? row * size + k : k * size + row);
        for (let k = -rad; k <= rad; k += 1) if (k >= 0 && k < size) sum += a[at(k)];
        for (let k = 0; k < size; k += 1) {
          b[at(k)] = sum / (2 * rad + 1);
          if (k + rad + 1 < size) sum += a[at(k + rad + 1)];
          if (k - rad >= 0) sum -= a[at(k - rad)];
        }
      }
      [a, b] = [b, a];
    }
  }
  let max = 0;
  for (const d of a) if (d > max) max = d;
  if (max > 0) for (let i = 0; i < a.length; i += 1) a[i] /= max;
  return a;
}

// Glyph colour for a legend swatch: dark on light swatches, white on dark ones.
export function inkOn(hex) {
  const lin = [1, 3, 5].map((k) => { const c = parseInt(hex.slice(k, k + 2), 16) / 255; return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4; });
  return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] > 0.3 ? '#15171c' : '#ffffff';
}

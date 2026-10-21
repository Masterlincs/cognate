export const fixed = (x, d = 3) => x.toFixed(d);
export const delta = (x, d = 3) => (Math.abs(x) < 0.5 * 10 ** -d ? '±' : x > 0 ? '+' : '−') + Math.abs(x).toFixed(d);
export const scaleLinear = ([d0, d1], [r0, r1]) => (x) => r0 + ((x - d0) / (d1 - d0)) * (r1 - r0);
export const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

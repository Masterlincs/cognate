// Preload the two Latin body/heading fonts so text does not reflow when they arrive.
export const handle = ({ event, resolve }) =>
  resolve(event, {
    preload: ({ type, path }) =>
      type === 'js' || type === 'css' || (type === 'font' && /(inter|fraunces)-latin-(wght|opsz)-normal/.test(path))
  });

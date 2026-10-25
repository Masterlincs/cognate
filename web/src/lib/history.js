// Back/Forward restoration is kept per history entry, not per URL, so two visits to the same card
// keep their own scroll position and list length. Each entry the explorer makes carries an id in
// page.state; an entry made before the page ran (first load, reload, a plain link) has none and is
// known by its query instead.

// ids from before a reload stay in history, so each page load numbers its entries under its own prefix
const session = Math.random().toString(36).slice(2, 8);
let count = 0;

export const newEntryId = () => `${session}.${++count}`;

// an id never starts with "?" and a query is "" or starts with "?", so the two kinds of key never clash
export const entryKey = (id, search) => id ?? search;

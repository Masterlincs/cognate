// Scryfall's bulk data picks an arbitrary printing for each card, and for some of the landing-page
// cards that is a crossover (Universes Beyond) printing. These are plain core-set printings instead:
// the earliest black-border, non-promo, high-resolution English printing from a core or expansion set
// (a masters, commander or draft-innovation set for Swords to Plowshares, which has none).
// Keys are Oracle IDs. Display only; the release data is unchanged.
export const PLAIN_PRINTINGS = {
  '4457ed35-7c10-48c8-9776-456485fdf070': 'https://cards.scryfall.io/normal/front/4/3/435589bb-27c6-4a6d-9d63-394d5092b9d8.jpg?1783942371', // Lightning Bolt
  '5748ebf1-24e3-499d-ab7c-c2cebd462a24': 'https://cards.scryfall.io/normal/front/7/c/7cc4adb1-85af-40ba-bd36-ccd2dc1cb227.jpg?1783944679', // Giant Growth
  'b1544f21-7e98-461b-aed5-e748b0168c52': 'https://cards.scryfall.io/normal/front/b/e/be0bda98-e3f7-4a9e-b05d-5f8529e762ab.jpg?1783939362', // Swords to Plowshares
  '8539f295-5d58-4436-a73a-b9277c4c7795': 'https://cards.scryfall.io/normal/front/3/3/336c1d89-7d7e-42f4-a1e1-55f85b3c47e1.jpg?1783944660', // Rampant Growth
  '68954295-54e3-4303-a6bc-fc4547a4e3a3': 'https://cards.scryfall.io/normal/front/d/a/da7bf82d-437a-4c28-98a4-4d092ba8d063.jpg?1783943929', // Llanowar Elves
  '9243bd95-5467-490d-9bd2-4d5ce1ebc589': 'https://cards.scryfall.io/normal/front/c/5/c5b10c5f-261f-4301-b675-d1f52b859360.jpg?1783943699', // Ghosts of the Innocent
};

// How a card's other face is shown in the grid: 'back' = its printed back image (transform,
// modal double-faced), 'rotate' = the same image turned over (flip cards), null = one face.
export function flipKind(card) {
  if (typeof card.image_back === 'string' && card.image_back) return 'back';
  return card.layout === 'flip' ? 'rotate' : null;
}

export function faceName(card, flipped) {
  return (flipped ? card.faces[1]?.name : card.faces[0]?.name) || card.name;
}

// another size of a Scryfall card image: small, normal, large or art_crop
export const cardImage = (url, version) => url?.replace(/\/(small|normal|large|png|art_crop|border_crop)\//, `/${version}/`) ?? null;

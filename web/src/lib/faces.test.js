import { describe, expect, it } from 'vitest';
import { faceName, flipKind } from './faces.js';

const two = [{ name: 'Front' }, { name: 'Back' }];
describe('card faces', () => {
  it('double-faced cards show their printed back', () => {
    expect(flipKind({ layout: 'transform', image_back: 'b.jpg', faces: two })).toBe('back');
    expect(flipKind({ layout: 'modal_dfc', image_back: 'b.jpg', faces: two })).toBe('back');
  });
  it('flip cards turn the same image over', () => expect(flipKind({ layout: 'flip', image_back: null, faces: two })).toBe('rotate'));
  it('adventure, split, normal and old-bundle cards do not flip', () => {
    for (const layout of ['adventure', 'split', 'normal']) expect(flipKind({ layout, image_back: null, faces: two })).toBeNull();
    expect(flipKind({ layout: 'transform', faces: two })).toBeNull();
  });
  it('names the face that is showing', () => {
    const card = { name: 'Front // Back', layout: 'transform', image_back: 'b.jpg', faces: two };
    expect(faceName(card, false)).toBe('Front');
    expect(faceName(card, true)).toBe('Back');
    expect(faceName({ name: 'Solo', faces: [] }, false)).toBe('Solo');
  });
});

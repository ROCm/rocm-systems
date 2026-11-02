// Portals cancel the desktop density zoom, so their CSS width must use the
// anchor's viewport width rather than Autocomplete's unscaled clientWidth.
export const matchAnchorWidth = {
  name: 'matchAnchorWidth', enabled: true, phase: 'beforeRead',
  fn: ({ state }) => {
    const width = state.elements.reference.getBoundingClientRect().width;
    state.elements.popper.style.width = `${width}px`;
    state.rects.popper.width = width;
  },
};

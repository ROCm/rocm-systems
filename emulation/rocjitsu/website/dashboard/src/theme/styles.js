export const detailActionStyles = {
  justifyContent: 'flex-start',
  textAlign: 'left',
  borderRadius: 1,
  px: 0.5,
  py: 0.25,
  '&:hover': { backgroundColor: 'action.hover' },
  '&:focus-visible': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: 1 },
};

export const visuallyHiddenStyles = {
  position: 'absolute',
  width: '1px',
  height: '1px',
  overflow: 'hidden',
  clip: 'rect(0 0 0 0)',
  clipPath: 'inset(50%)',
  whiteSpace: 'nowrap',
};

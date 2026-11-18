// 44px-tall outlined search field: MUI's small-size label offset assumes a 40px field, so the
// resting label (shown in place of a placeholder) is moved to the middle of the taller box.
export const tallSearchFieldSx = {
  '& .MuiInputBase-root': { minHeight: 44 },
  '& .MuiInputBase-input': { py: '10px' },
  '& .MuiInputLabel-root:not(.MuiInputLabel-shrink)': { transform: 'translate(14px, 12px) scale(1)' },
};

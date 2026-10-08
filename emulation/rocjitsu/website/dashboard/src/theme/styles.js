import { sansFont } from './tokens';

// App owns attribution text and links; this keeps its three-line footer on v10 tokens.
export const dashboardFooterStyles = {
  borderTop: 1,
  borderColor: 'divider',
  bgcolor: 'background.default',
  color: 'text.secondary',
  px: { xs: 2, md: 3 },
  py: 2.5,
  fontFamily: sansFont,
  fontSize: 12,
  lineHeight: '20px',
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

import { textStyles } from '../../theme/typography';
import { cloneElement, useId, useState } from 'react';
import { Box, Popper } from '@mui/material';
import { alpha } from '@mui/material/styles';

// Match chart tooltips without coupling ordinary controls to ECharts.
export default function HoverTooltip({ title, children }) {
  const id = useId();
  const [anchor, setAnchor] = useState(null);
  const open = Boolean(anchor && title);
  const anchorToPointer = (event) => {
    const { clientX, clientY, currentTarget } = event;
    setAnchor({
      contextElement: currentTarget,
      getBoundingClientRect: () => new DOMRect(clientX, clientY, 0, 0),
    });
  };
  return <>
    <Box component="span" sx={{ display: 'inline-flex' }}
      onMouseEnter={anchorToPointer} onMouseMove={anchorToPointer} onMouseLeave={() => setAnchor(null)}
      onFocus={(event) => setAnchor(event.currentTarget)} onBlur={() => setAnchor(null)}
      onKeyDown={(event) => { if (event.key === 'Escape') setAnchor(null); }}>
      {cloneElement(children, { 'aria-describedby': open ? [children.props['aria-describedby'], id].filter(Boolean).join(' ') : children.props['aria-describedby'] })}
    </Box>
    <Popper role="presentation" open={open} anchorEl={anchor} placement="bottom-start" modifiers={[
      { name: 'offset', options: { offset: anchor?.contextElement ? [12, 20] : [0, 4] } },
      { name: 'preventOverflow', options: { padding: 8 } },
      { name: 'flip', options: { padding: 8 } },
    ]} sx={{ zIndex: (theme) => theme.zIndex.tooltip, pointerEvents: 'none' }}>
      <Box id={id} role="tooltip" sx={{ bgcolor: 'background.paper', color: 'text.primary', border: 1, borderColor: 'dashboard.border', borderRadius: '6px', padding: '6px 9px', ...textStyles.tooltip, maxWidth: 'min(340px, calc(100vw - 16px))', overflowWrap: 'anywhere', boxShadow: (theme) => `0 4px 16px ${alpha(theme.palette.dashboard.shadowColor, 0.18)}` }}>{title}</Box>
    </Popper>
  </>;
}

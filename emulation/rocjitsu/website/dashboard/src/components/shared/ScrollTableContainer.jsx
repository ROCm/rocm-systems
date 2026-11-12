import { useEffect, useRef } from 'react';
import { TableContainer } from '@mui/material';

// Scrollable table area whose vertical scrollbar starts below the sticky header,
// so the track never runs beside the header row. The header height is measured,
// because tables differ in header density and wrapping.
export default function ScrollTableContainer({ sx, ...props }) {
  const ref = useRef(null);
  useEffect(() => {
    const node = ref.current;
    const head = node?.querySelector('thead');
    if (!head) return undefined;
    const measure = () => node.style.setProperty('--table-head-height', `${head.offsetHeight}px`);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(head);
    return () => observer.disconnect();
  }, []);
  return <TableContainer ref={ref} {...props} sx={[
    (theme) => ({
      maxWidth: '100%', overflow: 'auto', border: `1px solid ${theme.palette.dashboard.border}`, borderRadius: `${theme.shape.borderRadius}px`,
      // Header colour continues under the scrollbar gutter beside the header row.
      background: `linear-gradient(${theme.palette.dashboard.tableHeader}, ${theme.palette.dashboard.tableHeader}) top / 100% var(--table-head-height, 0px) no-repeat`,
      '&::-webkit-scrollbar-track': { marginTop: 'var(--table-head-height, 0px)', background: 'transparent' },
    }),
    ...(Array.isArray(sx) ? sx : [sx]),
  ]} />;
}

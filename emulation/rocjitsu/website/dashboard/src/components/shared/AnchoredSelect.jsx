import { useLayoutEffect, useRef, useState } from 'react';
import { Select } from '@mui/material';

// MUI Select defaults to a minimum unscaled width, letting long options enlarge
// the menu. Keep both edges on the visible field, including after resize.
export default function AnchoredSelect({ MenuProps = {}, ...props }) {
  const anchor = useRef(null);
  const [width, setWidth] = useState();
  useLayoutEffect(() => {
    const node = anchor.current;
    const measure = () => setWidth(node.getBoundingClientRect().width);
    measure();
    const observer = new ResizeObserver(measure);
    observer.observe(node);
    window.addEventListener('resize', measure);
    return () => { observer.disconnect(); window.removeEventListener('resize', measure); };
  }, []);
  return <Select {...props} ref={anchor} MenuProps={{
    // Keep page width stable when opening a non-dialog dropdown.
    disableScrollLock: true,
    ...MenuProps,
    slotProps: {
      ...MenuProps.slotProps,
      paper: {
        ...MenuProps.slotProps?.paper,
        style: { ...MenuProps.slotProps?.paper?.style, width, minWidth: width },
      },
    },
  }} />;
}

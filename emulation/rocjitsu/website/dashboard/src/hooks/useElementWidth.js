import { useCallback, useState } from 'react';

// Tracks an element's rendered width; the callback ref attaches the observer.
export default function useElementWidth() {
  const [width, setWidth] = useState(0);
  const ref = useCallback((node) => {
    if (!node) return undefined;
    setWidth(node.getBoundingClientRect().width);
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(node);
    return () => observer.disconnect();
  }, []);
  return [ref, width];
}

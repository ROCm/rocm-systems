import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

// Density follows CSS viewport width, not physical pixels: equal browser windows
// keep equal text/layout sizes across displays. Browser zoom can still reflow.
const MIN_DASHBOARD_SCALE = 0.7392;
const BASELINE_VIEWPORT_WIDTH = 1920;
const VIEWPORT_FIT_FACTOR = 0.72;

function updateDashboardScale() {
  const scale = Math.max(MIN_DASHBOARD_SCALE, MIN_DASHBOARD_SCALE * window.innerWidth / BASELINE_VIEWPORT_WIDTH * VIEWPORT_FIT_FACTOR);
  document.documentElement.style.setProperty('--dashboard-scale', String(scale));
}
updateDashboardScale();
window.addEventListener('resize', updateDashboardScale);
if (import.meta.hot) import.meta.hot.dispose(() => window.removeEventListener('resize', updateDashboardScale));

createRoot(document.getElementById('root')).render(
  <StrictMode>
    <App />
  </StrictMode>,
);

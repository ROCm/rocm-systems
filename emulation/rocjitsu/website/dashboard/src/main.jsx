import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './index.css';

// Match the physical viewport width to the 1920px baseline. innerWidth is in
// CSS pixels; devicePixelRatio accounts for display scaling and browser zoom.
function updateDashboardScale() {
  const scale = Math.max(0.73920, 0.73920 * window.innerWidth * window.devicePixelRatio / 1920 * 0.72);
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

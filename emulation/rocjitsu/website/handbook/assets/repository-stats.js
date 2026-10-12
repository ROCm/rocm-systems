// Show GitHub repository counts in the header and mobile navigation.
// Loaded by MkDocs extra_javascript; failures preserve the build snapshot.
(() => {
  const cacheKey = 'rocjitsu-repository-stats';
  const maxAge = 15 * 60 * 1000;
  const labels = { stargazers_count: 'Stars', forks_count: 'Forks' };

  function validCounts(data) {
    return data && Object.keys(labels).every(
      key => Number.isSafeInteger(data[key]) && data[key] >= 0,
    );
  }

  function render(data) {
    document.querySelectorAll('[data-repository-stat]').forEach(element => {
      const key = element.dataset.repositoryStat;
      const count = data[key].toLocaleString('en-US');
      element.textContent = count;
      element.title = `${labels[key]}: ${count} · ROCm/rocm-systems`;
      element.setAttribute('aria-label', element.title);
    });
  }

  async function load() {
    try {
      const cached = JSON.parse(sessionStorage.getItem(cacheKey));
      if (cached && validCounts(cached.data) && Number.isFinite(cached.time)
          && Date.now() >= cached.time) {
        render(cached.data);
        if (Date.now() - cached.time < maxAge) return;
      }
    } catch {
      // Storage may be unavailable; fetching counts does not require it.
    }

    try {
      const response = await fetch('https://api.github.com/repos/ROCm/rocm-systems');
      if (!response.ok) return;
      const data = await response.json();
      if (!validCounts(data)) return;
      render(data);
      try {
        sessionStorage.setItem(cacheKey, JSON.stringify({ time: Date.now(), data: {
          stargazers_count: data.stargazers_count,
          forks_count: data.forks_count,
        } }));
      } catch {
        // Keep the displayed counts when storage is unavailable.
      }
    } catch {
      // Keep the build snapshot or cached counts when GitHub cannot be reached.
    }
  }

  return load();
})();

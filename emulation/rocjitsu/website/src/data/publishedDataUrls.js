function withTrailingSlash(value) {
  return value.endsWith('/') ? value : `${value}/`;
}

export function resolvePublishedDataUrls({
  dataBaseUrl = import.meta.env.VITE_DASHBOARD_DATA_BASE_URL,
  assetBaseUrl = import.meta.env.BASE_URL,
  baseURI = document.baseURI,
} = {}) {
  const configured = typeof dataBaseUrl === 'string' ? dataBaseUrl.trim() : '';
  const relative = configured || `${assetBaseUrl}data/`;
  const directory = new URL(withTrailingSlash(relative), baseURI);
  return {
    dataBaseUrl: directory.href,
    metadataUrl: new URL('metadata.json', directory).href,
    indexUrl: new URL('index.json', directory).href,
  };
}

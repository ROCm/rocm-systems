import { vi } from 'vitest';

// Only React's root event/selection surface; callers own browser APIs and cleanup.
export function createReactRootSurface() {
  const browser = new EventTarget();
  const document = new EventTarget();
  const container = new EventTarget();
  Object.assign(container, { nodeType: 1, tagName: 'DIV', namespaceURI: 'http://www.w3.org/1999/xhtml', ownerDocument: document });
  Object.assign(document, { nodeType: 9, body: container, activeElement: null, defaultView: browser });
  browser.HTMLElement = class {};
  browser.HTMLIFrameElement = class extends browser.HTMLElement {};
  browser.document = document;
  vi.stubGlobal('window', browser);
  vi.stubGlobal('document', document);
  vi.stubGlobal('IS_REACT_ACT_ENVIRONMENT', true);
  return { browser, document, container };
}

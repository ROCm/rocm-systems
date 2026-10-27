# Bundled IBM Plex fonts

This directory contains the dashboard's bundled IBM Plex WOFF2 fonts and their
license files. IBM Plex Sans is a variable font; the 400/500/600 declarations
share `ibm-plex-sans-400.woff2`. IBM Plex Mono has separate 400/500/600 faces.

`src/index.css` declares the six normal-weight faces with local relative `url()`
references so Vite processes them in development and production. No font CDN is
requested by the dashboard. `src/theme/theme.js` imports both OFL files with
`?url&no-inline`, exposing their emitted URLs as `theme.fontLicenses.sans` and
`theme.fontLicenses.mono`; delivery does not depend on `publicDir` being enabled.

The fonts are licensed under the SIL Open Font License, Version 1.1. See
`ibmplexsans-OFL.txt` and `ibmplexmono-OFL.txt`. Dashboard application licensing
and footer attribution are separate from these font licenses.

#!/usr/bin/env node

import {
  createReadStream,
  existsSync,
  realpath,
  realpathSync,
  stat,
} from 'node:fs';
import path from 'node:path';

function isDashboardDataDirectory(directory) {
  return existsSync(path.join(directory, 'metadata.json'))
    && existsSync(path.join(directory, 'index.json'));
}

export function resolveDashboardDataDirectory(input) {
  if (!input || typeof input !== 'string' || input.startsWith('-')) {
    throw new Error(
      'Usage: npm run dev:data -- <data-directory> [-- <vite args>]\n'
      + '   or: npm run preview:data -- <data-directory>',
    );
  }

  const resolved = path.resolve(input);
  const nested = path.join(resolved, 'data');
  if (isDashboardDataDirectory(resolved)) return resolved;
  if (isDashboardDataDirectory(nested)) return nested;
  throw new Error(
    `Expected a dashboard data directory with metadata.json and index.json: ${resolved}`,
  );
}

export function parseServeDataArgs(argv) {
  const args = argv.filter((arg) => arg !== '--');
  const preview = args.includes('--preview');
  const rest = args.filter((arg) => arg !== '--preview');
  const directory = rest[0];
  resolveDashboardDataDirectory(directory);
  return {
    directory,
    preview,
    viteArgs: rest.slice(1),
  };
}

export function createDashboardDataMiddleware(dataDirectory) {
  const root = realpathSync(dataDirectory);
  const prefix = '/data/';

  return function dashboardDataMiddleware(req, res, next) {
    const pathname = (req.url ?? '').split('?')[0];
    if (pathname !== '/data' && !pathname.startsWith(prefix)) {
      next();
      return;
    }

    let relative;
    try {
      relative = pathname === '/data' || pathname === '/data/'
        ? ''
        : decodeURIComponent(pathname.slice(prefix.length));
    } catch {
      res.statusCode = 400;
      res.end();
      return;
    }
    if (!relative || relative.endsWith('/')) {
      res.statusCode = 404;
      res.end();
      return;
    }

    const file = path.resolve(root, relative);
    const escaped = path.relative(root, file);
    if (!escaped || escaped.startsWith('..') || path.isAbsolute(escaped)) {
      res.statusCode = 400;
      res.end();
      return;
    }

    realpath(file, (resolveError, resolvedFile) => {
      if (resolveError) {
        res.statusCode = 404;
        res.end();
        return;
      }

      const resolvedRelative = path.relative(root, resolvedFile);
      if (!resolvedRelative
        || resolvedRelative.startsWith('..')
        || path.isAbsolute(resolvedRelative)) {
        res.statusCode = 400;
        res.end();
        return;
      }

      stat(resolvedFile, (error, info) => {
        if (error || !info.isFile()) {
          res.statusCode = 404;
          res.end();
          return;
        }
        if (path.extname(resolvedFile) === '.json') {
          res.setHeader('Content-Type', 'application/json; charset=utf-8');
        }
        res.setHeader('Cache-Control', 'no-store');
        createReadStream(resolvedFile).pipe(res);
      });
    });
  };
}

export function serveDashboardData(dataDirectory) {
  const middleware = createDashboardDataMiddleware(dataDirectory);
  return {
    name: 'dashboard-data-dir',
    configureServer(server) {
      server.middlewares.use(middleware);
    },
    configurePreviewServer(server) {
      server.middlewares.use(middleware);
    },
  };
}

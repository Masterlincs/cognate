# Card Atlas

A static SvelteKit site for the effect-similarity model. Two pages:

- `/` is the explorer: a map of every card, a search box, the selected card's nearest neighbours and a side-by-side compare.
- `/how-it-works` is the case study: the problem, the data, the training, a before-and-after against the pretrained model, the results and the known failures.

Everything is prerendered to plain files. There is no server code; the explorer fetches JSON and binary shards from `data/` at run time.

## Build

```sh
npm ci
npm test
npm run build      # writes build/
npm run preview    # serves build/ on http://localhost:4173
```

`npm run build` runs `vite build`, then `scripts/prune-build.mjs`, which keeps only the release named by `public/data/current.json` and its site data in `build/data`. Older releases stay in `public/data` for rollback but are not shipped.

## Data

The site reads three things from `public/data/` (all git-ignored except `current.json`):

| Path | Made by | Contents |
|---|---|---|
| `current.json` | hand / export | `{"release_id": ...}`, the release the site shows |
| `<release_id>/` | `uv run mtgemb export ...` | release bundle: card shards, neighbour shards, names, tags ([SCHEMA.md](SCHEMA.md)) |
| `site-<release_id>/` | `uv run mtgemb site-data --release <release_id>` | map coordinates (`map.bin`), region labels (`map.json`) and pretrained-model neighbours for the before-and-after |

The case-study numbers come from `src/lib/generated/metrics.json`, which is committed. Regenerate it from the run records with:

```sh
uv run mtgemb site-data --metrics-only
```

`site-data` refuses to overwrite an existing `site-<release_id>/` folder; a new release gets a new folder. It writes to `site-<release_id>.partial` and renames at the end; on Windows, stop `npm run dev` first, because the dev server's file watcher blocks the rename. If the site folder is missing, the explorer still works and the map and before-and-after say they are unavailable.

`npm run fixture` writes a small synthetic bundle and points `current.json` at it, for development without the real data. Fixture names and rules text are made up. `git checkout public/data/current.json` switches back. `npm run build` fails if `current.json` names a release that is not in `public/data`.

## Deploy

Copy `build/` to any static host. The site uses relative paths, so it also works below a path prefix. Pages are written as `how-it-works.html`, so the server must map extensionless URLs to `.html`. For nginx:

```nginx
server {
    listen 80;
    server_name example.org;
    root /var/www/card-atlas;

    location / {
        try_files $uri $uri.html $uri/ =404;
    }
    location /_app/immutable/ {
        add_header Cache-Control "public, max-age=31536000, immutable";
    }
    gzip on;
    gzip_types application/json application/javascript text/css image/svg+xml application/octet-stream;
}
```

Card images are loaded from Scryfall at view time and are not part of the build.

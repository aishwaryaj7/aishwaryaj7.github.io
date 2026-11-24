# CLAUDE.md

Personal portfolio site (https://aishwaryaj7.github.io) plus the source code of some of the projects it shows.

## Layout

- **`site/`** is the website: Astro 7, static output, no UI framework. `.github/workflows/deploy-portfolio.yml` builds it (`npm ci --omit=dev && npm run build` in `site/`) and publishes `site/dist` to GitHub Pages on every push to `main`.
- **`ai-portfolio/projects/*`** are independent Python projects (own `pyproject.toml`/`requirements.txt`, README). Not part of the site build. ListingAudit lives in its own repo (github.com/aishwaryaj7/listingaudit).
- Root `pyproject.toml`/`uv.lock` are left over from the old MkDocs setup and are not used by the site.

## Working on the site

```bash
cd site
npm install
npm run dev     # http://localhost:4321 (background: npx astro dev --background / npx astro dev stop)
npm run build   # must pass before handing work back
npm run data    # regenerate src/data/generated/*.json (see below)
```

- `build.format: 'file'` keeps old URLs working (`/blog/<slug>.html`, `/projects/<slug>.html`). Link internally without the extension (`/projects/doctract`).
- Fonts come from Astro's fonts API (`astro.config.mjs`). **One font family everywhere: Geist** (`--font-sans`). Geist Mono (`--font-mono`) is used only for code blocks and inline code in `src/styles/prose.css`.
- Theme: dark-first. Tokens live on `:root` (light) and `:root[data-theme='dark']` in `src/styles/global.css`; use tokens, not hard-coded colours. Topic colours `--t-agents|tools|rag|mlops|speech` are shared by the hero map and the project covers.

### Content

- `src/content/projects/*.mdx` and `src/content/blog/*.md(x)`; schema in `src/content.config.ts`.
  - Projects: `title`, `tagline`, `summary` (card + page lede), `description` (meta), `metrics` (featured card), `stack` ("Built with" tiles; first 4 show on the home card), `skills` (tiles), `demo`, `code`, `order`, `featured`.
  - Blog: `title`, `description`, `date`, `tags`, `draft`. Reading time is computed from the body.
- Project page structure (keep it short and visual): **Overview** (2–3 sentences, plus a real visual: screenshots via `Shots`, or a data chart) → **How it works** (`Architecture` with logos) → **Highlights** (`Highlights`, keywords only, no explanations) → **Results** only where real numbers exist (table + `ResultsSummary`). No "How to use", "Tech stack" or "Skills demonstrated" sections, no figure numbers, no legends.
- Components for MDX live in `src/components/project/`. `Logo.astro` maps keys to logos (simple-icons + vendored LobeHub SVGs in `src/assets/logos`, MIT); unknown names render as a text badge. Screenshots live in `src/assets/projects/<slug>/` and are optimised by `astro:assets`.
- Don't put claims on the site that the project code doesn't support.

### Generated data (`npm run data`, `scripts/build-data.mjs`)

Outputs are committed; the site build never calls these APIs.
- `embedding-map.json`: every paragraph of projects + posts embedded with all-MiniLM-L6-v2 (transformers.js) and projected with UMAP. **Re-run after adding or editing content** so the hero map stays current.
- `energy-prices.json`: one day of DE-LU day-ahead prices from api.energy-charts.info (CC BY 4.0, SMARD.de; credited on the Energy page and card).
- `waveform.json`: envelope + detected pauses of `ai-portfolio/projects/speech2text/data/test_audio.mp3` (needs `ffmpeg`).

The ML packages for this are devDependencies only; CI installs with `--omit=dev`.

## Home page

`src/pages/index.astro`: hero over `EmbeddingMap` (hover a dot to read the paragraph) → Selected work (`ProjectCard` with `Cover` art) → Writing → Career (`Trajectory`, newest first, horizontal with `ScrollRails` slider) → Contact. Career entries are in `src/data/site.ts`. No nav links in the header, and no resume/CV anywhere on the site (the owner doesn't want it published).

## Conventions

- Never commit or push; the owner reviews a local preview and commits and pushes. Give commands instead.
- Show a preview (dev server + screenshots) for every visible change and wait for approval.
- Writing style on the site: plain, short, no emojis, no marketing words.
- Commit messages are short and lowercase-ish, describing the change.

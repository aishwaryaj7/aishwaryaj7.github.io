# Aishwarya Jauhari Sharma · Portfolio

Source of [aishwaryaj7.github.io](https://aishwaryaj7.github.io): my machine learning and LLM projects, and the write-ups behind them.

## Projects

| Project | What it is | Code |
| --- | --- | --- |
| [ListingAudit](https://aishwaryaj7.github.io/projects/listingaudit) | LangGraph agent that audits shop listings against EU consumer law, with MCP tools and human review | [listingaudit](https://github.com/aishwaryaj7/listingaudit) |
| [Energy Price Forecasting](https://aishwaryaj7.github.io/projects/energy-price-forecasting) | Day-ahead electricity price forecasting for DE, FR and NL, tracked in MLflow and served on Cloud Run | [ai-portfolio/projects/energy_price_forecasting](ai-portfolio/projects/energy_price_forecasting) |
| [DocTract](https://aishwaryaj7.github.io/projects/doctract) | Local RAG over PDFs with pgvector and Mistral 7B | [ai-portfolio/projects/doctract](ai-portfolio/projects/doctract) |
| [Speech2Text](https://aishwaryaj7.github.io/projects/speech2text) | Audio transcription with HuBERT, split at natural pauses | [ai-portfolio/projects/speech2text](ai-portfolio/projects/speech2text) |
| [Doc Parser](https://aishwaryaj7.github.io/projects/doc-parser) | PDF to JSON or Markdown, with questions about tables | [ai-portfolio/projects/doc-parser](ai-portfolio/projects/doc-parser) |

Each project folder under `ai-portfolio/projects/` has its own README and dependencies.

## The site

Built with [Astro](https://astro.build) in [`site/`](site/) and deployed to GitHub Pages by
[`.github/workflows/deploy-portfolio.yml`](.github/workflows/deploy-portfolio.yml) on every push to `main`.

```bash
cd site
npm install
npm run dev      # http://localhost:4321
npm run build    # static site in site/dist
```

The hero background is a map of every paragraph on the site, embedded with all-MiniLM-L6-v2 and laid out with UMAP.
`npm run data` regenerates it, along with the price and waveform data behind the project visuals.

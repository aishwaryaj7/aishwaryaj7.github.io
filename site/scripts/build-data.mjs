// Generates the data behind the site's visuals. Run with `npm run data` after adding or editing
// posts/projects; the outputs are committed, so the site build itself stays fast and offline.
//
//   src/data/generated/embedding-map.json  every paragraph of my writing, embedded with
//                                          all-MiniLM-L6-v2 and projected to 2D with UMAP
//   src/data/generated/energy-prices.json  one real day of German day-ahead prices (Energy-Charts)
//   src/data/generated/waveform.json       amplitude envelope of the Speech2Text test recording
import fs from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const out = path.join(root, 'src/data/generated');
fs.mkdirSync(out, { recursive: true });
const only = process.argv[2]; // optional: embedding | energy | waveform

// Deterministic randomness so the map doesn't reshuffle on every run
function mulberry32(seed) {
  return () => {
    seed |= 0;
    seed = (seed + 0x6d2b79f5) | 0;
    let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/* ------------------------------------------------------------------ */
/* 1. Embedding map                                                    */
/* ------------------------------------------------------------------ */
const TOPICS = {
  agents: { label: 'Agents', slugs: ['listingaudit', 'langgraph-agents'] },
  tools: { label: 'Tool use · MCP', slugs: ['mcp-deep-dive'] },
  rag: { label: 'RAG · documents', slugs: ['doctract', 'doctract-rag-journey', 'doc-parser'] },
  mlops: { label: 'MLOps · forecasting', slugs: ['energy-price-forecasting', 'energy-price-forecasting-mlops'] },
  speech: { label: 'Speech', slugs: ['speech2text', 'speech2text-journey'] },
};
const topicOf = (slug) => Object.keys(TOPICS).find((t) => TOPICS[t].slugs.includes(slug));

const plain = (s) =>
  s
    .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/[*_`>#]/g, '')
    .replace(/\\/g, '')
    .replace(/\s+/g, ' ')
    .trim();

function paragraphs(body) {
  const text = body
    .replace(/```[\s\S]*?```/g, '\n\n')
    .replace(/^import .*$/gm, '')
    .replace(/^<[A-Z][\s\S]*?\/>\s*$/gm, '');
  const blocks = text.split(/\n\s*\n/);
  const result = [];
  for (const block of blocks) {
    const lines = block.trim().split('\n');
    if (!lines[0] || lines[0].startsWith('#') || lines[0].startsWith('|')) continue;
    // Lists: every item counts as its own paragraph
    const items = lines.every((l) => /^\s*([-*]|\d+\.)\s/.test(l) || /^\s{2,}/.test(l))
      ? lines.map((l) => l.replace(/^\s*([-*]|\d+\.)\s+/, ''))
      : [lines.join(' ')];
    for (const item of items) {
      const p = plain(item);
      if (p.split(' ').length >= 9) result.push(p);
    }
  }
  return result;
}

async function buildEmbeddingMap() {
  const { pipeline } = await import('@huggingface/transformers');
  const { UMAP } = await import('umap-js');

  const docs = [];
  const chunks = [];
  for (const kind of ['projects', 'blog']) {
    const dir = path.join(root, 'src/content', kind);
    for (const file of fs.readdirSync(dir).filter((f) => /\.mdx?$/.test(f)).sort()) {
      const raw = fs.readFileSync(path.join(dir, file), 'utf8');
      const [, fm, body] = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)$/);
      const slug = file.replace(/\.mdx?$/, '');
      const title = JSON.parse(fm.match(/^title: (.*)$/m)[1]);
      const doc = docs.push({ title, href: `/${kind}/${slug}`, topic: topicOf(slug) }) - 1;
      for (const p of paragraphs(body)) chunks.push({ doc, text: p });
    }
  }
  console.log(`embedding ${chunks.length} paragraphs from ${docs.length} documents…`);

  const embed = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', { dtype: 'fp32' });
  const vectors = [];
  for (let i = 0; i < chunks.length; i += 32) {
    const batch = chunks.slice(i, i + 32).map((c) => c.text);
    const t = await embed(batch, { pooling: 'mean', normalize: true });
    vectors.push(...t.tolist());
  }

  const umap = new UMAP({ nComponents: 2, nNeighbors: 12, minDist: 0.35, spread: 1.2, random: mulberry32(7) });
  const xy = umap.fit(vectors);

  // Normalise to 0..1 with a small margin
  const xs = xy.map((p) => p[0]);
  const ys = xy.map((p) => p[1]);
  const [x0, x1, y0, y1] = [Math.min(...xs), Math.max(...xs), Math.min(...ys), Math.max(...ys)];
  const points = chunks.map((c, i) => ({
    x: +((xy[i][0] - x0) / (x1 - x0)).toFixed(4),
    y: +((xy[i][1] - y0) / (y1 - y0)).toFixed(4),
    d: c.doc,
    text: c.text.length > 170 ? c.text.slice(0, 167).replace(/\s+\S*$/, '') + '…' : c.text,
  }));

  const data = {
    model: 'all-MiniLM-L6-v2',
    projection: 'UMAP (2D)',
    generated: new Date().toISOString().slice(0, 10),
    topics: Object.fromEntries(Object.entries(TOPICS).map(([k, v]) => [k, v.label])),
    docs,
    points,
  };
  fs.writeFileSync(path.join(out, 'embedding-map.json'), JSON.stringify(data));
  console.log(`embedding-map.json: ${points.length} points`);
}

/* ------------------------------------------------------------------ */
/* 2. One real day of day-ahead prices                                 */
/* ------------------------------------------------------------------ */
async function buildEnergy(day = '2026-11-17', bzn = 'DE-LU') {
  const res = await fetch(`https://api.energy-charts.info/price?bzn=${bzn}&start=${day}&end=${day}`);
  const json = await res.json();
  const data = {
    day,
    bzn,
    unit: json.unit,
    prices: json.price.map((p) => +p.toFixed(2)),
    source: 'Energy-Charts (Fraunhofer ISE)',
    license: json.license_info,
  };
  fs.writeFileSync(path.join(out, 'energy-prices.json'), JSON.stringify(data));
  console.log(`energy-prices.json: ${data.prices.length} values for ${bzn} on ${day}`);
}

/* ------------------------------------------------------------------ */
/* 3. Waveform of the Speech2Text test recording                       */
/* ------------------------------------------------------------------ */
function buildWaveform(bins = 200) {
  const audio = path.join(root, '../ai-portfolio/projects/speech2text/data/test_audio.mp3');
  const RATE = 8000;
  const pcm = execFileSync('ffmpeg', ['-v', 'error', '-i', audio, '-ac', '1', '-ar', String(RATE), '-f', 's16le', '-'], {
    maxBuffer: 64 * 1024 * 1024,
  });
  const samples = new Int16Array(pcm.buffer, pcm.byteOffset, pcm.length / 2);
  const rmsOf = (from, to) => {
    let sum = 0;
    for (let i = from; i < to; i++) sum += samples[i] * samples[i];
    return Math.sqrt(sum / Math.max(1, to - from));
  };

  // Amplitude envelope for drawing
  const size = Math.floor(samples.length / bins);
  const env = Array.from({ length: bins }, (_, b) => rmsOf(b * size, (b + 1) * size));
  const max = Math.max(...env);

  // Same rule as speech2text/utils.py: silences of 750 ms or more, 14 dB below the clip's level
  // (pydub detect_silence with silence_thresh = dBFS - 14), split in the middle of each silence
  const db = (r) => 20 * Math.log10(Math.max(r, 1e-9) / 32768);
  const thresh = db(rmsOf(0, samples.length)) - 14;
  const win = RATE / 100; // 10 ms
  const splits = [];
  let start = -1;
  for (let i = 0; i + win <= samples.length; i += win) {
    const quiet = db(rmsOf(i, i + win)) < thresh;
    if (quiet && start < 0) start = i;
    if ((!quiet || i + 2 * win > samples.length) && start >= 0) {
      if (i - start >= 0.75 * RATE) splits.push(+(((start + i) / 2) / RATE).toFixed(2));
      start = -1;
    }
  }

  const data = {
    source: 'speech2text/data/test_audio.mp3',
    seconds: +(samples.length / RATE).toFixed(2),
    bins: env.map((v) => +(v / max).toFixed(3)),
    splits,
  };
  fs.writeFileSync(path.join(out, 'waveform.json'), JSON.stringify(data));
  console.log(`waveform.json: ${bins} bins, ${splits.length} split points at ${splits.join(', ')} s`);
}

if (!only || only === 'energy') await buildEnergy();
if (!only || only === 'waveform') buildWaveform();
if (!only || only === 'embedding') await buildEmbeddingMap();

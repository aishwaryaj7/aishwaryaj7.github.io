// @ts-check
import { defineConfig, fontProviders } from 'astro/config';

import mdx from '@astrojs/mdx';

export default defineConfig({
  site: 'https://aishwaryaj7.github.io',

  // Emit /blog/post.html (not /blog/post/index.html) so URLs from the old site keep working
  build: { format: 'file' },

  trailingSlash: 'never',
  devToolbar: { enabled: false },
  // Mermaid is one large lazy-loaded chunk, fetched only by pages that contain a ```mermaid block
  vite: { build: { chunkSizeWarningLimit: 2500 } },

  fonts: [
    {
      provider: fontProviders.google(),
      name: 'Geist',
      cssVariable: '--font-sans',
      weights: ['300 800'],
      fallbacks: ['ui-sans-serif', 'system-ui', 'sans-serif'],
    },
    {
      provider: fontProviders.google(),
      name: 'Geist Mono',
      cssVariable: '--font-mono',
      weights: ['400 500'],
      fallbacks: ['ui-monospace', 'monospace'],
    },
  ],

  markdown: {
    syntaxHighlight: { type: 'shiki', excludeLangs: ['mermaid', 'math'] },
    shikiConfig: {
      themes: { light: 'github-light', dark: 'github-dark-dimmed' },
      defaultColor: false,
      wrap: false,
    },
  },

  integrations: [mdx()],
});
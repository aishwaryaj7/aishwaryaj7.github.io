// Enhancements for long-form pages: code block toolbars and themed diagrams.

/* ---------- Code blocks: language label + copy button ---------- */
const LANG_NAMES: Record<string, string> = {
  py: 'Python', python: 'Python', bash: 'Shell', sh: 'Shell', shell: 'Shell', dockerfile: 'Dockerfile',
  json: 'JSON', yaml: 'YAML', yml: 'YAML', ts: 'TypeScript', js: 'JavaScript', sql: 'SQL', toml: 'TOML',
  plaintext: 'Text', text: 'Text', txt: 'Text',
};

const COPY_ICON =
  '<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>';

for (const pre of document.querySelectorAll<HTMLPreElement>('.prose pre.astro-code')) {
  const lang = (pre.dataset.language ?? 'text').toLowerCase();
  const wrapper = document.createElement('div');
  wrapper.className = 'code-block';
  const bar = document.createElement('div');
  bar.className = 'code-bar';
  bar.innerHTML = `<span>${LANG_NAMES[lang] ?? lang}</span><button type="button" class="copy-btn" aria-label="Copy code">${COPY_ICON}<span>Copy</span></button>`;
  pre.replaceWith(wrapper);
  wrapper.append(bar, pre);

  const btn = bar.querySelector<HTMLButtonElement>('.copy-btn')!;
  const label = btn.querySelector('span')!;
  btn.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(pre.innerText.replace(/\n$/, ''));
      label.textContent = 'Copied';
      btn.setAttribute('data-copied', '');
    } catch {
      label.textContent = 'Press ⌘C';
    }
    setTimeout(() => {
      label.textContent = 'Copy';
      btn.removeAttribute('data-copied');
    }, 1800);
  });
}

/* ---------- Diagrams: Mermaid, loaded only on pages that have one ---------- */
// Shiki skips mermaid (see excludeLangs in astro.config.mjs), leaving <pre><code class="language-mermaid">
const sources = [...document.querySelectorAll<HTMLElement>('.prose code.language-mermaid')].map((code) => code.closest('pre')!);

if (sources.length) {
  const figures = sources.map((pre) => {
    const fig = document.createElement('figure');
    fig.className = 'diagram';
    fig.dataset.state = 'loading';
    fig.dataset.source = pre.textContent ?? '';
    pre.replaceWith(fig);
    return fig;
  });

  const token = (name: string) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

  const render = async () => {
    const { default: mermaid } = await import('mermaid');
    const font = token('--font-sans') || 'system-ui, sans-serif';
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: 'strict',
      theme: 'base',
      look: 'classic',
      fontFamily: font,
      themeVariables: {
        fontFamily: font,
        fontSize: '13px',
        background: 'transparent',
        primaryColor: token('--surface'),
        primaryTextColor: token('--text'),
        primaryBorderColor: token('--border-strong'),
        secondaryColor: token('--bg-subtle'),
        tertiaryColor: token('--bg-subtle'),
        lineColor: token('--text-faint'),
        textColor: token('--text-muted'),
        edgeLabelBackground: token('--bg-subtle'),
        clusterBkg: 'transparent',
        clusterBorder: token('--border'),
        nodeBorder: token('--border-strong'),
        noteBkgColor: token('--surface'),
      },
      flowchart: {
        curve: 'rounded',
        nodeSpacing: 16,
        rankSpacing: 26,
        padding: 8,
        diagramPadding: 4,
        htmlLabels: true,
        useMaxWidth: true,
        wrappingWidth: 160,
      },
    });

    await Promise.all(
      figures.map(async (fig, i) => {
        // Render inside the figure so labels are measured at the figure's font size
        const { svg } = await mermaid.render(`diagram-${i}-${Date.now()}`, fig.dataset.source!, fig);
        fig.innerHTML = svg;
        fig.dataset.state = 'ready';
      }),
    );
  };

  render();
  document.addEventListener('themechange', render);
}

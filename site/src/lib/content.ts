import { getCollection, type CollectionEntry } from 'astro:content';

export async function getProjects() {
  const all = await getCollection('projects');
  return all.sort((a, b) => a.data.order - b.data.order);
}

export async function getPosts() {
  const all = await getCollection('blog', ({ data }) => import.meta.env.DEV || !data.draft);
  return all.sort((a, b) => b.data.date.valueOf() - a.data.date.valueOf());
}

/** ~230 words per minute, ignoring code blocks a reader skims */
export function readingTime(entry: CollectionEntry<'blog'> | CollectionEntry<'projects'>) {
  const prose = (entry.body ?? '').replace(/```[\s\S]*?```/g, ' ');
  const words = prose.split(/\s+/).filter(Boolean).length;
  return Math.max(1, Math.round(words / 230));
}

export const formatMonth = (d: Date) =>
  d.toLocaleDateString('en-GB', { month: 'short', year: 'numeric', timeZone: 'UTC' });

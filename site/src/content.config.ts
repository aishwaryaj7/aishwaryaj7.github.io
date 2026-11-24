import { defineCollection } from 'astro:content';
import { glob } from 'astro/loaders';
import { z } from 'astro/zod';

const projects = defineCollection({
  loader: glob({ pattern: '*.{md,mdx}', base: './src/content/projects' }),
  schema: z.object({
    title: z.string(),
    tagline: z.string(),
    /** Short text for the card on the home page */
    summary: z.string(),
    /** Lead paragraph on the project page and <meta name="description"> */
    description: z.string(),
    /** Headline results, shown on the featured card and the project page */
    metrics: z.array(z.object({ value: z.string(), label: z.string() })).optional(),
    stack: z.array(z.string()),
    /** Skills the project shows, as tiles on the project page */
    skills: z.array(z.string()).default([]),
    tags: z.array(z.enum(['genai', 'mlops', 'ml'])),
    demo: z.url().optional(),
    code: z.url().optional(),
    /** Lower comes first on the home page */
    order: z.number(),
    featured: z.boolean().default(false),
  }),
});

const blog = defineCollection({
  loader: glob({ pattern: '*.{md,mdx}', base: './src/content/blog' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    date: z.coerce.date(),
    tags: z.array(z.string()),
    draft: z.boolean().default(false),
  }),
});

export const collections = { projects, blog };

export const site = {
  name: 'Aishwarya Jauhari Sharma',
  shortName: 'Aishwarya',
  role: 'Machine Learning Engineer',
  location: 'Germany',
  description:
    'Machine learning engineer in Germany with 5+ years across applied research, e-commerce and enterprise software, now focused on LLM engineering: RAG and agents. Projects, write-ups and CV.',
  email: 'aishwarya.jauhari@icloud.com',
  github: 'https://github.com/aishwaryaj7',
  linkedin: 'https://www.linkedin.com/in/aishwaryajsharma/',
  openToWork: true,
};

export type TimelineEntry = {
  years: string;
  role: string;
  org: string;
  /** What the role was about, in a few words */
  note?: string;
  kind: 'work' | 'study' | 'break';
  current?: boolean;
};

// Newest first, read left to right.
export const timeline: TimelineEntry[] = [
  { years: '2024 – Now', role: 'Career break', org: 'Maternity and childcare', note: 'Building LLM agents and RAG systems', kind: 'break', current: true },
  { years: '2022 – 2024', role: 'ML Engineer', org: 'Fraunhofer FHR', note: 'Deep learning, computer vision and MLOps', kind: 'work' },
  { years: '2020 – 2022', role: 'Data Analyst', org: 'eBay, Berlin', note: 'ML models, analytics, stakeholder engagement', kind: 'work' },
  { years: '2019 – 2022', role: 'M.Sc. Digital Engineering', org: 'OVGU Magdeburg', note: 'Specialisation in ML and data science', kind: 'study' },
  { years: '2016 – 2018', role: 'Software Engineer', org: 'Accenture, Bangalore', note: 'SAP ABAP development for retail clients', kind: 'work' },
  { years: '2012 – 2016', role: 'B.E. Instrumentation', org: 'VTU, Bangalore', kind: 'study' },
];

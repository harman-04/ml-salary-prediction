export const ageOptions = [
  'Under 18 years old',
  '18-24 years old',
  '25-34 years old',
  '35-44 years old',
  '45-54 years old',
  '55-64 years old',
  '65 years or older',
];

export const countryOptions = [
  'United States of America',
  'India',
  'Germany',
  'United Kingdom of Great Britain and Northern Ireland',
  'Canada',
  'France',
  'Brazil',
  'Poland',
  'Netherlands',
  'Australia',
  'Ukraine',
  'Spain',
  'Italy',
  'Sweden',
  'Switzerland',
  'Austria',
  'Czech Republic',
  'Russian Federation',
];

export const orgSizeOptions = [
  'Just me - I am a freelancer, sole proprietor, etc.',
  '2 to 9 employees',
  '10 to 19 employees',
  '20 to 99 employees',
  '100 to 499 employees',
  '500 to 999 employees',
  '1,000 to 4,999 employees',
  '5,000 to 9,999 employees',
  '10,000 or more employees',
  'I don’t know',
];

export const employmentOptions = [
  'Employed, full-time',
  'Independent contractor, freelancer, or self-employed',
  'Student, full-time',
  'Student, part-time',
  'Not employed, and not looking for work',
  'Not employed, but looking for work',
  'Employed, part-time',
  'Retired',
];

export const edLevelOptions = [
  'Primary/elementary school',
  'Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)',
  'Some college/university study without earning a degree',
  'Associate degree (A.A., A.S., etc.)',
  'Bachelor’s degree (B.A., B.S., B.Eng., etc.)',
  'Master’s degree (M.A., M.S., M.Eng., MBA, etc.)',
  'Professional degree (JD, MD, Ph.D, Ed.D, etc.)',
  'Something else',
];

export const devTypeOptions = [
  'Academic researcher', // Added from PDF [cite: 33]
  'Blockchain', // Added from PDF [cite: 33]
  'Cloud infrastructure engineer',
  'Data or business analyst', // Added from PDF [cite: 33]
  'Data engineer',
  'Data scientist or machine learning specialist',
  'Database administrator', // Added from PDF [cite: 33]
  'Designer', // Added from PDF [cite: 33]
  'Developer Advocate', // Added from PDF [cite: 33]
  'Developer, AI', // Added from PDF [cite: 33]
  'Developer, back-end',
  'Developer, desktop or enterprise applications',
  'Developer, embedded applications or devices',
  'Developer Experience', // Added from PDF [cite: 33]
  'Developer, front-end',
  'Developer, full-stack',
  'Developer, game or graphics', // Added from PDF [cite: 34]
  'Developer, mobile',
  'Developer, QA or test',
  'DevOps specialist',
  'Educator', // Added from PDF [cite: 34]
  'Engineer, site reliability', // Added from PDF [cite: 34]
  'Engineering manager',
  'Hardware Engineer', // Added from PDF [cite: 34]
  'Marketing or sales professional', // Added from PDF [cite: 34]
  'Product manager', // Added from PDF [cite: 34]
  'Project manager', // Added from PDF [cite: 34]
  'Research & Development role',
  'Scientist', // Added from PDF [cite: 34]
  'Senior Executive (C-Suite, VP, etc.)',
  'Student', // Added from PDF [cite: 34]
  'System administrator', // Added from PDF [cite: 34]
  'Security professional', // Added from PDF [cite: 34]
  'Other (please specify):',
];

export const remoteWorkOptions = [
  'Remote',
  'Hybrid (some remote, some in-person)',
  'In-person',
];

export const jobSatOptions = [
  'Very satisfied',
  'Slightly satisfied',
  'Neither satisfied nor dissatisfied',
  'Slightly dissatisfied',
  'Very dissatisfied',
];

// **CRITICAL:** Mapping for JobSat to numerical values as expected by your backend if it was trained numerically.
// You MUST verify these numerical values match your backend's internal mapping or training data.
// Replace the placeholder numbers below with your verified backend mapping.
export const jobSatMapping = {
  "Very dissatisfied": 0, // VERIFY THIS NUMBER FROM YOUR BACKEND
  "Slightly dissatisfied": 4, // VERIFY THIS NUMBER FROM YOUR BACKEND
  "Neither satisfied nor dissatisfied": 5, // VERIFY THIS NUMBER FROM YOUR BACKEND
  "Slightly satisfied": 6, // VERIFY THIS NUMBER FROM YOUR BACKEND
  "Very satisfied": 10, // VERIFY THIS NUMBER FROM YOUR BACKEND
};

export const techSkills = {
  Language: [
    'Bash/Shell (all shells)', 'C', 'C#', 'C++', 'Dart', 'Elixir', 'Erlang', 'F#', 'Go',
    'HTML/CSS', 'Haskell', 'Java', 'JavaScript', 'Julia', 'Kotlin', 'Lua', 'Objective-C',
    'PHP', 'PowerShell', 'Python', 'R', 'Ruby', 'Rust', 'Scala', 'SQL', 'Swift', 'TypeScript', 'VBA',
    'GDScript', 'Apex', 'Assembly', 'Lisp', 'Fortran', 'COBOL', 'APL', 'Prolog', 'Clojure', 'Crystal', 'Flow',
    'Zig', 'Common Lisp', 'Ocaml', 'Smalltalk', 'Nim', 'Raku', 'SAS', 'Ada', 'Less', 'Sass', 'CSS', 'XML', 'PL/SQL',
    'T-SQL', 'CoffeeScript', 'Visual Basic .NET', 'Pascal', 'Scheme', 'VHDL', 'Verilog', 'D', 'Forth', 'Logo', 'Awk', 'Perl',
    'Pike', 'PureBasic', 'Rebol', 'Rexx', 'Simula', 'Solidity', 'Tcl', 'UnrealScript', 'Vala', 'WebAssembly'
  ],
  Database: [
    'BigQuery', 'Cassandra', 'Cloud Firestore', 'Couchbase', 'Databricks SQL', 'DuckDB',
    'DynamoDB', 'Elasticsearch', 'Firebase Realtime Database', 'H2', 'MariaDB',
    'Microsoft Access', 'Microsoft SQL Server', 'MongoDB', 'MySQL', 'Neo4j', 'Oracle',
    'PostgreSQL', 'Redis', 'Snowflake', 'SQLite', 'Supabase'
  ],
  Platform: [
    'Amazon Web Services (AWS)', 'Azure DevOps', 'Cloudflare', 'Cloudways', 'Consul', 'Digital Ocean',
    'Docker', 'Fly.io', 'Google Cloud', 'Heroku', 'Hetzner', 'Kubernetes', 'Linode, now Akamai',
    'Managed Hosting', 'Microsoft Azure', 'Netlify', 'OpenShift', 'OpenStack',
    'Oracle Cloud Infrastructure (OCI)', 'Render', 'Supabase', 'Vercel', 'VMware'
  ],
  Webframe: [
    'Angular', 'AngularJS', 'ASP.NET', 'ASP.NET Core', 'Astro', 'Blazor', 'Django',
    'Deno', 'Express', 'FastAPI', 'Flask', 'Gatsby', 'Htmx', 'jQuery', 'Laravel',
    'Next.js', 'NestJS', 'Node.js', 'Nuxt.js', 'Phoenix', 'React', 'Remix', 'Ruby on Rails',
    'Svelte', 'Spring Boot', 'Solid.js', 'Symfony', 'WordPress'
  ],
  MiscTech: [
    '.NET (5+)', '.NET Framework (1.0 - 4.8)', 'Apache Kafka', 'Apache Spark', 'CUDA',
    'Dapr', 'Electron', 'Flutter', 'Hadoop', 'Hugging Face Transformers', 'Ionic',
    'Keras', 'LLVM', 'MLflow', '.NET MAUI', 'NumPy', 'OpenCV', 'OpenGL', 'Pandas', 'Qt',
    'RabbitMQ', 'React Native', 'Ruff', 'Scikit-learn', 'Spring Framework', 'SwiftUI',
    'Tauri', 'TensorFlow', 'Torch/PyTorch', 'Xamarin'
  ],
  ToolsTech: [
    'Ansible', 'APT', 'Bun', 'Chocolatey', 'Composer', 'Deno', 'Docker', 'Gatsby',
    'Git', 'GitHub', 'GitLab', 'Go', 'Godot', 'Gradle', 'Homebrew', 'Jira', 'Jupyter',
    'Kubernetes', 'Make', 'Maven (build tool)', 'MSBuild', 'npm', 'NuGet', 'Nix', 'Pacman',
    'Pip', 'pnpm', 'Podman', 'PowerShell', 'PyTorch', 'Rust', 'Terraform', 'Unity 3D',
    'Unreal Engine', 'Vite', 'Webpack', 'Yarn'
  ],
  NEWCollabTools: [
    'Asana', 'Basecamp', 'Confluence', 'Discord', 'Google Chat', 'Google Docs',
    'Google Meet', 'Google Sheets', 'Jira', 'Microsoft Teams', 'Notion', 'Slack',
    'Trello', 'Zoom'
  ],
  OfficeStackAsync: [
    'Asana', 'Basecamp', 'Confluence', 'Google Docs', 'Google Sheets', 'Jira',
    'Notion', 'Trello'
  ],
  OfficeStackSync: [
    'Discord', 'Google Chat', 'Google Meet', 'Microsoft Teams', 'Slack Calls', 'Zoom'
  ],
  AISearchDev: [
    'Bard', 'Bing AI', 'ChatGPT', 'Codeium', 'Copilot', 'Google Gemini Code Assistance',
    'Perplexity AI', 'Tabnine', 'WolframAlpha', 'AWS CodeWhisperer'
  ],
  // ADDED NEW CATEGORY FOR AIToolHere
  AITool: [
    'Hugging Face', 'Google Cloud AI', 'Azure AI Platform', 'OpenAI API', 'TensorFlow',
    'PyTorch', 'Keras', 'Scikit-learn', 'GCP Vertex AI', 'AWS SageMaker', 'MLflow',
    'ONNX', 'Ray', 'Tencent Cloud AI', 'Baidu AI Cloud', 'MindSpore', 'PaddlePaddle',
    'None', 'Don\'t know'
  ]
};


export const knowledgeOptions = [
  'Strongly disagree',
  'Disagree',
  'Neutral',
  'Agree',
  'Strongly agree',
];

export const frequencyOptions = [
  'Never',
  'Rarely',
  'Sometimes',
  'Often',
  'Daily',
];

export const jobSatPointsOptions = [
  'Strongly disagree',
  'Disagree',
  'Neutral',
  'Agree',
  'Strongly agree',
];
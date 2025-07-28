// src/lib/schema.js
import { z } from 'zod';

export const predictionSchema = z.object({
  YearsCodePro: z.coerce.number().min(0, "Years of professional coding must be 0 or more"),
  WorkExp: z.coerce.number().min(0, "Total work experience must be 0 or more"),
  Age: z.string().min(1, "Age range is required"),
  EdLevel: z.string().min(1, "Education level is required"),
  DevType: z.string().min(1, "Developer type is required"),
  Country: z.string().min(1, "Country is required"),
  OrgSize: z.string().min(1, "Organization size is required"),
  RemoteWork: z.string().min(1, "Remote work preference is required"),
  Employment: z.string().min(1, "Employment status is required"),

  // **JobSat: Frontend will now map this to a number before sending**
  // This schema keeps it a string because the UI collects it as such,
  // but we'll convert it in onSubmit of SalaryPredictor.jsx
  JobSat: z.string().min(1, "Job satisfaction is required"),

  // --- ALL OTHER FIELDS from Postman payload (with defaults if not collected in UI) ---
   CompTotal: z.coerce.number().min(0, "Total compensation must be 0 or more").optional().nullable().default(null), // RE-ADDED CompTotal

  // Multi-select tech stacks (even if not in UI, define with default empty array)
  LanguageHaveWorkedWith: z.array(z.string()).optional().default([]),
  DatabaseHaveWorkedWith: z.array(z.string()).optional().default([]),
  PlatformHaveWorkedWith: z.array(z.string()).optional().default([]),
  WebframeHaveWorkedWith: z.array(z.string()).optional().default([]),
  MiscTechHaveWorkedWith: z.array(z.string()).optional().default([]),
  ToolsTechHaveWorkedWith: z.array(z.string()).optional().default([]),
  NEWCollabToolsHaveWorkedWith: z.array(z.string()).optional().default([]),
  OfficeStackAsyncHaveWorkedWith: z.array(z.string()).optional().default([]),
  OfficeStackSyncHaveWorkedWith: z.array(z.string()).optional().default([]),
  AISearchDevHaveWorkedWith: z.array(z.string()).optional().default([]),
  AIToolHaveWorkedWith: z.array(z.string()).optional().default([]),

  // Knowledge/Frequency/JobSatPoints (assuming they are strings in backend, default to empty string)
  Knowledge_1: z.string().min(1, "Knowledge of Cloud Computing is required"), // Added min(1) for validation
  Knowledge_2: z.string().min(1, "Knowledge of Web Development is required"), // Added min(1) for validation
  Frequency_1: z.string().min(1, "Frequency of strategy driving is required"), // Added min(1) for validation
  JobSatPoints_1: z.string().min(1, "Satisfaction with career growth is required"), // Added min(1) for validation
});
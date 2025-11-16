import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

type AgentCtor = new (...args: any[]) => AgentInstance;
type AgentInstance = {
  run?: (input: string) => Promise<unknown> | unknown;
  invoke?: (input: string) => Promise<unknown> | unknown;
  execute?: (input: string) => Promise<unknown> | unknown;
};

type OllamaCtor = new (...args: any[]) => unknown;

type CliOptions = {
  filePath: string;
  model: string;
  ollamaUrl: string;
  maxChars: number;
};

const SYSTEM_PROMPT = `You are a meticulous editor. Summarize the supplied text file and
share an informed opinion about its overall quality. Always respond with JSON
that matches the following schema:

{
  "summary": "<2-4 sentence overview>",
  "opinion": "<constructive opinion about clarity, tone, usefulness, etc.>"
}

If the content is truncated or incomplete, clearly mention that in both the
summary and opinion. Avoid inventing details that are not present in the file.`;

async function loadModule(modulePath: string): Promise<Record<string, any> | null> {
  try {
    return await import(modulePath);
  } catch (error) {
    if (error instanceof Error &&
        "code" in error &&
        typeof (error as NodeJS.ErrnoException).code === "string" &&
        (error as NodeJS.ErrnoException).code === "ERR_MODULE_NOT_FOUND") {
      return null;
    }
    throw error;
  }
}

async function resolveAgentCtor(): Promise<AgentCtor> {
  const root = await loadModule("mastra");
  if (root?.Agent) {
    return root.Agent as AgentCtor;
  }
  for (const candidate of ["mastra/agent", "mastra/agents"] as const) {
    const mod = await loadModule(candidate);
    if (mod?.Agent) {
      return mod.Agent as AgentCtor;
    }
  }
  throw new Error("Unable to locate Mastra's Agent export. Check your mastra version.");
}

async function resolveOllamaCtor(): Promise<OllamaCtor> {
  const root = await loadModule("mastra");
  for (const key of ["Ollama", "OllamaLLM"]) {
    if (root?.[key]) {
      return root[key] as OllamaCtor;
    }
  }
  for (const candidate of ["mastra/llms/ollama", "mastra/llm/ollama"]) {
    const mod = await loadModule(candidate);
    for (const key of ["Ollama", "OllamaLLM"]) {
      if (mod?.[key]) {
        return mod[key] as OllamaCtor;
      }
    }
  }
  throw new Error("Unable to find Mastra's Ollama integration. Please install the correct package.");
}

function ensureString(value: unknown, fallback = ""): string {
  if (typeof value === "string" && value.length > 0) {
    return value;
  }
  return fallback;
}

function parseAgentResponse(raw: string): { summary: string; opinion: string; rawResponse: string } {
  try {
    const payload = JSON.parse(raw);
    return {
      summary: ensureString(payload.summary, raw.trim()),
      opinion: ensureString(payload.opinion, "Unable to parse opinion from response."),
      rawResponse: raw,
    };
  } catch (error) {
    return {
      summary: raw.trim(),
      opinion: "Unable to parse opinion from response.",
      rawResponse: raw,
    };
  }
}

async function buildAgent(model: string, ollamaUrl: string): Promise<AgentInstance> {
  const [Agent, Ollama] = await Promise.all([resolveAgentCtor(), resolveOllamaCtor()]);
  const llm = new (Ollama as any)({ model, baseUrl: ollamaUrl });
  const agent = new (Agent as any)({
    name: "file_analyst",
    llm,
    instructions: SYSTEM_PROMPT,
    systemPrompt: SYSTEM_PROMPT,
    prompt: SYSTEM_PROMPT,
  });
  return agent as AgentInstance;
}

async function callAgent(agent: AgentInstance, prompt: string): Promise<string> {
  const runner = agent.run ?? agent.invoke ?? agent.execute;
  if (!runner) {
    throw new Error("Mastra agent does not expose run/invoke/execute methods.");
  }
  const result = await runner.call(agent, prompt);
  if (typeof result === "string") {
    return result;
  }
  if (typeof result === "object" && result !== null) {
    if ("output" in result && typeof (result as { output: unknown }).output === "string") {
      return (result as { output: string }).output;
    }
    return JSON.stringify(result, null, 2);
  }
  return String(result);
}

async function readFileWithLimit(path: string, maxChars: number): Promise<{ content: string; truncated: boolean }> {
  const data = await readFile(path, "utf8");
  if (data.length <= maxChars) {
    return { content: data, truncated: false };
  }
  return { content: data.slice(0, maxChars), truncated: true };
}

async function analyzeFile(opts: CliOptions): Promise<void> {
  const absolutePath = resolve(opts.filePath);
  const { content, truncated } = await readFileWithLimit(absolutePath, opts.maxChars);
  const truncationNote = truncated
    ? `The file content was truncated to the first ${opts.maxChars.toLocaleString()} characters.`
    : "";
  const prompt = `Analyze the file named "${absolutePath}".\n${truncationNote}\n\n<content>\n${content}\n</content>`;

  const agent = await buildAgent(opts.model, opts.ollamaUrl);
  const raw = await callAgent(agent, prompt);
  const analysis = parseAgentResponse(raw);

  console.log("Summary:\n" + analysis.summary);
  console.log("\nOpinion:\n" + analysis.opinion);
}

function parseArgs(argv: string[]): CliOptions {
  if (argv.length === 0) {
    throw new Error("Usage: ts-node src/index.ts <file> [--model llama3] [--ollama-url http://127.0.0.1:11434] [--max-chars 12000]");
  }
  let filePath: string | null = null;
  let model = "llama3";
  let ollamaUrl = "http://127.0.0.1:11434";
  let maxChars = 12_000;

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--") && filePath === null) {
      filePath = arg;
      continue;
    }
    if (arg === "--model") {
      model = argv[++i] ?? model;
      continue;
    }
    if (arg === "--ollama-url") {
      ollamaUrl = argv[++i] ?? ollamaUrl;
      continue;
    }
    if (arg === "--max-chars") {
      const value = Number(argv[++i]);
      if (!Number.isFinite(value) || value <= 0) {
        throw new Error("--max-chars must be a positive number");
      }
      maxChars = Math.floor(value);
      continue;
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  if (!filePath) {
    throw new Error("File path is required.");
  }

  return { filePath, model, ollamaUrl, maxChars };
}

async function main(): Promise<void> {
  try {
    const options = parseArgs(process.argv.slice(2));
    await analyzeFile(options);
  } catch (error) {
    console.error((error as Error).message);
    process.exit(1);
  }
}

void main();

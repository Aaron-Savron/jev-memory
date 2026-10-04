import type { DecisionInput, DecisionProvider, DecisionResult } from './types.js';

export function validateDecisionResult(input: DecisionInput, value: unknown, identity: string): DecisionResult {
  if (!value || typeof value !== 'object') throw new TypeError('Invalid decision response');
  const result = value as DecisionResult;
  if (result.model !== identity || !Array.isArray(result.decisions) || result.decisions.length !== input.candidates.length) throw new TypeError('Incomplete response or unexpected model identity');
  const ids = new Set(input.candidates.map(c => c.id));
  for (const item of result.decisions) {
    if (!item || !ids.delete(item.id) || !Number.isFinite(item.score) || item.score < 0 || item.score > 1) throw new TypeError('Invalid decision ID/probability');
    if (input.operation === 'relationship' && !['same', 'contradicts', 'unrelated'].includes(item.relation ?? '')) throw new TypeError('Invalid relationship');
  }
  if (ids.size) throw new TypeError('Missing decisions');
  return result;
}

export class HttpDecisionProvider implements DecisionProvider {
  readonly identity: string;
  readonly #url: string;
  readonly #token: string;
  #failures = 0;
  #retryAt = 0;

  constructor(options: { url: string; token: string; identity: string; allowInsecureLoopback?: boolean }) {
    const url = new URL(options.url);
    const loopback = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
    if (url.protocol !== 'https:' && !(options.allowInsecureLoopback && loopback && url.protocol === 'http:')) throw new TypeError('Decision service requires HTTPS, or explicitly enabled loopback HTTP');
    if (url.username || url.password || url.search || url.hash) throw new TypeError('Credentials/query parameters are not allowed in the service URL');
    if (!options.token || !options.identity) throw new TypeError('Token and pinned model identity required');
    this.#url = options.url.replace(/\/$/, '') + '/v1/memory/decide';
    this.#token = options.token; this.identity = options.identity;
  }

  async decide(input: DecisionInput, signal: AbortSignal): Promise<DecisionResult> {
    if (Date.now() < this.#retryAt) throw new Error('Decision circuit open');
    try {
      const response = await fetch(this.#url, {
        method: 'POST', redirect: 'error', signal,
        headers: { authorization: `Bearer ${this.#token}`, 'content-type': 'application/json' },
        body: JSON.stringify(input),
      });
      if (!response.ok || !response.body) throw new Error(`Decision service HTTP ${response.status}`);
      if (Number(response.headers.get('content-length') ?? 0) > 262144) throw new Error('Oversized decision response');
      const reader = response.body.getReader(); const parts: Uint8Array[] = []; let bytes = 0;
      try {
        while (true) {
          const { value, done } = await reader.read(); if (done) break;
          bytes += value.length;
          if (bytes > 262144) throw new Error('Oversized decision response');
          parts.push(value);
        }
      } catch (e) { await reader.cancel().catch(() => undefined); throw e; }
      finally { reader.releaseLock(); }
      const result = validateDecisionResult(input, JSON.parse(Buffer.concat(parts).toString('utf8')), this.identity);
      this.#failures = 0; this.#retryAt = 0; return result;
    } catch (e) {
      if (++this.#failures >= 3) this.#retryAt = Date.now() + 10000;
      throw e;
    }
  }
}

type OpenJevQuestion =
  | { type: 'noul'; instructions: string; criteria: { true: string; false: string } }
  | { type: 'choice'; instructions: string; criteria: Record<string, string> };

type OpenJevAnswer = {
  noul?: unknown;
  value?: unknown;
  choice?: unknown;
  probabilities?: unknown;
};

/**
 * Calls the native Open-Jev `/v1/systemone` API directly.
 *
 * This works with the hosted API and with any self-hosted Open-Jev-compatible
 * server. It translates the memory provider contract into independent Noul or
 * Choice questions, then validates and normalizes the typed response.
 */
export class OpenJevApiProvider implements DecisionProvider {
  readonly identity: string;
  readonly #url: string;
  readonly #apiKey: string;
  readonly #model: string;
  readonly #headers: Record<string, string>;
  #failures = 0;
  #retryAt = 0;

  constructor(options: {
    url?: string;
    apiKey?: string;
    model?: string;
    identity?: string;
    allowInsecureLoopback?: boolean;
    headers?: Record<string, string>;
  } = {}) {
    const raw = options.url ?? 'https://api.openjev.sh/v1/systemone';
    const url = new URL(raw);
    const loopback = ['localhost', '127.0.0.1', '[::1]'].includes(url.hostname);
    if (url.protocol !== 'https:' && !(options.allowInsecureLoopback && loopback && url.protocol === 'http:')) {
      throw new TypeError('Open-Jev API requires HTTPS, or explicitly enabled loopback HTTP');
    }
    if (url.username || url.password || url.search || url.hash) throw new TypeError('Credentials/query parameters are not allowed in the API URL');
    if (url.pathname === '/' || url.pathname === '') url.pathname = '/v1/systemone';
    else if (url.pathname.endsWith('/v1')) url.pathname += '/systemone';
    else if (!url.pathname.endsWith('/systemone')) url.pathname = url.pathname.replace(/\/$/, '') + '/v1/systemone';
    this.#url = url.toString();
    this.#apiKey = options.apiKey ?? '';
    this.#model = options.model ?? 'openjev';
    this.identity = options.identity ?? this.#model;
    this.#headers = { ...(options.headers ?? {}) };
    for (const name of Object.keys(this.#headers)) {
      if (!/^[a-z0-9-]+$/i.test(name) || name.toLowerCase() === 'authorization' || name.toLowerCase() === 'host') {
        throw new TypeError('Invalid custom API header');
      }
    }
  }

  async decide(input: DecisionInput, signal: AbortSignal): Promise<DecisionResult> {
    if (Date.now() < this.#retryAt) throw new Error('Decision circuit open');
    const questions: Record<string, OpenJevQuestion> = {};
    for (let index = 0; index < input.candidates.length; index++) {
      const candidate = input.candidates[index]!;
      const id = `candidate_${index}`;
      if (input.operation === 'relationship') {
        questions[id] = {
          type: 'choice',
          instructions: `Compare the previous claim with the candidate claim. Previous: ${candidate.previous ?? '[missing]'}. Candidate: ${candidate.text}`,
          criteria: {
            same: 'The claims express the same fact with compatible qualifiers.',
            contradicts: 'The claims cannot both apply in the same context and time.',
            unrelated: 'The claims differ, are compatible, or do not provide enough evidence.',
          },
        };
      } else {
        questions[id] = {
          type: 'noul',
          instructions: `${input.operation === 'retain' ? 'Is this candidate useful durable memory with source support?' : 'Does this memory help with the current task?'} Candidate: ${candidate.text}`,
          criteria: {
            true: 'Yes. It is useful evidence, a preference, a constraint, a project fact, a decision, or unfinished work.',
            false: 'No. It is filler, speculative, unsafe, unsupported, or unrelated.',
          },
        };
      }
    }
    try {
      const headers: Record<string, string> = {
        ...this.#headers,
        'content-type': 'application/json',
      };
      if (this.#apiKey) headers.authorization = `Bearer ${this.#apiKey}`;
      const response = await fetch(this.#url, {
        method: 'POST', redirect: 'error', signal, headers,
        body: JSON.stringify({ model: this.#model, state: input.context, questions }),
      });
      if (!response.ok || !response.body) throw new Error(`Open-Jev API HTTP ${response.status}`);
      const value = await readJsonBody(response, 262144);
      if (!value || typeof value !== 'object') throw new TypeError('Invalid Open-Jev response');
      const body = value as { model?: unknown; answers?: unknown };
      if (body.model !== undefined && body.model !== this.#model) throw new TypeError('Unexpected Open-Jev model identity');
      if (!body.answers || typeof body.answers !== 'object') throw new TypeError('Open-Jev response has no answers');
      const answers = body.answers as Record<string, OpenJevAnswer>;
      const decisions = input.candidates.map((candidate, index) => {
        const answer = answers[`candidate_${index}`];
        if (!answer || typeof answer !== 'object') throw new TypeError('Open-Jev response is missing a candidate answer');
        if (input.operation === 'relationship') {
          const relation = answer.choice;
          if (relation !== 'same' && relation !== 'contradicts' && relation !== 'unrelated') throw new TypeError('Invalid Open-Jev relationship');
          const probabilities = asProbabilities(answer.probabilities);
          return { id: candidate.id, score: probabilities?.[relation] ?? 0, relation };
        }
        const raw = typeof answer.noul === 'number' ? answer.noul : answer.value;
        if (typeof raw !== 'number' || !Number.isFinite(raw) || raw < 0 || raw > 1) throw new TypeError('Invalid Open-Jev Noul probability');
        return { id: candidate.id, score: raw };
      });
      this.#failures = 0; this.#retryAt = 0;
      return validateDecisionResult(input, { model: this.identity, decisions }, this.identity);
    } catch (error) {
      if (++this.#failures >= 3) this.#retryAt = Date.now() + 10000;
      throw error;
    }
  }
}

/** Short alias for applications that refer to the hosted service as Jev. */
export class JevApiProvider extends OpenJevApiProvider {}

function asProbabilities(value: unknown): Record<string, number> | null {
  if (!value || typeof value !== 'object') return null;
  const result: Record<string, number> = {};
  for (const [key, score] of Object.entries(value)) {
    if (typeof score !== 'number' || !Number.isFinite(score) || score < 0 || score > 1) return null;
    result[key] = score;
  }
  return result;
}

async function readJsonBody(response: Response, maxBytes: number): Promise<unknown> {
  if (Number(response.headers.get('content-length') ?? 0) > maxBytes) throw new Error('Oversized decision response');
  const reader = response.body!.getReader(); const parts: Uint8Array[] = []; let bytes = 0;
  try {
    while (true) {
      const { value, done } = await reader.read(); if (done) break;
      bytes += value.length;
      if (bytes > maxBytes) throw new Error('Oversized decision response');
      parts.push(value);
    }
  } catch (error) { await reader.cancel().catch(() => undefined); throw error; }
  finally { reader.releaseLock(); }
  return JSON.parse(Buffer.concat(parts).toString('utf8'));
}

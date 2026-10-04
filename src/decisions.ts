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

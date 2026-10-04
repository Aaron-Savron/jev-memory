import { createHash } from 'node:crypto';
import type { Access, Candidate, MemoryEvent, Scope } from './types.js';

export const MAX_TEXT = 32768;
export const KINDS = new Set(['preference', 'constraint', 'fact', 'decision', 'open_loop', 'episode', 'procedure']);
const MODALITIES = new Set(['desired', 'observed', 'reported', 'hypothesis']);
export function assertId(value: unknown): asserts value is string {
  if (typeof value !== 'string' || !value.length || value.length > 512 || /[\u0000-\u001f]/u.test(value)) throw new TypeError('Invalid identifier');
}
export function assertScope(scope: Scope): void {
  if (!scope || !['user', 'project', 'workspace', 'conversation'].includes(scope.kind)) throw new TypeError('Invalid scope');
  assertId(scope.id);
}
export function assertAccess(access: Access): void {
  assertId(access.principalId);
  if (!Array.isArray(access.scopes) || access.scopes.length > 32) throw new TypeError('At most 32 authorized scopes');
  access.scopes.forEach(assertScope);
}
export function finiteTime(value: number): void {
  if (!Number.isSafeInteger(value) || value < 0) throw new TypeError('Invalid timestamp');
}
export function validateCandidates(event: MemoryEvent, candidates: Candidate[]): Candidate[] {
  if (!Array.isArray(candidates) || candidates.length > 128) throw new TypeError('At most 128 candidates per event');
  const seen = new Set<string>();
  return candidates.map(c => {
    if (!c || !Number.isInteger(c.start) || !Number.isInteger(c.end) || c.start < 0 || c.end <= c.start || c.end > event.text.length || c.end - c.start > 4096) throw new TypeError('Candidate must reference a bounded source span');
    if (!KINDS.has(c.kind) || !MODALITIES.has(c.modality)) throw new TypeError('Invalid candidate classification');
    const key = `${c.start}:${c.end}`;
    if (seen.has(key)) throw new TypeError('Duplicate candidate source span');
    seen.add(key);
    if (c.subject !== undefined) assertId(c.subject);
    if (c.predicate !== undefined) assertId(c.predicate);
    for (const t of [c.expiresAt, c.validFrom, c.validTo]) if (t !== undefined) finiteTime(t);
    if (c.validFrom !== undefined && c.validTo !== undefined && c.validTo <= c.validFrom) throw new TypeError('Invalid validity interval');
    if (c.dependencies !== undefined) {
      if (!Array.isArray(c.dependencies) || c.dependencies.length > 32) throw new TypeError('At most 32 dependencies');
      c.dependencies.forEach(assertId);
    }
    return { start: c.start, end: c.end, kind: c.kind, modality: c.modality,
      ...(c.subject === undefined ? {} : { subject: c.subject }), ...(c.predicate === undefined ? {} : { predicate: c.predicate }),
      ...(c.expiresAt === undefined ? {} : { expiresAt: c.expiresAt }), ...(c.validFrom === undefined ? {} : { validFrom: c.validFrom }),
      ...(c.validTo === undefined ? {} : { validTo: c.validTo }), dependencies: [...(c.dependencies ?? [])] };
  });
}
export function validateEvent(event: MemoryEvent): MemoryEvent {
  assertId(event.id); assertId(event.principalId); assertScope(event.scope); finiteTime(event.occurredAt);
  if (!['user', 'tool', 'document', 'assistant'].includes(event.source)) throw new TypeError('Invalid source');
  if (typeof event.text !== 'string' || !event.text.trim() || event.text.length > MAX_TEXT) throw new TypeError('Event text must contain 1..32768 characters');
  if (event.verified !== undefined && typeof event.verified !== 'boolean') throw new TypeError('Invalid verification flag');
  return { id: event.id, principalId: event.principalId, scope: { kind: event.scope.kind, id: event.scope.id },
    source: event.source, text: event.text, occurredAt: event.occurredAt,
    ...(event.verified === undefined ? {} : { verified: event.verified }),
    ...(event.candidates === undefined ? {} : { candidates: validateCandidates(event, event.candidates) }) };
}
export function hash(value: unknown): string {
  return createHash('sha256').update(JSON.stringify(value)).digest('hex');
}
// Detect common credentials before storing or transmitting a candidate. Applications
// should also redact their own domain-specific secrets before record().
export function containsSecret(text: string): boolean {
  return /-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16})\b|(?:api[_ -]?key|password|secret|token)\s*[:=]\s*["']?[^\s"']{8,}/i.test(text);
}
export function scopeKey(scope: Scope): string { return JSON.stringify([scope.kind, scope.id]); }
export function allowed(record: { principalId: string; scope: Scope }, access: Access): boolean {
  return record.principalId === access.principalId && access.scopes.some(s => s.kind === record.scope.kind && s.id === record.scope.id);
}

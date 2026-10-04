export type Scope = { kind: 'user' | 'project' | 'workspace' | 'conversation'; id: string };
// These scopes must come from the authenticated application, never a model or request body.
export type Access = { principalId: string; scopes: readonly Scope[] };
export type Kind = 'preference' | 'constraint' | 'fact' | 'decision' | 'open_loop' | 'episode' | 'procedure';
export type Modality = 'desired' | 'observed' | 'reported' | 'hypothesis';
export type Status = 'active' | 'pending' | 'conflicted' | 'superseded' | 'stale';
export type SourceRef = { eventId: string; start: number; end: number };
export type Candidate = {
  start: number; end: number; kind: Kind; modality: Modality;
  subject?: string; predicate?: string; expiresAt?: number;
  validFrom?: number; validTo?: number;
  dependencies?: string[];
};
export type MemoryEvent = {
  id: string; principalId: string; scope: Scope;
  source: 'user' | 'tool' | 'document' | 'assistant';
  text: string; occurredAt: number;
  // A tool-specific adapter must establish verification. Transport success is insufficient.
  verified?: boolean;
  candidates?: Candidate[];
};
export type MemoryRecord = {
  id: string; principalId: string; scope: Scope; text: string;
  kind: Kind; modality: Modality; subject: string | null; predicate: string | null;
  support: 'user_explicit' | 'tool_verified' | 'source_reported' | 'tentative';
  status: Status; revision: number; recordedAt: number; retiredAt: number | null;
  observedAt: number; validFrom: number | null; validTo: number | null;
  expiresAt: number | null; supersedesId: string | null;
  sourceRefs: SourceRef[]; dependencies: string[];
};
export type SearchInput = Access & { query: string; limit?: number; validAt?: number; knownAt?: number };
export type SearchHit = { record: MemoryRecord; score: number; mandatory: boolean; truncatedMandatory?: boolean };
export type DecisionInput = {
  operation: 'retain' | 'relevance' | 'relationship';
  context: string; candidates: { id: string; text: string; previous?: string }[];
  deadlineMs: number;
};
export type DecisionResult = {
  model: string;
  decisions: { id: string; score: number; relation?: 'same' | 'contradicts' | 'unrelated' }[];
};
export interface DecisionProvider {
  readonly identity: string;
  decide(input: DecisionInput, signal: AbortSignal): Promise<DecisionResult>;
}
export interface Extractor {
  extract(event: MemoryEvent, signal: AbortSignal): Promise<Candidate[]>;
}
export type ContextPacket = {
  id: string;
  status: 'local' | 'jev' | 'jev_shadow' | 'local_fallback' | 'budget_exceeded';
  context: string; tokens: number; tokenCount: 'exact' | 'byte_upper_bound';
  selected: { id: string; revision: number; sources: SourceRef[]; score: number }[];
  conflicts: string[]; requiredTokens?: number; elapsedMs: number;
};
export type Receipt = Omit<ContextPacket, 'context'> & { scopes: readonly Scope[]; scopeRevision: string; model: string | null };
export type PrepareInput = Access & {
  request: string; tokenBudget?: number; deadlineMs?: number;
  baseContext?: string; validAt?: number; knownAt?: number;
};
export type MemoryOptions = {
  decisions?: DecisionProvider; extractor?: Extractor;
  tokenizer?: (text: string) => number; clock?: () => number;
  retentionThreshold?: number; relevanceThreshold?: number;
  maxCandidates?: number; remoteCandidates?: number; cacheSize?: number;
  decisionTimeoutMs?: number;
  semanticWrites?: 'shadow' | 'active';
  semanticReads?: 'shadow' | 'active';
  telemetry?: (event: { name: string; durationMs?: number; count?: number; status?: string }) => void;
};

#!/usr/bin/env node
import { readFileSync } from 'node:fs';
import { randomUUID } from 'node:crypto';
import { createMemory, HttpDecisionProvider, OpenJevApiProvider, SqliteStore } from './index.js';
import type { Scope } from './types.js';

const args = process.argv.slice(2);
function flag(name: string, fallback?: string): string | undefined {
  const index = args.indexOf(name);
  if (index < 0) return fallback;
  const value = args[index + 1];
  if (!value || value.startsWith('--')) throw new Error(`${name} requires a value`);
  args.splice(index, 2); return value;
}

async function main() {
  const path = flag('--db', process.env.JEV_MEMORY_DB ?? './memory.sqlite')!;
  const principalId = flag('--owner', process.env.JEV_MEMORY_OWNER);
  const tokenBudget = Number(flag('--budget', '1800'));
  const deadlineMs = Number(flag('--deadline', '350'));
  const active = args.includes('--active'); if (active) args.splice(args.indexOf('--active'), 1);
  const scopes: Scope[] = [];
  while (args.includes('--scope')) {
    const text = flag('--scope')!; const colon = text.indexOf(':');
    if (colon < 1) throw new Error('Scope must be kind:id');
    scopes.push({ kind: text.slice(0, colon) as Scope['kind'], id: text.slice(colon + 1) });
  }
  const json = args.includes('--json'); if (json) args.splice(args.indexOf('--json'), 1);
  const [command, ...rest] = args;
  if (!command || command === 'help' || command === '--help') {
    console.log('jev-memory <command> --db memory.sqlite --owner alice --scope project:app\n\nCommands: search <query>, context <request>, inspect <id>, record <event.json>,\n          drain, retry, correct <id> <text>, forget <id>, forget-source <id>,\n          explain <packet-id>, export, doctor\n\nUse --json for structured output. Repeat --scope for additional authorized scopes.\nContext: --budget 1800 --deadline 350. --active enables semantic decisions.\n\nProviders: JEV_MEMORY_URL + JEV_MEMORY_TOKEN + JEV_MEMORY_MODEL uses the package decision server.\n          OPENJEV_API_KEY + OPENJEV_URL (optional) calls native /v1/systemone directly.'); return;
  }
  if (!principalId && command !== 'doctor') throw new Error('Set --owner or JEV_MEMORY_OWNER');
  const access = { principalId: principalId ?? '_', scopes };
  const decisions = process.env.JEV_MEMORY_URL ? new HttpDecisionProvider({
    url: process.env.JEV_MEMORY_URL, token: process.env.JEV_MEMORY_TOKEN ?? '',
    identity: process.env.JEV_MEMORY_MODEL ?? '', allowInsecureLoopback: true,
  }) : (process.env.OPENJEV_API_KEY || process.env.JEV_API_KEY || process.env.OPENJEV_URL || process.env.JEV_API_URL)
    ? new OpenJevApiProvider({
        url: process.env.OPENJEV_URL ?? process.env.JEV_API_URL,
        apiKey: process.env.OPENJEV_API_KEY ?? process.env.JEV_API_KEY,
        model: process.env.OPENJEV_MODEL ?? 'openjev',
        identity: process.env.OPENJEV_MODEL ?? 'openjev',
        allowInsecureLoopback: true,
      })
    : undefined;
  const memory = createMemory({ store: new SqliteStore(path), decisions, decisionTimeoutMs: deadlineMs,
    semanticReads: active ? 'active' : 'shadow', semanticWrites: active ? 'active' : 'shadow' });
  try {
    let result: unknown;
    switch (command) {
      case 'doctor': result = memory.health(); break;
      case 'record': {
        if (!rest[0]) throw new Error('record requires an event JSON file');
        const event = JSON.parse(readFileSync(rest[0], 'utf8'));
        if (event.principalId !== access.principalId || !scopes.some(s => s.kind === event.scope?.kind && s.id === event.scope?.id)) throw new Error('Event is outside the supplied owner/scopes');
        memory.record(event); result = { queued: event.id }; break;
      }
      case 'drain': result = await memory.processPending({ principalId }); break;
      case 'retry': result = { requeued: memory.reconsider(access) }; break;
      case 'search': result = memory.search({ ...access, query: rest.join(' ') }); break;
      case 'context': result = await memory.prepare({ ...access, request: rest.join(' '), tokenBudget, deadlineMs }); break;
      case 'inspect': result = memory.get(access, rest[0] ?? ''); break;
      case 'explain': result = memory.explain(access, rest[0] ?? ''); break;
      case 'export': result = memory.export(access, rest[0] ?? '', 100); break;
      case 'forget': result = memory.forget(access, rest[0] ?? ''); break;
      case 'forget-source': result = memory.forgetSource(access, rest[0] ?? ''); break;
      case 'correct': {
        const old = memory.get(access, rest[0] ?? ''); if (!old) throw new Error('Memory not found');
        result = memory.correct(access, old.id, old.revision, { id: randomUUID(), principalId: access.principalId, scope: old.scope, source: 'user', text: rest.slice(1).join(' '), occurredAt: Date.now() }); break;
      }
      default: throw new Error('Unknown command; run jev-memory help');
    }
    if (!json && command === 'search') {
      const hits = result as ReturnType<typeof memory.search>;
      if (!hits.length) console.log('No matching memories.');
      for (const { record } of hits) console.log(`${record.id}  ${record.kind}  ${record.status}\n  ${record.text}\n  ${record.sourceRefs.length} source(s) · ${record.scope.kind}:${record.scope.id}\n`);
    } else if (!json && command === 'context') {
      const packet = result as Awaited<ReturnType<typeof memory.prepare>>;
      console.log(`${packet.selected.length} memories · ${packet.tokens} ${packet.tokenCount === 'exact' ? 'tokens' : 'token upper bound'} · ${packet.elapsedMs.toFixed(1)} ms · ${packet.status}\n` + packet.context);
    } else console.log(JSON.stringify(result, null, 2));
  } finally { memory.close(); }
}
main().catch(error => { console.error(`jev-memory: ${error instanceof Error ? error.message : 'Operation failed'}`); process.exitCode = 1; });

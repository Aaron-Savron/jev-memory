import { performance } from 'node:perf_hooks';
import { createMemory, SqliteStore } from '../dist/index.js';

const memory = createMemory({ store: new SqliteStore() });
const scope = { kind: 'project', id: 'bench' };
const access = { principalId: 'bench-user', scopes: [scope] };
for (let i = 0; i < 10000; i++) {
  const text = `Component ${i} uses certificate rotation every ${30 + i % 5} days.`;
  memory.remember({ id: String(i), principalId: access.principalId, scope, source: 'user', text, occurredAt: 1000,
    candidates: [{ start: 0, end: text.length, kind: 'fact', modality: 'reported' }] });
  if ((i+1) % 1000 === 0) console.error(`Loaded ${i+1} records`);
}
const timings = [];
for (let i = 0; i < 110; i++) {
  const start = performance.now();
  await memory.prepare({ ...access, request: `certificate rotation component ${i}`, tokenBudget: 1800 });
  if (i >= 10) timings.push(performance.now() - start);
}
timings.sort((a, b) => a - b);
console.log(JSON.stringify({ kind: 'local-memory-only', node: process.version, records: 10000,
  iterations: timings.length, p50Ms: timings[49], p95Ms: timings[94], p99Ms: timings[98],
  storage: 'in-memory SQLite; includes receipt writes; excludes model inference and agent quality' }, null, 2));
memory.close();

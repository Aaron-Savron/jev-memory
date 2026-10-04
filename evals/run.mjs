// Mechanical multi-session invariants, not an LLM task-success benchmark.
import assert from 'node:assert/strict';
import { createMemory, SqliteStore } from '../dist/index.js';

let checks = 0;
for (let i = 0; i < 40; i++) {
  const m = createMemory({ store: new SqliteStore() });
  const userScope = { kind: 'user', id: `user-${i}` };
  const scope = { kind: 'project', id: `project-${i}` };
  const access = { principalId: `user-${i}`, scopes: [userScope, scope] };
  function save(id, text, target = scope, candidate = {}) {
    m.remember({ id, principalId: access.principalId, scope: target, source: 'user', text, occurredAt: Date.now(),
      candidates: [{ start: 0, end: text.length, kind: 'fact', modality: 'reported', ...candidate }] });
  }
  save('pref', 'Use concise documentation.', userScope, { kind: 'preference', modality: 'desired' });
  save('deploy', 'The deployment uses Docker.');
  const old = m.search({ ...access, query: 'deployment' }).find(h => h.record.kind === 'fact').record;
  save('procedure', 'Build a Docker container before deployment.', scope, { kind: 'procedure', dependencies: [old.id] });
  for (let j = 0; j < 12; j++) save(`noise-${j}`, `Unrelated conversation topic ${j}.`);
  const packet = await m.prepare({ ...access, request: 'deployment', tokenBudget: 4000 });
  assert.ok(packet.selected.some(s => s.id === old.id)); checks++;
  assert.ok(packet.context.includes('concise documentation')); checks++;
  m.correct(access, old.id, old.revision, { id: 'fix', principalId: access.principalId, scope, source: 'user', text: 'The deployment uses systemd.', occurredAt: Date.now() });
  const next = await m.prepare({ ...access, request: 'deployment', tokenBudget: 4000 });
  assert.ok(next.context.includes('systemd') && !next.context.includes('Docker')); checks++;
  assert.equal((await m.prepare({ principalId: 'other-user', scopes: [scope], request: 'deployment' })).selected.length, 0); checks++;
  m.forgetSource(access, 'fix');
  assert.ok(!(await m.prepare({ ...access, request: 'deployment', tokenBudget: 4000 })).context.includes('systemd')); checks++;
  m.close();
}
console.log(JSON.stringify({ narratives: 40, invariantChecks: checks, passed: checks,
  model: 'none', purpose: 'scope, older recall, preferences, correction propagation, source deletion',
  limitation: 'Synthetic mechanical checks; not Jev quality or agent task success.' }, null, 2));

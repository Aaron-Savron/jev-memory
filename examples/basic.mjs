import { createMemory, SqliteStore } from '../dist/index.js';

const memory = createMemory({ store: new SqliteStore() });
const scope = { kind: 'project', id: 'demo' };
const access = { principalId: 'demo-user', scopes: [scope] };
function saved(id, text, options = {}) {
  return { id, principalId: access.principalId, scope, source: 'user', text, occurredAt: Date.now(),
    candidates: [{ start: 0, end: text.length, kind: 'fact', modality: 'reported', ...options }] };
}
memory.remember(saved('deployment', 'Deploy this app through Docker.', { subject: 'app', predicate: 'deployment' }));
const old = memory.search({ ...access, query: 'deploy' })[0].record;
memory.remember(saved('procedure', 'Build the container before deployment.', { kind: 'procedure', dependencies: [old.id] }));
memory.correct(access, old.id, old.revision, saved('correction', 'Deploy this app through systemd.', { subject: 'app', predicate: 'deployment' }));
console.log('Synthetic demo: a correction retires Docker and invalidates its dependent procedure.');
const packet = await memory.prepare({ ...access, request: 'How should we deploy?', tokenBudget: 2000 });
console.log(packet.context);
console.log(memory.export(access).records.map(r => ({ text: r.text, status: r.status })));
memory.close();

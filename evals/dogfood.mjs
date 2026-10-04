import { createMemory, JevApiProvider, SqliteStore } from '../dist/index.js';

const access = { principalId: 'dogfood-user', scopes: [{ kind: 'project', id: 'cofound' }] };
const scope = access.scopes[0];

// Run against a real Jev API when credentials are supplied. The default local
// provider keeps this smoke test deterministic and free to run in CI.
const decisions = process.env.OPENJEV_API_KEY
  ? new JevApiProvider({
      url: process.env.OPENJEV_URL ?? 'https://api.openjev.sh',
      apiKey: process.env.OPENJEV_API_KEY,
      model: process.env.OPENJEV_MODEL ?? 'openjev',
    })
  : {
      identity: 'dogfood-local-jev',
      async decide(input) {
        const contextWords = new Set(input.context.toLowerCase().split(/[^a-z0-9]+/).filter(w => w.length > 3));
        return {
          model: 'dogfood-local-jev',
          decisions: input.candidates.map(candidate => {
            if (input.operation === 'relationship') {
              const oldText = candidate.previous?.toLowerCase() ?? '';
              const newText = candidate.text.toLowerCase();
              const contradicts = (oldText.includes('systemd') && newText.includes('docker')) || (oldText.includes('docker') && newText.includes('systemd'));
              return { id: candidate.id, score: contradicts ? 0.98 : 0.95, relation: contradicts ? 'contradicts' : 'same' };
            }
            const words = candidate.text.toLowerCase().split(/[^a-z0-9]+/).filter(w => w.length > 3);
            const overlap = words.filter(word => contextWords.has(word)).length;
            const durable = /systemd|docker|database_url|runnable|concise|blocked|restart/i.test(candidate.text);
            return { id: candidate.id, score: durable && (input.operation === 'retain' || overlap > 0) ? 0.97 : 0.08 };
          }),
        };
      },
    };

const memory = createMemory({
  store: new SqliteStore(),
  decisions,
  semanticReads: 'active',
  semanticWrites: 'active',
  decisionTimeoutMs: 1000,
});

function record(id, source, text, extra = {}) {
  memory.record({ id, principalId: access.principalId, scope, source, text, occurredAt: Date.now(), ...extra });
}

try {
  record('chat-1', 'user', 'The staging app runs under systemd on the Atlas host. Restart it with systemctl restart cofound.');
  record('chat-2', 'user', 'I prefer concise documentation with runnable examples.');
  record('tool-1', 'tool', 'The deployment check failed because staging is missing DATABASE_URL.', { verified: true });
  record('assistant-1', 'assistant', 'Deployment succeeded and everything is healthy.');

  const drained = await memory.processPending({ principalId: access.principalId });
  const before = await memory.prepare({ ...access, request: 'Fix the staging deployment', tokenBudget: 2200, deadlineMs: 1000 });
  const beforeRecords = memory.search({ ...access, query: 'staging deployment' }).map(hit => hit.record.text);

  const systemd = memory.search({ ...access, query: 'systemd' }).find(hit => hit.record.text.includes('systemd'))?.record;
  let correction;
  if (systemd) {
    correction = memory.correct(access, systemd.id, systemd.revision, {
      id: 'correction-1', principalId: access.principalId, scope, source: 'user',
      text: 'The staging app runs in Docker on the Atlas host.', occurredAt: Date.now(),
    });
  }
  const afterCorrection = await memory.prepare({ ...access, request: 'Fix the staging deployment', tokenBudget: 2200, deadlineMs: 1000 });
  memory.forgetSource(access, 'tool-1');
  const afterForget = memory.search({ ...access, query: 'DATABASE_URL' }).map(hit => hit.record.text).filter(text => text.includes('DATABASE_URL'));
  const assistantClaimStored = memory.export(access).records.some(record => record.text.includes('everything is healthy'));

  console.log(JSON.stringify({
    provider: decisions.identity,
    drained,
    before: { status: before.status, selected: before.selected.length, usefulContext: before.context, matchingRecords: beforeRecords },
    correction: correction ? { text: correction.text, supersedesId: correction.supersedesId } : null,
    afterCorrection: { status: afterCorrection.status, usefulContext: afterCorrection.context },
    afterForget: { databaseUrlMatches: afterForget },
    assistantClaimStored,
  }, null, 2));
} finally {
  memory.close();
}

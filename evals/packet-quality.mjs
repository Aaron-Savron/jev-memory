// Local packet quality: retrieval and curation without a semantic model.
// Measures whether prepare() surfaces the right records for a request and
// keeps the wrong ones out. Gold sets are per-narrative.
import assert from 'node:assert/strict';
import { createMemory, SqliteStore } from '../dist/index.js';

const project = { kind: 'project', id: 'atlas' };
const user = { kind: 'user', id: 'alice' };
const access = { principalId: 'alice', scopes: [user, project] };

function seed(narrative) {
  const m = createMemory({ store: new SqliteStore(), semanticReads: 'active', semanticWrites: 'active' });
  for (const item of narrative.records) {
    m.remember({
      id: item.id,
      principalId: 'alice',
      scope: item.scope === 'user' ? user : project,
      source: item.source ?? 'user',
      ...(item.verified === undefined ? {} : { verified: item.verified }),
      text: item.text,
      occurredAt: item.at,
      candidates: [{
        start: 0,
        end: item.text.length,
        kind: item.kind ?? 'fact',
        modality: item.modality ?? 'reported',
        ...(item.subject ? { subject: item.subject } : {}),
        ...(item.predicate ? { predicate: item.predicate } : {}),
      }],
    });
  }
  return m;
}

const narratives = [
  {
    name: 'staging deploy fix',
    records: [
      { id: 'sys', text: 'The staging app runs under systemd on the Atlas host.', at: 100 },
      { id: 'url', text: 'The deploy failed because staging is missing DATABASE_URL.', at: 110 },
      { id: 'tea', text: 'I generally prefer green tea.', at: 120, scope: 'user', kind: 'preference', modality: 'desired' },
      { id: 'desk', text: 'The user bought a new standing desk.', at: 130 },
      { id: 'movie', text: 'A movie character mentioned deployment once.', at: 140 },
      { id: 'port', text: 'API responses are cached for 60 seconds.', at: 150 },
    ],
    request: 'Fix the staging deployment',
    mustInclude: ['sys', 'url'],
    mustExclude: ['tea', 'desk', 'port'],
    // "movie" shares the token "deployment" with the request. Local FTS cannot
    // tell a topical sentence from a useful fact; semantic scoring is what drops it.
    expectNoiseWithoutModel: ['movie'],
  },
  {
    name: 'api docs preference',
    records: [
      { id: 'doc-user', text: 'Keep documentation short.', at: 100, scope: 'user', kind: 'preference', modality: 'desired', subject: 'user', predicate: 'documentation' },
      { id: 'doc-proj', text: 'Write exhaustive API references with runnable examples.', at: 110, kind: 'preference', modality: 'desired', subject: 'user', predicate: 'documentation' },
      { id: 'sdk', text: 'The public SDK API is TypeScript and publishes as @acme/sdk.', at: 120 },
      { id: 'tea', text: 'I generally prefer green tea.', at: 130, scope: 'user', kind: 'preference', modality: 'desired' },
    ],
    request: 'Write the API documentation',
    mustInclude: ['doc-proj', 'sdk'],
    mustExclude: ['doc-user', 'tea'],
  },
  {
    name: 'migration blockers',
    records: [
      { id: 'backup', text: 'A verified backup from the last 24 hours is required before the migration.', at: 100 },
      { id: 'target', text: 'The migration target is PostgreSQL; production still runs SQLite.', at: 110 },
      { id: 'tea', text: 'I generally prefer green tea.', at: 120, scope: 'user', kind: 'preference', modality: 'desired' },
      { id: 'elephant', text: "PostgreSQL's mascot is an elephant named Slonik.", at: 130 },
    ],
    request: 'Continue the database migration',
    mustInclude: ['backup', 'target'],
    mustExclude: ['tea', 'elephant'],
  },
  {
    name: 'correction propagates',
    records: [
      { id: 'old', text: 'The staging app runs in Docker on the Atlas host.', at: 100 },
    ],
    request: 'How does staging deploy with Docker?',
    mustInclude: ['old'],
    mustExclude: [],
    mutate: (m) => {
      const old = m.search({ ...access, query: 'staging Docker' }).find(h => h.record.text.includes('Docker'))?.record;
      if (!old) throw new Error('missing old record');
      m.correct(access, old.id, old.revision, {
        id: 'fix', principalId: 'alice', scope: project, source: 'user',
        text: 'The staging app runs under systemd on the Atlas host.', occurredAt: 200,
        candidates: [{ start: 0, end: 48, kind: 'fact', modality: 'reported' }],
      });
    },
    afterRequest: 'How does staging deploy under systemd?',
    afterMustInclude: ['systemd'],
    afterMustExclude: ['Docker'],
  },
  {
    name: 'source deletion removes claim',
    records: [
      { id: 'tool-outcome', text: 'The deploy succeeded because DATABASE_URL was set.', at: 100, source: 'tool', verified: true, kind: 'episode', modality: 'observed' },
      { id: 'keep', text: 'The deploy checklist requires restarting systemd after release.', at: 110 },
    ],
    request: 'What did the deploy DATABASE_URL check show?',
    mustInclude: ['tool-outcome', 'keep'],
    mutate: (m) => { m.forgetSource(access, 'tool-outcome'); },
    afterRequest: 'What did the deploy DATABASE_URL check show?',
    afterMustInclude: ['keep'],
    afterMustExclude: ['DATABASE_URL was set'],
  },
];

let checks = 0;
const failures = [];
for (const narrative of narratives) {
  const m = seed(narrative);
  try {
    const packet = await m.prepare({ ...access, request: narrative.request, tokenBudget: 4000 });
    for (const id of narrative.mustInclude ?? []) {
      checks++;
      const ok = packet.context.includes(id) || packet.selected.some(s => s.id === id)
        || packet.context.includes(seedText(narrative, id));
      if (!ok) failures.push(`${narrative.name}: missing ${id}`);
    }
    for (const id of narrative.mustExclude ?? []) {
      checks++;
      const text = seedText(narrative, id);
      if (packet.context.includes(text)) failures.push(`${narrative.name}: leaked ${id}`);
    }
    for (const id of narrative.expectNoiseWithoutModel ?? []) {
      checks++;
      const text = seedText(narrative, id);
      if (!packet.context.includes(text)) failures.push(`${narrative.name}: expected model-less noise missing ${id}`);
    }
    if (narrative.mutate) narrative.mutate(m);
    if (narrative.afterRequest) {
      const after = await m.prepare({ ...access, request: narrative.afterRequest, tokenBudget: 4000 });
      for (const needle of narrative.afterMustInclude ?? []) {
        checks++;
        if (!after.context.includes(needle)) failures.push(`${narrative.name}: after-missing ${needle}`);
      }
      for (const needle of narrative.afterMustExclude ?? []) {
        checks++;
        if (after.context.includes(needle)) failures.push(`${narrative.name}: after-leaked ${needle}`);
      }
    }
  } finally {
    m.close();
  }
}

function seedText(narrative, id) {
  return narrative.records.find(r => r.id === id)?.text ?? id;
}

const passed = checks - failures.length;
const result = {
  narratives: narratives.length,
  checks,
  passed,
  failed: failures.length,
  failures,
  purpose: 'local retrieval and curation quality without a semantic model',
  limitation: 'Gold sets are hand-written per narrative. No agent task-success claim.',
};
console.log(JSON.stringify(result, null, 2));
if (failures.length) process.exitCode = 1;

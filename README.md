# Jev Memory

Source-backed memory for agent harnesses. SQLite keeps the records locally. A self-hosted decision server (Qwen3-0.6B + LoRA, trained on a broad multi-domain memory corpus) decides which source spans are worth retaining and which memories help the current task.

Corrections retire old versions and invalidate dependent procedures. User preferences, project facts, conversation state, and tool outcomes have explicit scopes. A request to do something is different from evidence that it happened.

Requires Node 22.13 or later. No runtime npm dependencies or mandatory vector database.

## Install

The package is currently built as a tarball, not published to npm:

```sh
npm ci
npm test
npm pack
# In your application:
npm install ./svrn-jev-memory-0.1.0.tgz
```

## Use

```ts
import { createMemory, SqliteStore, HttpDecisionProvider } from '@svrn/jev-memory';

const memory = createMemory({
  store: new SqliteStore('./memory.sqlite'),
  decisions: new HttpDecisionProvider({
    url: process.env.JEV_MEMORY_URL!,
    token: process.env.JEV_MEMORY_TOKEN!,
    identity: process.env.JEV_MEMORY_MODEL!,
  }),
  semanticReads: 'shadow',
  semanticWrites: 'shadow',
});

const access = {
  principalId: 'alice',
  scopes: [{ kind: 'project' as const, id: 'app' }],
};

memory.record({
  id: 'message-123', principalId: 'alice', scope: access.scopes[0],
  source: 'user', text: 'We chose PostgreSQL, but have not migrated yet.',
  occurredAt: Date.now(),
});

// Run in a worker. Jobs survive restarts and process 16 candidates at a time.
await memory.processPending({ principalId: 'alice', limit: 16 });

const packet = await memory.prepare({
  ...access, request: 'Continue the database migration',
  tokenBudget: 1800, deadlineMs: 350,
});
// Handle budget_exceeded explicitly, then add packet.context as quoted context.
```

## Use a Jev API directly

If you have a hosted Jev key or a native Open-Jev server, use the `/v1/systemone` provider. The same code works for both by changing the URL and key. See the [Open-Jev API docs](https://openjev.sh/docs) for the native request shape.

```ts
import { createMemory, OpenJevApiProvider, SqliteStore } from '@svrn/jev-memory';

const memory = createMemory({
  store: new SqliteStore('./memory.sqlite'),
  decisions: new OpenJevApiProvider({
    url: process.env.OPENJEV_URL ?? 'https://api.openjev.sh',
    apiKey: process.env.OPENJEV_API_KEY,
    model: process.env.OPENJEV_MODEL ?? 'openjev',
  }),
});
```

For the CLI, set `OPENJEV_API_KEY` and optionally `OPENJEV_URL` and `OPENJEV_MODEL`. Use `JEV_MEMORY_URL`, `JEV_MEMORY_TOKEN`, and `JEV_MEMORY_MODEL` when calling this package's authenticated decision server instead. Remote URLs must use HTTPS. Loopback HTTP is allowed when explicitly enabled for local development.

Without a decision provider, explicit supported memories and local retrieval work. Unstructured proposals remain pending. Supply your provider's tokenizer for exact token counts; the fallback uses UTF-8 bytes as a conservative token bound.

Semantic reads and writes default to **shadow**. Reads preserve local ranking; writes remain pending. Set either to `active` after evaluating that model and policy on your workload. The default active thresholds are 0.6 for relevance and 0.9 for retention. These scores are not guarantees of correctness.

`remember(event)` saves up to 16 explicitly approved source spans synchronously. Use it for a user's Save Memory action or a verified tool adapter. `correct(access, id, revision, event)` applies an authorized user correction immediately. Tool adapters must establish what a result proves; exit status alone does not prove task completion.

## Runtime behavior

- Search uses authorized FTS postings and bounded candidates. Applicable constraints bypass semantic scoring.
- Preferences are curated: only query-matching preferences join a task packet, so an unrelated personal preference does not crowd every prompt. Constraints always apply.
- Preferences can override matching general preferences in a more specific scope when subject/predicate keys are supplied. A more specific preference masks its broader counterpart even when only the broader text matched the query.
- `validAt` filters supplied validity intervals. `knownAt` selects the versions available at that recording time. Unknown dates remain unknown.
- `revalidate(access, packet)` checks revisions, expiry, and dependencies before a harness uses remembered procedures.
- Scoring has a deadline, a circuit breaker, and bounded caches. Outages use local retrieval. Late responses cannot restore deleted records.
- Duplicate events are idempotent. Concurrent writes use leases and scope revisions. Failed jobs have bounded retries and a visible dead state.
- `reconsider(access)` requeues dead jobs and pending shadow proposals after a model/policy change.

The host supplies authenticated principals and allowed scopes. This library is not an authentication service. Scopes cannot come from model output. Stable project IDs, workspace aliases, and branch identities belong to the host adapter.

For atomic application writes, execute `memory.eventCommand(event)` in the application's completion transaction. A trigger creates its job in the same transaction. Do not await inference inside a database transaction.

## Inspect and delete

```sh
jev-memory search 'deployment' --db memory.sqlite --owner alice --scope project:app
jev-memory context 'Deploy the app' --db memory.sqlite --owner alice --scope project:app
jev-memory doctor --db memory.sqlite
```

The CLI also supports `record`, `drain`, `retry`, `inspect`, `correct`, `explain`, `export`, `forget`, and `forget-source`. Use `--json` for scripts. Context accepts `--budget`, `--deadline`, and `--active`.

`explain` reads a stored retrieval receipt. `export` pages records and source references; it is not a full database backup. `forget` uses source-first deletion: it also removes the supporting events and other claims supported only by those events. Independent support can preserve another claim. Logical deletion is immediate; WAL pages, backups, and prior exports need separate lifecycle management.

The store applies owner-only file permissions. Encryption belongs to the host or an encrypted filesystem. Redact domain-specific secrets before recording; common credential patterns are rejected.

## Server and validation

See [server/README.md](server/README.md) for Docker, Modal T4, and the experimental memory head. See [VALIDATION.md](VALIDATION.md) for measured results and limits.

```sh
npm test
npm run eval
npm run dogfood
npm run bench
node examples/basic.mjs
```

`npm run dogfood` runs a small end-to-end conversation through ingestion, retention, context preparation, correction, and source deletion. It uses a deterministic local provider by default. Set `OPENJEV_API_KEY` to run the same flow against the native Jev API.

The engine supports extractive candidates and an optional background extractor. It does not automatically watch repository files, infer stable project identities, train on private conversations, promote successful workflows into procedures, or sync devices. A harness can supply verified spans, dependencies, and lifecycle events through the API.

Code: MIT. Downloaded Open-Jev/Qwen weights and the adapted head: Apache-2.0 under their included notices.

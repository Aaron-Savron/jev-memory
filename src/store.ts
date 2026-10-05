import { DatabaseSync } from 'node:sqlite';
import { chmodSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { randomUUID } from 'node:crypto';
import type { Access, Candidate, MemoryEvent, MemoryRecord, Receipt, SearchHit, SearchInput } from './types.js';
import { allowed, assertAccess, assertId, containsSecret, finiteTime, hash, scopeKey, validateEvent } from './validation.js';

type Row = Record<string, unknown>;
export type SqlCommand = { sql: string; params: (string | number | null)[] };
export type Job = { event: MemoryEvent; token: string; attempts: number; watermark: number; progress: number; prepared?: Candidate[] };

const SCHEMA = `
CREATE TABLE IF NOT EXISTS jmem_meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
INSERT OR IGNORE INTO jmem_meta VALUES ('schema', 1);
CREATE TABLE IF NOT EXISTS jmem_events (
 principal TEXT NOT NULL, id TEXT NOT NULL, scope TEXT NOT NULL, body TEXT NOT NULL,
 digest TEXT NOT NULL, recorded_at INTEGER NOT NULL, PRIMARY KEY(principal,id)
);
CREATE TABLE IF NOT EXISTS jmem_revocations (
 principal TEXT NOT NULL, id TEXT NOT NULL, at INTEGER NOT NULL, PRIMARY KEY(principal,id)
);
CREATE TABLE IF NOT EXISTS jmem_scopes (
 principal TEXT NOT NULL, scope TEXT NOT NULL, revision INTEGER NOT NULL, PRIMARY KEY(principal,scope)
);
CREATE TABLE IF NOT EXISTS jmem_jobs (
 principal TEXT NOT NULL, event_id TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
 attempts INTEGER NOT NULL DEFAULT 0, available_at INTEGER NOT NULL DEFAULT 0,
 lease_until INTEGER NOT NULL DEFAULT 0, token TEXT, error TEXT,
 progress INTEGER NOT NULL DEFAULT 0, prepared TEXT,
 PRIMARY KEY(principal,event_id),
 FOREIGN KEY(principal,event_id) REFERENCES jmem_events(principal,id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS jmem_job_ready ON jmem_jobs(status,available_at,lease_until);
CREATE TABLE IF NOT EXISTS jmem_records (
 id TEXT PRIMARY KEY, principal TEXT NOT NULL, scope TEXT NOT NULL, body TEXT NOT NULL,
 status TEXT NOT NULL, revision INTEGER NOT NULL, recorded_at INTEGER NOT NULL,
 retired_at INTEGER, expires_at INTEGER, valid_from INTEGER, valid_to INTEGER,
 fingerprint TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS jmem_record_scope ON jmem_records(principal,scope,status,recorded_at);
CREATE INDEX IF NOT EXISTS jmem_record_fingerprint ON jmem_records(principal,scope,fingerprint,status);
CREATE INDEX IF NOT EXISTS jmem_record_kind ON jmem_records(principal,scope,json_extract(body,'$.kind'),status);
CREATE TABLE IF NOT EXISTS jmem_sources (
 record_id TEXT NOT NULL REFERENCES jmem_records(id) ON DELETE CASCADE,
 principal TEXT NOT NULL, event_id TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL,
 PRIMARY KEY(record_id,principal,event_id,start,end),
 FOREIGN KEY(principal,event_id) REFERENCES jmem_events(principal,id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS jmem_sources_event ON jmem_sources(principal,event_id);
CREATE TABLE IF NOT EXISTS jmem_dependencies (
 record_id TEXT NOT NULL REFERENCES jmem_records(id) ON DELETE CASCADE,
 depends_on TEXT NOT NULL REFERENCES jmem_records(id) ON DELETE CASCADE,
 PRIMARY KEY(record_id,depends_on)
);
CREATE INDEX IF NOT EXISTS jmem_dependency_target ON jmem_dependencies(depends_on);
CREATE VIRTUAL TABLE IF NOT EXISTS jmem_fts USING fts5(id UNINDEXED,text,scope_token,tokenize='unicode61');
CREATE TABLE IF NOT EXISTS jmem_receipts (
 id TEXT PRIMARY KEY, principal TEXT NOT NULL, body TEXT NOT NULL, at INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS jmem_receipt_owner ON jmem_receipts(principal,at);
CREATE TABLE IF NOT EXISTS jmem_decisions (
 key TEXT PRIMARY KEY, principal TEXT NOT NULL, scope TEXT NOT NULL,
 model TEXT NOT NULL, body TEXT NOT NULL, expires_at INTEGER NOT NULL
);
CREATE TRIGGER IF NOT EXISTS jmem_event_revoked BEFORE INSERT ON jmem_events
 WHEN EXISTS(SELECT 1 FROM jmem_revocations WHERE principal=NEW.principal AND id=NEW.id)
 BEGIN SELECT RAISE(ABORT,'Source revoked'); END;
CREATE TRIGGER IF NOT EXISTS jmem_event_mismatch BEFORE INSERT ON jmem_events
 WHEN EXISTS(SELECT 1 FROM jmem_events WHERE principal=NEW.principal AND id=NEW.id AND digest<>NEW.digest)
 BEGIN SELECT RAISE(ABORT,'Event ID reused with different content'); END;
CREATE TRIGGER IF NOT EXISTS jmem_event_job AFTER INSERT ON jmem_events
 BEGIN INSERT INTO jmem_jobs(principal,event_id) VALUES(NEW.principal,NEW.id); END;
`;

export class SqliteStore {
  readonly db: DatabaseSync;
  readonly path: string;
  readonly clock: () => number;
  #closed = false;
  #statements = new Map<string, ReturnType<DatabaseSync["prepare"]>>();

  constructor(path = ':memory:', clock: () => number = Date.now) {
    this.path = path === ':memory:' ? path : resolve(path);
    this.clock = clock;
    if (path !== ':memory:') mkdirSync(dirname(this.path), { recursive: true, mode: 0o700 });
    this.db = new DatabaseSync(this.path);
    if (path !== ':memory:' && process.platform !== 'win32') chmodSync(this.path, 0o600);
    this.db.exec('PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000; PRAGMA journal_mode=WAL; PRAGMA synchronous=FULL;');
    this.transaction(() => {
      this.db.exec(SCHEMA);
      const version = this.#statement("SELECT value FROM jmem_meta WHERE key='schema'").get() as Row;
      if (version.value !== 1) throw new Error('Unsupported memory schema version');
    });
    if (path !== ':memory:' && process.platform !== 'win32') {
      for (const suffix of ['-wal', '-shm']) {
        try { chmodSync(this.path + suffix, 0o600); } catch (e) {
          if ((e as NodeJS.ErrnoException).code !== 'ENOENT') throw e;
        }
      }
    }
  }

  #statement(sql: string): ReturnType<DatabaseSync["prepare"]> {
    let statement = this.#statements.get(sql);
    if (!statement) {
      statement = this.db.prepare(sql);
      this.#statements.set(sql, statement);
      if (this.#statements.size > 256) this.#statements.delete(this.#statements.keys().next().value!);
    }
    return statement;
  }

  transaction<T>(fn: () => T): T {
    if (this.#closed) throw new Error('Memory store closed');
    this.db.exec('BEGIN IMMEDIATE');
    try {
      const result = fn();
      if (result && typeof (result as { then?: unknown }).then === 'function') throw new TypeError('Transactions must be synchronous');
      this.db.exec('COMMIT'); return result;
    } catch (e) { this.db.exec('ROLLBACK'); throw e; }
  }

  // Execute this command in the application's completion transaction. The trigger
  // creates its outbox job atomically. Never assemble event SQL yourself.
  eventCommand(input: MemoryEvent): SqlCommand {
    const event = validateEvent(input);
    if (containsSecret(event.text)) throw new TypeError('Redact credentials before recording memory');
    return {
      sql: 'INSERT INTO jmem_events(principal,id,scope,body,digest,recorded_at) VALUES(?,?,?,?,?,?) ON CONFLICT(principal,id) DO NOTHING',
      params: [event.principalId, event.id, scopeKey(event.scope), JSON.stringify(event), hash(event), this.clock()],
    };
  }

  record(event: MemoryEvent): void {
    const command = this.eventCommand(event);
    this.transaction(() => this.#statement(command.sql).run(...command.params));
  }

  watermark(principal: string, scope: MemoryEvent['scope']): number {
    return Number(this.#statement('SELECT revision FROM jmem_scopes WHERE principal=? AND scope=?').get(principal, scopeKey(scope))?.revision ?? 0);
  }

  revision(access: Access): string {
    assertAccess(access);
    return hash(access.scopes.map(s => [scopeKey(s), this.watermark(access.principalId, s)]).sort());
  }

  #bump(principal: string, scope: MemoryEvent['scope']): void {
    this.#statement('INSERT INTO jmem_scopes VALUES(?,?,1) ON CONFLICT(principal,scope) DO UPDATE SET revision=revision+1').run(principal, scopeKey(scope));
    this.#statement('DELETE FROM jmem_decisions WHERE principal=? AND scope=?').run(principal, scopeKey(scope));
  }

  claim(principalId: string | undefined, leaseMs = 30000, eventId?: string): Job | null {
    if (principalId !== undefined) assertId(principalId);
    return this.transaction(() => {
      const now = this.clock();
      this.#statement("UPDATE jmem_jobs SET status='dead',token=NULL,error='lease_exhausted' WHERE status='running' AND attempts>=5 AND lease_until<=?").run(now);
      const row = this.#statement(`SELECT j.*,e.body FROM jmem_jobs j JOIN jmem_events e ON e.principal=j.principal AND e.id=j.event_id
        WHERE j.status IN ('pending','running') AND j.available_at<=? AND j.lease_until<=? AND j.attempts<5
        ${principalId ? 'AND j.principal=?' : ''} ${eventId ? 'AND j.event_id=?' : ''} ORDER BY e.recorded_at LIMIT 1`).get(now, now, ...(principalId ? [principalId] : []), ...(eventId ? [eventId] : [])) as Row | undefined;
      if (!row) return null;
      const token = randomUUID();
      this.#statement("UPDATE jmem_jobs SET status='running',attempts=attempts+1,lease_until=?,token=?,error=NULL WHERE principal=? AND event_id=?").run(now + leaseMs, token, String(row.principal), String(row.event_id));
      const event = JSON.parse(String(row.body)) as MemoryEvent;
      return { event, token, attempts: Number(row.attempts) + 1, watermark: this.watermark(event.principalId, event.scope), progress: Number(row.progress), ...(row.prepared ? { prepared: JSON.parse(String(row.prepared)) as Candidate[] } : {}) };
    });
  }

  retry(job: Job, reason: 'inference_unavailable' | 'version_changed' | 'invalid_candidate'): void {
    const dead = job.attempts >= 5;
    this.#statement('UPDATE jmem_jobs SET status=?,lease_until=0,available_at=?,error=? WHERE principal=? AND event_id=? AND token=?').run(dead ? 'dead' : 'pending', this.clock() + Math.min(60000, 1000 * 4 ** (job.attempts - 1)), reason, job.event.principalId, job.event.id, job.token);
  }

  finish(job: Job, records: MemoryRecord[], cursor?: { candidates: Candidate[]; next: number; complete: boolean }): boolean {
    return this.transaction(() => {
      const e = job.event;
      const row = this.#statement("SELECT token,lease_until FROM jmem_jobs WHERE principal=? AND event_id=? AND status='running'").get(e.principalId, e.id);
      if (!row || row.token !== job.token || Number(row.lease_until) <= this.clock()) return false;
      if (this.watermark(e.principalId, e.scope) !== job.watermark) return false;
      for (const record of records) {
        if (record.principalId !== e.principalId || scopeKey(record.scope) !== scopeKey(e.scope)) throw new Error('Proposal scope mismatch');
        this.#put(record);
      }
      this.#statement('UPDATE jmem_jobs SET status=?,lease_until=0,token=NULL,error=NULL,attempts=0,progress=?,prepared=? WHERE principal=? AND event_id=? AND token=?').run(cursor && !cursor.complete ? 'pending' : 'done', cursor?.next ?? job.progress, cursor ? JSON.stringify(cursor.candidates) : null, e.principalId, e.id, job.token);
      if (records.length) this.#bump(e.principalId, e.scope);
      return true;
    });
  }

  #put(record: MemoryRecord): void {
    for (const ref of record.sourceRefs) {
      const source = this.#statement('SELECT body FROM jmem_events WHERE principal=? AND id=?').get(record.principalId, ref.eventId);
      if (!source) throw new Error('Missing or revoked source');
      const event = JSON.parse(String(source.body)) as MemoryEvent;
      if (scopeKey(event.scope) !== scopeKey(record.scope) || event.text.slice(ref.start, ref.end) !== record.text) throw new Error('Unsupported memory span');
    }
    if (!record.sourceRefs.length) throw new Error('Missing source support');
    for (const id of record.dependencies) {
      const dependency = this.#raw(id);
      if (!dependency || dependency.principalId !== record.principalId || scopeKey(dependency.scope) !== scopeKey(record.scope) || dependency.status !== 'active') throw new Error('Dependency must be active in the same private scope');
    }
    const fingerprint = hash([record.text, record.kind, record.modality, record.support, record.subject, record.predicate, record.validFrom, record.validTo, record.expiresAt, [...record.dependencies].sort()]);
    const duplicate = record.supersedesId ? undefined : this.#statement("SELECT id FROM jmem_records WHERE principal=? AND scope=? AND fingerprint=? AND status='active' LIMIT 1").get(record.principalId, scopeKey(record.scope), fingerprint);
    const targetId = duplicate ? String(duplicate.id) : record.id;
    if (!duplicate) {
      const inserted = this.#statement('INSERT INTO jmem_records VALUES(?,?,?,?,?,?,?,?,?,?,?,?)').run(record.id, record.principalId, scopeKey(record.scope), JSON.stringify(record), record.status, record.revision, record.recordedAt, record.retiredAt, record.expiresAt, record.validFrom, record.validTo, fingerprint);
      this.#statement('INSERT INTO jmem_fts(rowid,id,text,scope_token) VALUES(?,?,?,?)').run(inserted.lastInsertRowid, record.id, record.text, hash([record.principalId, scopeKey(record.scope)]));
      for (const dep of record.dependencies) this.#statement('INSERT INTO jmem_dependencies VALUES(?,?)').run(record.id, dep);
    }
    for (const ref of record.sourceRefs) this.#statement('INSERT OR IGNORE INTO jmem_sources VALUES(?,?,?,?,?)').run(targetId, record.principalId, ref.eventId, ref.start, ref.end);
  }

  #hydrate(row: Row): MemoryRecord {
    const record = JSON.parse(String(row.body)) as MemoryRecord;
    record.status = row.status as MemoryRecord['status']; record.revision = Number(row.revision);
    record.retiredAt = row.retired_at === null ? null : Number(row.retired_at);
    record.sourceRefs = (this.#statement('SELECT event_id,start,end FROM jmem_sources WHERE record_id=? ORDER BY event_id').all(record.id) as Row[]).map(r => ({ eventId: String(r.event_id), start: Number(r.start), end: Number(r.end) }));
    record.dependencies = (this.#statement('SELECT depends_on FROM jmem_dependencies WHERE record_id=?').all(record.id) as Row[]).map(r => String(r.depends_on));
    return record;
  }

  #raw(id: string): MemoryRecord | null {
    const row = this.#statement('SELECT * FROM jmem_records WHERE id=?').get(id) as Row | undefined;
    return row ? this.#hydrate(row) : null;
  }

  get(access: Access, id: string): MemoryRecord | null {
    assertAccess(access); assertId(id);
    const record = this.#raw(id);
    return record && allowed(record, access) ? record : null;
  }

  #where(access: Access): { sql: string; params: string[] } {
    assertAccess(access);
    return { sql: `r.principal=? AND r.scope IN (${access.scopes.map(() => '?').join(',') || 'NULL'})`, params: [access.principalId, ...access.scopes.map(scopeKey)] };
  }

  search(input: SearchInput): SearchHit[] {
    if (typeof input.query !== 'string' || input.query.length > 32768) throw new TypeError('Invalid query');
    const limit = input.limit ?? 48;
    if (!Number.isInteger(limit) || limit < 1 || limit > 256) throw new TypeError('Search limit must be 1..256');
    const validAt = input.validAt ?? this.clock(); const knownAt = input.knownAt ?? this.clock();
    finiteTime(validAt); finiteTime(knownAt);
    const historical = input.knownAt !== undefined || input.validAt !== undefined;
    const where = this.#where(input);
    const filters = `${where.sql} AND r.recorded_at<=? AND (r.retired_at IS NULL OR r.retired_at>?)
      AND (r.expires_at IS NULL OR r.expires_at>?) AND (r.valid_from IS NULL OR r.valid_from<=?) AND (r.valid_to IS NULL OR r.valid_to>?)
      AND r.status IN (${historical ? "'active','conflicted','superseded'" : "'active','conflicted'"})
      AND NOT EXISTS (SELECT 1 FROM jmem_dependencies dep JOIN jmem_records target ON target.id=dep.depends_on
        WHERE dep.record_id=r.id AND (target.status<>'active' OR (target.expires_at IS NOT NULL AND target.expires_at<=${validAt}) OR (target.valid_from IS NOT NULL AND target.valid_from>${validAt}) OR (target.valid_to IS NOT NULL AND target.valid_to<=${validAt})))`;
    const params = [...where.params, knownAt, knownAt, validAt, validAt, validAt];
    // Constraints are always applicable. Preferences still need to match the
    // request, unless this is an explicit empty-query browse. This keeps a
    // personal preference such as a drink choice out of every agent prompt.
    const mandatoryKinds = input.query.trim() ? "'constraint'" : "'constraint','preference'";
    const mandatory = this.#statement(`SELECT r.* FROM jmem_records r WHERE ${filters} AND json_extract(r.body,'$.kind') IN (${mandatoryKinds}) ORDER BY r.recorded_at DESC LIMIT 129`).all(...params) as Row[];
    const terms = [...new Set(input.query.toLocaleLowerCase().match(/[\p{L}\p{N}_]{2,}/gu) ?? [])].slice(0, 32);
    const scopeTerms = input.scopes.map(s => '"' + hash([input.principalId, scopeKey(s)]) + '"').join(' OR ');
    const match = `scope_token:(${scopeTerms}) AND text:(${terms.map(t => '"' + t.replaceAll('"', '""') + '"').join(' OR ')})`;
    // CROSS JOIN fixes the loop order: traverse authorized FTS postings once, then
    // look up each row by rowid. A scope-index-first plan repeats MATCH per record.
    const result = terms.length && input.scopes.length ? this.#statement(`SELECT r.*,bm25(jmem_fts,0,1,0) AS rank FROM jmem_fts CROSS JOIN jmem_records r WHERE r.rowid=jmem_fts.rowid AND jmem_fts MATCH ? AND ${filters} ORDER BY rank LIMIT ?`).all(match, ...params, limit) as Row[] : [];
    const map = new Map<string, SearchHit>();
    for (const row of [...mandatory, ...result]) {
      const record = this.#hydrate(row);
      const must = record.kind === 'constraint' || record.kind === 'preference';
      if (!map.has(record.id)) map.set(record.id, { record, mandatory: must, score: must ? 1 : Math.min(0.85, 0.35 + Math.abs(Number(row.rank ?? 0))) });
    }
    // A more-specific preference can mask a user preference even when only the
    // user text matched the query. Pull same-slot counterparts so that mask runs.
    const slots = new Map<string, [string | null, string | null, string]>();
    for (const hit of map.values()) {
      if (hit.record.kind === 'preference' && hit.record.predicate) slots.set(hash([hit.record.subject, hit.record.predicate, hit.record.modality]), [hit.record.subject, hit.record.predicate, hit.record.modality]);
    }
    if (slots.size) {
      const clause = [...slots.keys()].map(() => `(json_extract(r.body,'$.subject') IS ? AND json_extract(r.body,'$.predicate') IS ? AND json_extract(r.body,'$.modality')=?)`).join(' OR ');
      const rows = this.#statement(`SELECT r.* FROM jmem_records r WHERE ${filters} AND json_extract(r.body,'$.kind')='preference' AND (${clause}) LIMIT 32`).all(...params, ...[...slots.values()].flat()) as Row[];
      for (const row of rows) {
        const record = this.#hydrate(row);
        if (map.has(record.id) || record.scope.kind === 'user') continue;
        map.set(record.id, { record, mandatory: true, score: 1 });
      }
    }
    // More-specific preferences mask matching user preferences without deleting them.
    const overrides = new Set([...map.values()].filter(h => h.record.scope.kind !== 'user' && h.record.predicate).map(h => hash([h.record.subject, h.record.predicate, h.record.modality])));
    const output = [...map.values()].filter(h => !(h.record.scope.kind === 'user' && h.record.predicate && overrides.has(hash([h.record.subject, h.record.predicate, h.record.modality]))));
    if (mandatory.length >= 129 && output[0]) output[0].truncatedMandatory = true;
    return output;
  }

  related(access: Access, subject: string, predicate: string): MemoryRecord[] {
    const w = this.#where(access);
    return (this.#statement(`SELECT r.* FROM jmem_records r WHERE ${w.sql} AND r.status IN ('active','conflicted') AND json_extract(r.body,'$.subject')=? AND json_extract(r.body,'$.predicate')=? ORDER BY r.recorded_at DESC LIMIT 4`).all(...w.params, subject, predicate) as Row[]).map(r => this.#hydrate(r));
  }

  #staleDependents(id: string): void {
    const rows = this.#statement(`WITH RECURSIVE affected(id) AS (
      SELECT record_id FROM jmem_dependencies WHERE depends_on=? UNION SELECT d.record_id FROM jmem_dependencies d JOIN affected a ON d.depends_on=a.id
    ) SELECT r.* FROM jmem_records r JOIN affected a ON r.id=a.id`).all(id) as Row[];
    for (const row of rows) {
      this.#statement("UPDATE jmem_records SET status='stale',revision=revision+1 WHERE id=? AND status IN ('active','conflicted')").run(String(row.id));
      const record = JSON.parse(String(row.body)) as MemoryRecord;
      this.#bump(record.principalId, record.scope);
    }
  }

  correct(access: Access, id: string, expectedRevision: number, source: MemoryEvent, record: MemoryRecord): MemoryRecord {
    assertAccess(access);
    return this.transaction(() => {
      const old = this.get(access, id);
      if (!old || old.status === 'superseded' || old.revision !== expectedRevision) throw new Error('Memory revision conflict');
      if (source.source !== 'user' || !allowed(source, access) || source.principalId !== old.principalId || scopeKey(source.scope) !== scopeKey(old.scope)) throw new Error('Correction requires an authorized user source in the same scope');
      if (record.supersedesId !== id || record.id === id) throw new Error('Invalid correction version');
      const command = this.eventCommand(source);
      this.#statement(command.sql).run(...command.params);
      this.#statement("UPDATE jmem_records SET status='superseded',retired_at=?,revision=revision+1 WHERE id=?").run(this.clock(), id);
      this.#put(record);
      this.#statement("UPDATE jmem_jobs SET status='done',lease_until=0,token=NULL WHERE principal=? AND event_id=?").run(source.principalId, source.id);
      this.#staleDependents(id); this.#bump(old.principalId, old.scope);
      return this.#raw(record.id)!;
    });
  }

  // Source-first deletion removes unsupported claims and invalidates their dependents.
  forgetSource(access: Access, eventId: string): { removedRecords: number; logicalPurge: boolean; physicalPurge: 'not_guaranteed' } {
    assertAccess(access); assertId(eventId);
    return this.transaction(() => {
      const row = this.#statement('SELECT body FROM jmem_events WHERE principal=? AND id=?').get(access.principalId, eventId);
      if (!row) return { removedRecords: 0, logicalPurge: true, physicalPurge: 'not_guaranteed' as const };
      const event = JSON.parse(String(row.body)) as MemoryEvent;
      if (!allowed(event, access)) throw new Error('Source not accessible');
      const ids = this.#statement('SELECT record_id FROM jmem_sources WHERE principal=? AND event_id=?').all(access.principalId, eventId) as Row[];
      this.#statement('INSERT OR REPLACE INTO jmem_revocations VALUES(?,?,?)').run(access.principalId, eventId, this.clock());
      this.#statement('DELETE FROM jmem_events WHERE principal=? AND id=?').run(access.principalId, eventId);
      let removedRecords = 0;
      for (const row of ids) {
        const id = String(row.record_id);
        const surviving = this.#statement('SELECT 1 FROM jmem_sources WHERE record_id=? LIMIT 1').get(id);
        if (!surviving) {
          this.#staleDependents(id);
          this.#statement('DELETE FROM jmem_fts WHERE rowid=(SELECT rowid FROM jmem_records WHERE id=?)').run(id);
          this.#statement('DELETE FROM jmem_records WHERE id=?').run(id); removedRecords++;
        }
      }
      this.#bump(event.principalId, event.scope);
      this.#statement('DELETE FROM jmem_receipts WHERE principal=?').run(access.principalId);
      return { removedRecords, logicalPurge: true, physicalPurge: 'not_guaranteed' as const };
    });
  }

  forget(access: Access, id: string): ReturnType<SqliteStore['forgetSource']> {
    const record = this.get(access, id);
    if (!record) return { removedRecords: 0, logicalPurge: true, physicalPurge: 'not_guaranteed' };
    // Forgetting a claim also forgets its original supporting events. This explicit
    // source-oriented policy prevents a later extractor from recreating it.
    let removedRecords = 0;
    for (const ref of record.sourceRefs) removedRecords += this.forgetSource(access, ref.eventId).removedRecords;
    return { removedRecords, logicalPurge: true, physicalPurge: 'not_guaranteed' };
  }

  saveReceipt(principal: string, receipt: Receipt): void {
    this.transaction(() => {
      this.#statement('INSERT INTO jmem_receipts VALUES(?,?,?,?)').run(receipt.id, principal, JSON.stringify(receipt), this.clock());
      this.#statement('DELETE FROM jmem_receipts WHERE principal=? AND id NOT IN (SELECT id FROM jmem_receipts WHERE principal=? ORDER BY at DESC LIMIT 256)').run(principal, principal);
    });
  }

  explain(access: Access, id: string): Receipt | null {
    assertAccess(access); assertId(id);
    const row = this.#statement('SELECT body FROM jmem_receipts WHERE id=? AND principal=?').get(id, access.principalId);
    if (!row) return null;
    const receipt = JSON.parse(String(row.body)) as Receipt;
    if (receipt.scopes.some(scope => !access.scopes.some(s => s.kind === scope.kind && s.id === scope.id))) return null;
    if (receipt.selected.some(r => !this.get(access, r.id))) return null;
    return receipt;
  }

  export(access: Access, afterId = '', limit = 100): { schema: 1; records: MemoryRecord[]; next: string | null } {
    assertId(afterId || '_');
    if (!Number.isInteger(limit) || limit < 1 || limit > 1000) throw new TypeError('Export limit must be 1..1000');
    const w = this.#where(access);
    const rows = this.#statement(`SELECT r.* FROM jmem_records r WHERE ${w.sql} AND r.id>? ORDER BY r.id LIMIT ?`).all(...w.params, afterId, limit + 1) as Row[];
    const records = rows.slice(0, limit).map(r => this.#hydrate(r));
    return { schema: 1, records, next: rows.length > limit ? records.at(-1)!.id : null };
  }

  health(): { schema: 1; jobs: Record<string, number>; integrity: string } {
    const jobs: Record<string, number> = {};
    for (const row of this.#statement('SELECT status,count(*) AS count FROM jmem_jobs GROUP BY status').all() as Row[]) jobs[String(row.status)] = Number(row.count);
    return { schema: 1, jobs, integrity: String(this.#statement('PRAGMA quick_check').get()?.quick_check) };
  }

  reconsider(access: Access, limit = 64): number {
    assertAccess(access);
    if (!Number.isInteger(limit) || limit < 1 || limit > 256) throw new TypeError('Retry limit must be 1..256');
    return this.transaction(() => {
      const w = this.#where(access);
      const rows = this.#statement(`SELECT e.id,e.body FROM jmem_events e JOIN jmem_jobs j ON j.principal=e.principal AND j.event_id=e.id
        WHERE e.principal=? AND e.scope IN (${access.scopes.map(() => '?').join(',') || 'NULL'})
        AND (j.status='dead' OR (j.status='done' AND EXISTS(SELECT 1 FROM jmem_sources s JOIN jmem_records r ON r.id=s.record_id WHERE s.principal=e.principal AND s.event_id=e.id AND r.status='pending')))
        ORDER BY e.recorded_at LIMIT ?`).all(...w.params, limit) as Row[];
      for (const row of rows) {
        const event = JSON.parse(String(row.body)) as MemoryEvent;
        const pending = this.#statement("SELECT r.id FROM jmem_records r JOIN jmem_sources s ON s.record_id=r.id WHERE s.principal=? AND s.event_id=? AND r.status='pending'").all(access.principalId, event.id) as Row[];
        for (const record of pending) {
          this.#statement('DELETE FROM jmem_fts WHERE rowid=(SELECT rowid FROM jmem_records WHERE id=?)').run(String(record.id));
          this.#statement('DELETE FROM jmem_records WHERE id=?').run(String(record.id));
        }
        this.#statement("UPDATE jmem_jobs SET status='pending',attempts=0,available_at=0,lease_until=0,token=NULL,error=NULL,progress=0,prepared=NULL WHERE principal=? AND event_id=?").run(access.principalId, event.id);
        this.#bump(access.principalId, event.scope);
      }
      return rows.length;
    });
  }

  close(): void { if (!this.#closed) { this.#statements.clear(); this.db.close(); this.#closed = true; } }
}

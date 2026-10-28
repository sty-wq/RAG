import contextlib
import fcntl
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from ..common import PhaseError, save_json, save_bytes, digest


def root():
    return Path(os.environ.get('CABLE_DATA_DIR', 'data')).resolve()


def ident(): return uuid.uuid4().hex


class Store:
    def __init__(self, path=None):
        self.root=Path(path or root()).resolve()
        self.root.mkdir(parents=True,exist_ok=True)
        self.db=self.root/'workflow.sqlite3'
        with (self.root/'.migration.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            try:self.migrate()
            finally:fcntl.flock(lock,fcntl.LOCK_UN)

    def connect(self):
        c=sqlite3.connect(self.db,timeout=3,isolation_level=None)
        c.row_factory=sqlite3.Row
        c.execute('PRAGMA foreign_keys=ON')
        c.execute('PRAGMA busy_timeout=3000')
        return c

    def migrate(self):
        with contextlib.closing(self.connect()) as c:
            c.execute('PRAGMA journal_mode=WAL')
            version=c.execute('PRAGMA user_version').fetchone()[0]
            if version>4: raise PhaseError('SCHEMA_TOO_NEW')
            if version==4:return
            if version==3:
                from .migration_v4 import migrate
                migrate(c);return
            if version==2:
                from .migration_v3 import migrate
                migrate(c)
                from .migration_v4 import migrate as migrate_v4
                migrate_v4(c);return
            c.executescript('''
            CREATE TABLE IF NOT EXISTS jobs(
              id TEXT PRIMARY KEY, name TEXT NOT NULL, input_hash TEXT NOT NULL,
              original TEXT NOT NULL, status TEXT NOT NULL, created REAL NOT NULL,
              revision INTEGER NOT NULL DEFAULT 0, ir_ref TEXT NOT NULL,
              result_ref TEXT, error TEXT, page_count INTEGER, mode TEXT NOT NULL,
              cloud_enabled INTEGER NOT NULL, upload_key TEXT UNIQUE);
            CREATE TABLE IF NOT EXISTS page_stages(
              job_id TEXT NOT NULL REFERENCES jobs(id), page_index INTEGER NOT NULL,
              stage TEXT NOT NULL, status TEXT NOT NULL, artifact_ref TEXT, error TEXT,
              updated REAL NOT NULL, PRIMARY KEY(job_id,page_index));
            CREATE TABLE IF NOT EXISTS outbox(
              id INTEGER PRIMARY KEY, event_key TEXT NOT NULL UNIQUE, task TEXT NOT NULL,
              args TEXT NOT NULL, created REAL NOT NULL, published REAL, publishes INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS provider_attempts(
              id TEXT PRIMARY KEY, provider TEXT NOT NULL, endpoint TEXT NOT NULL, profile TEXT,
              document_id TEXT, page_id TEXT, input_strategy TEXT, input_hash TEXT,
              request_key TEXT, started_at REAL NOT NULL, finished_at REAL, latency_ms REAL,
              status TEXT NOT NULL, provider_request_id TEXT, estimated_billing_units INTEGER NOT NULL DEFAULT 0,
              reported_billing_info TEXT, retry_of TEXT, cache_hit INTEGER NOT NULL DEFAULT 0,
              error_type TEXT, kind TEXT NOT NULL, mode TEXT, billing_status TEXT);
            CREATE INDEX IF NOT EXISTS attempts_key ON provider_attempts(request_key);
            CREATE TABLE IF NOT EXISTS budget(id INTEGER PRIMARY KEY CHECK(id=1), max_calls INTEGER NOT NULL,
              reserved INTEGER NOT NULL DEFAULT 0 CHECK(reserved>=0));
            INSERT OR IGNORE INTO budget(id,max_calls) VALUES(1,400);
            CREATE TABLE IF NOT EXISTS review_items(
              id TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id), page_index INTEGER NOT NULL,
              cell_id TEXT NOT NULL, evidence TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'OPEN',
              revision INTEGER NOT NULL DEFAULT 0, UNIQUE(job_id,page_index,cell_id));
            CREATE TABLE IF NOT EXISTS decisions(
              id TEXT PRIMARY KEY, request_id TEXT NOT NULL UNIQUE, review_id TEXT NOT NULL REFERENCES review_items(id),
              actor TEXT NOT NULL, timestamp REAL NOT NULL, candidate_id TEXT, selected_form TEXT NOT NULL,
              manual_text TEXT, reason TEXT NOT NULL, expected_revision INTEGER NOT NULL,
              text TEXT, validation TEXT NOT NULL, payload_hash TEXT NOT NULL,
              UNIQUE(review_id,expected_revision));
            CREATE TABLE IF NOT EXISTS confusion_events(
              decision_id TEXT PRIMARY KEY REFERENCES decisions(id), observed TEXT NOT NULL, confirmed TEXT NOT NULL,
              context TEXT NOT NULL);
            PRAGMA user_version=1;
            ''')
            from .migration03 import migrate
            c.execute('BEGIN IMMEDIATE')
            try:
                migrate(c);c.commit()
            except Exception:
                c.rollback();raise
            from .migration_v3 import migrate as migrate_v3
            migrate_v3(c)
            from .migration_v4 import migrate as migrate_v4
            migrate_v4(c)

    @contextlib.contextmanager
    def tx(self):
        c=self.connect()
        try:
            c.execute('BEGIN IMMEDIATE')
            yield c
            c.commit()
        except Exception:
            c.rollback();raise
        finally:c.close()

    def all(self,sql,args=()):
        with contextlib.closing(self.connect()) as c:return [dict(r) for r in c.execute(sql,args)]

    def one(self,sql,args=()):
        rows=self.all(sql,args)
        return rows[0] if rows else None

    def job(self,j):
        value=self.one('SELECT * FROM jobs WHERE id=?',(j,))
        if not value:raise PhaseError('NOT_FOUND')
        return value

    def event(self,c,key,task,args):
        c.execute('INSERT OR IGNORE INTO outbox(event_key,task,args,created) VALUES(?,?,?,?)',
                  (key,task,json.dumps(args),time.time()))

    def artifact(self,relative,value):
        path=self.root/relative
        save_json(path,value)
        return str(path.relative_to(self.root))

    def read(self,ref):
        try:return json.loads(self.resolve(ref).read_text())
        except (OSError,ValueError):raise PhaseError('ARTIFACT_MISSING_OR_INVALID') from None

    def resolve(self,ref):
        p=(self.root/ref).resolve()
        if not p.is_relative_to(self.root):raise PhaseError('INVALID_ARTIFACT_PATH')
        return p

    def upload(self,data,name,mode='LIVE',cloud=True,upload_key=None,processing_contract='V2',ocr_provider=None):
        from .settings import switches
        settings=switches();cloud=bool(cloud and settings['cloud'])
        ocr_provider=ocr_provider or ('baidu' if processing_contract=='V2' else settings['provider'])
        if ocr_provider not in {'baidu','deepseek'}:raise PhaseError('UNSUPPORTED_CLOUD_PROVIDER')
        if processing_contract=='V2' and ocr_provider!='baidu':raise PhaseError('PROVIDER_REQUIRES_V3')
        if processing_contract not in {'V2','V3'}:raise PhaseError('INVALID_CONTRACT')
        if mode not in {'LIVE','REPLAY'}:raise PhaseError('INVALID_MODE')
        from ..admission import validate,log_rejection
        try:validate(data,name)
        except PhaseError as e:
            log_rejection(self.root,e.code,e.details);raise
        suffix=Path(name).suffix.lower()
        sha=digest(data)
        if upload_key:
            existing=self.one('SELECT * FROM jobs WHERE upload_key=?',(upload_key,))
            if existing:
                if existing['input_hash']!=sha or existing['mode']!=mode or existing['cloud_enabled']!=int(cloud) or existing['processing_contract']!=processing_contract or existing['ocr_provider']!=ocr_provider:raise PhaseError('IDEMPOTENCY_CONFLICT')
                return existing
        j=ident();original=f'uploads/{sha}{suffix}'
        save_bytes(self.root/original,data)
        ir=self.artifact(f'jobs/{j}/ir-0.json',{'schema_version':'ocr-ir-v3' if processing_contract=='V3' else 'phase02-v1','document_id':j,'revision':0,'pages':[]})
        try:
            with self.tx() as c:
                c.execute('INSERT INTO jobs(id,name,input_hash,original,status,created,ir_ref,mode,cloud_enabled,upload_key,processing_contract,ocr_provider) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                          (j,Path(name.replace('\\','/')).name[:200],sha,original,'CREATED',time.time(),ir,mode,int(cloud),upload_key,processing_contract,ocr_provider))
                self.event(c,f'ingest:{j}','document_ingest',[j])
        except sqlite3.IntegrityError:
            if upload_key:
                existing=self.one('SELECT * FROM jobs WHERE upload_key=?',(upload_key,))
                if existing and existing['input_hash']==sha and existing['mode']==mode and existing['cloud_enabled']==int(cloud) and existing['processing_contract']==processing_contract and existing['ocr_provider']==ocr_provider:return existing
            raise PhaseError('IDEMPOTENCY_CONFLICT') from None
        return self.job(j)

    def stage(self,j,p,stage):
        with self.tx() as c:
            c.execute('UPDATE page_stages SET stage=?,updated=? WHERE job_id=? AND page_index=?',
                      (stage,time.time(),j,p))

    def stats(self):
        budget=self.one('SELECT * FROM budget WHERE id=1')
        rows=self.all('SELECT * FROM provider_attempts')
        live=[r for r in rows if r['kind']=='OCR' and not r['cache_hit']]
        unlimited=budget['max_calls']==-1
        return {'planned_api_budget':None if unlimited else budget['max_calls'],'budget_unlimited':unlimited,'budget_reserved':budget['reserved'],
          'actual_ocr_calls':len(live),'actual_oauth_calls':sum(r['kind']=='OAUTH' for r in rows),
          'cache_hits':sum(r['cache_hit'] for r in rows),'replay_runs':sum(r['mode']=='REPLAY' for r in rows),
          'failed_calls':sum(r['status'] not in {'SUCCESS','CACHE_HIT'} for r in live),
          'retried_calls':sum(bool(r['retry_of']) for r in live),
          'automatic_retries':0,'intentional_force_live_calls':sum(r['mode']=='FORCE_LIVE' for r in live),
          'cache_hit_rate':sum(r['cache_hit'] for r in rows)/max(1,len([r for r in rows if r['kind']=='OCR'])),
          'estimated_billing_units':sum(r['estimated_billing_units'] for r in live),
          'known_billing_units':None,'remaining_experiment_budget':None if unlimited else budget['max_calls']-budget['reserved'],
          'account_remaining_quota':None,'external_exactly_once':False}

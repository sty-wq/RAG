"""Add observational history without reconstructing fictitious old stage events."""
import contextlib

def migrate(c):
    # Called inside BEGIN IMMEDIATE by Store migration. Existing evidence is untouched.
    version=c.execute('PRAGMA user_version').fetchone()[0]
    if version>=2:return
    c.execute('ALTER TABLE jobs ADD COLUMN updated REAL')
    c.execute('UPDATE jobs SET updated=created')
    statements=[
    '''CREATE TABLE stage_events(id INTEGER PRIMARY KEY,job_id TEXT NOT NULL, page_index INTEGER,
        scope TEXT NOT NULL,stage TEXT NOT NULL,occurred REAL NOT NULL,source TEXT NOT NULL)''',
    '''CREATE TABLE runtime_heartbeats(role TEXT PRIMARY KEY,updated REAL NOT NULL)''',
    '''CREATE TABLE operational_snapshots(name TEXT PRIMARY KEY,value TEXT NOT NULL,recorded REAL NOT NULL)''',
    '''CREATE TABLE benchmark_runs(run_id TEXT PRIMARY KEY,recorded REAL NOT NULL,build_version TEXT,dataset_version TEXT,metrics TEXT NOT NULL,source TEXT)''',
    '''CREATE TABLE structure_observations(job_id TEXT NOT NULL,page_index INTEGER NOT NULL,reason TEXT NOT NULL,updated REAL NOT NULL,PRIMARY KEY(job_id,page_index))''',
    'CREATE INDEX jobs_status_created ON jobs(status,created)',
    'CREATE INDEX jobs_created ON jobs(created)',
    'CREATE INDEX attempts_started ON provider_attempts(started_at)',
    'CREATE INDEX attempts_status_started ON provider_attempts(status,started_at)',
    'CREATE INDEX reviews_status ON review_items(status)',
    'CREATE INDEX decisions_timestamp ON decisions(timestamp)',
    'CREATE INDEX events_job_time ON stage_events(job_id,occurred)',
    '''CREATE TRIGGER jobs_created_event AFTER INSERT ON jobs BEGIN
      INSERT INTO stage_events(job_id,scope,stage,occurred,source) VALUES(NEW.id,'JOB',NEW.status,NEW.created,'db_trigger_v2');
      UPDATE jobs SET updated=NEW.created WHERE id=NEW.id; END''',
    '''CREATE TRIGGER jobs_status_event AFTER UPDATE OF status ON jobs WHEN OLD.status!=NEW.status BEGIN
      INSERT INTO stage_events(job_id,scope,stage,occurred,source) VALUES(NEW.id,'JOB',NEW.status,(julianday('now')-2440587.5)*86400,'db_trigger_v2');
      UPDATE jobs SET updated=(julianday('now')-2440587.5)*86400 WHERE id=NEW.id; END''',
    '''CREATE TRIGGER page_created_event AFTER INSERT ON page_stages BEGIN
      INSERT INTO stage_events(job_id,page_index,scope,stage,occurred,source) VALUES(NEW.job_id,NEW.page_index,'PAGE_CREATED',NEW.stage,NEW.updated,'db_trigger_v2'); END''',
    '''CREATE TRIGGER page_stage_event AFTER UPDATE OF stage ON page_stages WHEN OLD.stage!=NEW.stage BEGIN
      INSERT INTO stage_events(job_id,page_index,scope,stage,occurred,source) VALUES(NEW.job_id,NEW.page_index,'PAGE_STAGE',NEW.stage,NEW.updated,'db_trigger_v2');
      UPDATE jobs SET updated=NEW.updated WHERE id=NEW.job_id; END'''
    ]
    for statement in statements:c.execute(statement)
    c.execute('PRAGMA user_version=2')

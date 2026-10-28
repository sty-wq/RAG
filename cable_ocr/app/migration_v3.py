"""Preserve review and decision identities while permitting explicit LINE targets."""
def migrate(c):
    c.execute('PRAGMA foreign_keys=OFF')
    c.execute('BEGIN IMMEDIATE')
    try:
        before=[tuple(r) for r in c.execute('SELECT * FROM decisions ORDER BY id')]
        count=c.execute('SELECT count(*) FROM review_items').fetchone()[0]
        c.execute("ALTER TABLE jobs ADD COLUMN processing_contract TEXT NOT NULL DEFAULT 'V2'")
        c.execute('ALTER TABLE jobs ADD COLUMN parse_mode TEXT')
        for col in ('evidence_ref TEXT','parse_mode TEXT','enhancement_status TEXT','extraction_revision INTEGER NOT NULL DEFAULT 0'):
            c.execute('ALTER TABLE page_stages ADD COLUMN '+col)
        c.execute('''CREATE TABLE review_items_new(
          id TEXT PRIMARY KEY,job_id TEXT NOT NULL REFERENCES jobs(id),page_index INTEGER NOT NULL,
          cell_id TEXT,evidence TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'OPEN',revision INTEGER NOT NULL DEFAULT 0,
          target_kind TEXT NOT NULL DEFAULT 'CELL' CHECK(target_kind IN ('CELL','LINE')),target_id TEXT NOT NULL,
          extraction_revision INTEGER NOT NULL DEFAULT 0,
          review_requirement TEXT NOT NULL DEFAULT 'REQUIRED' CHECK(review_requirement IN ('REQUIRED','OPTIONAL')),
          CHECK((target_kind='CELL' AND cell_id IS NOT NULL AND target_id=cell_id) OR (target_kind='LINE' AND cell_id IS NULL)),
          UNIQUE(job_id,page_index,target_kind,target_id,extraction_revision),
          UNIQUE(job_id,page_index,cell_id,extraction_revision))''')
        c.execute('''INSERT INTO review_items_new(id,job_id,page_index,cell_id,evidence,status,revision,target_id)
          SELECT id,job_id,page_index,cell_id,evidence,status,revision,cell_id FROM review_items''')
        c.execute('DROP TABLE review_items')
        c.execute('ALTER TABLE review_items_new RENAME TO review_items')
        c.execute('CREATE INDEX reviews_status ON review_items(status)')
        c.execute('CREATE INDEX reviews_target ON review_items(job_id,target_kind,review_requirement,status)')
        c.execute('''CREATE TABLE review_requirement_events(id TEXT PRIMARY KEY,request_id TEXT UNIQUE NOT NULL,
          review_id TEXT NOT NULL REFERENCES review_items(id),actor TEXT NOT NULL,reason TEXT NOT NULL,
          expected_revision INTEGER NOT NULL,timestamp REAL NOT NULL,payload_hash TEXT NOT NULL)''')
        assert count==c.execute('SELECT count(*) FROM review_items').fetchone()[0]
        assert before==[tuple(r) for r in c.execute('SELECT * FROM decisions ORDER BY id')]
        assert not c.execute('PRAGMA foreign_key_check').fetchall()
        c.execute('PRAGMA user_version=3');c.commit()
    except BaseException:
        c.rollback();raise
    finally:c.execute('PRAGMA foreign_keys=ON')

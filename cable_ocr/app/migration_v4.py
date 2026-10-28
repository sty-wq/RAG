"""Pin each job to an OCR channel without changing historical results."""
def migrate(c):
    c.execute('BEGIN IMMEDIATE')
    try:
        c.execute("ALTER TABLE jobs ADD COLUMN ocr_provider TEXT NOT NULL DEFAULT 'baidu' CHECK(ocr_provider IN ('baidu','deepseek'))")
        c.execute('PRAGMA user_version=4')
        c.commit()
    except BaseException:
        c.rollback()
        raise

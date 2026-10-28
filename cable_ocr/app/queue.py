import json
import os
import time
import threading
from celery.signals import worker_ready,worker_shutdown
from celery import Celery
from .store import Store
from .workflow import ingest_document,process_page,finalize_document

celery=Celery('cable_ocr',broker=os.environ.get('CABLE_BROKER_URL','redis://127.0.0.1:6380/0'))
celery.conf.update(task_ignore_result=True,task_serializer='json',accept_content=['json'],
    task_acks_late=True,task_reject_on_worker_lost=True,worker_prefetch_multiplier=1,
    broker_transport_options={'visibility_timeout':300},task_time_limit=240,
    task_publish_retry=True,task_publish_retry_policy={'max_retries':3,'interval_start':1,'interval_step':2,'interval_max':5},
    broker_connection_timeout=5,broker_connection_max_retries=3,
    worker_hijack_root_logger=False)

@celery.task(name='document_ingest')
def document_ingest(j):ingest_document(j)
@celery.task(name='page_process')
def page_process(j,p):process_page(j,p)
@celery.task(name='document_finalize')
def document_finalize(j):finalize_document(j)


def dispatch_once(store=None,publish=None):
    s=store or Store();publish=publish or (lambda task,args:celery.send_task(task,args=args,retry=False))
    rows=s.all('SELECT * FROM outbox WHERE published IS NULL ORDER BY id LIMIT 50')
    count=0
    for row in rows:
        # Network publish outside transaction. Crash before acknowledgement produces deliberate duplicate delivery.
        publish(row['task'],json.loads(row['args']))
        with s.tx() as c:
            c.execute('UPDATE outbox SET published=?,publishes=publishes+1 WHERE id=?',(time.time(),row['id']))
            if row['task']=='document_ingest':c.execute("UPDATE jobs SET status='QUEUED' WHERE id=? AND status='CREATED'",(json.loads(row['args'])[0],))
        count+=1
    return count


def recover_pending(s):
    # Reconstruct transport intent from business truth after broker loss / process kill.
    # OS file locks fence still-running workers. No OCR retry is implied.
    slot=int(time.time()//60)
    with s.tx() as c:
        for j in c.execute("SELECT id FROM jobs WHERE page_count IS NULL AND status IN ('CREATED','QUEUED')").fetchall():
            s.event(c,f'recover-ingest:{j[0]}:{slot}','document_ingest',[j[0]])
        for p in c.execute("SELECT job_id,page_index FROM page_stages WHERE status IN ('QUEUED','PROCESSING') AND updated<?",(time.time()-60,)).fetchall():
            s.event(c,f'recover-page:{p[0]}:{p[1]}:{slot}','page_process',list(p))
        for j in c.execute("SELECT id FROM jobs WHERE status IN ('PROCESSING','WAITING_REVIEW')").fetchall():
            s.event(c,f'recover-finalize:{j[0]}:{slot}','document_finalize',[j[0]])


_heartbeat_stop=threading.Event()
def heartbeat(role):
    try:
        s=Store()
        with s.tx() as c:c.execute('INSERT OR REPLACE INTO runtime_heartbeats VALUES(?,?)',(role,time.time()))
    except Exception:pass

@worker_ready.connect
def start_heartbeat(**kwargs):
    def beat():
        while not _heartbeat_stop.is_set():
            heartbeat('worker');_heartbeat_stop.wait(10)
    threading.Thread(target=beat,daemon=True).start()

@worker_shutdown.connect
def stop_heartbeat(**kwargs):_heartbeat_stop.set()

def main():
    s=Store();delay=1;last_inventory=0
    while True:
        try:
            recover_pending(s);dispatch_once(s);delay=1
            heartbeat('dispatcher')
            if time.time()-last_inventory>60:
                from .observability import collect_artifacts
                collect_artifacts(s);last_inventory=time.time()
        except Exception as e:
            # No exception repr: transport URLs may include broker credentials.
            print('dispatcher:',type(e).__name__,flush=True);delay=min(30,delay*2)
        time.sleep(delay)

if __name__=='__main__':main()

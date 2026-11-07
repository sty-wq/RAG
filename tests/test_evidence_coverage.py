from fastapi.testclient import TestClient

from answer import Source
from api import Deps, build_real_deps, create_app
from evidence_coverage import assess_coverage
from test_answer import FakeEmbedder, FakeStore, HITS


def item(i, status, lines=None, passage=None, source=1):
    return dict(id=i,status=status,reason="核对原文",answer_ids=lines or [],
                evidence=[dict(source=source,passage=passage)] if passage else [])


def test_api_partial_and_absent_points(fake_llm):
    answer="该方法使用红黑树[1]。"
    llm=fake_llm(answer,{"requirements":["使用的方法","节点数量"]},
                 {"items":[item(2,"missing"),item("1","supported",["1"],"1")]})
    deps=Deps(FakeEmbedder(),FakeStore(HITS),llm)
    with TestClient(create_app(deps)) as client:
        res=client.post("/ask",json={"question":"用了什么方法，有多少个节点？","evidence_coverage":True})
    assert res.status_code==200
    result=res.json()
    coverage=result["evidence_coverage"]
    assert result['answer']==answer and coverage['ratio']==0.5
    assert coverage['points'][0]['evidence']==[{'source':1,'passage':1,'quote':HITS[0].text}]
    assert coverage['points'][1]['status']=='missing'
    assert answer not in llm.prompts[1]


def test_unverifiable_quote_is_not_support(fake_llm):
    source=Source(1,"doc","h","doc",1,"10 records")
    text="结果有100条[1]。"
    llm=fake_llm({"requirements":["数量"]},{"items":[item(1,"supported",[1],99)]})
    result=assess_coverage("多少条",text,[source],llm)
    assert result.covered is None and result.status=="unavailable"


def test_invalid_citation_and_service_failure(fake_llm):
    src=[Source(1,"doc","h","doc",1,"原文")]
    result=assess_coverage("多少条","结论[9]",src,fake_llm({"requirements":["数量"]}))
    assert result.ratio==0 and result.invalid_citations==[9]
    result=assess_coverage("多少条","结论[1]",src,fake_llm({"requirements":["数量"]},{"items":[]}))
    assert result.ratio is None and result.status=="unavailable"


def test_real_dependencies_construct_without_request(monkeypatch):
    import vector_store
    import llm
    import reranker
    monkeypatch.setattr(vector_store,"VectorStore",lambda: object())
    monkeypatch.setattr(llm,"DeepSeekClient",lambda: object())
    monkeypatch.setattr(reranker,"create_reranker",lambda: None)
    assert build_real_deps().store is not None

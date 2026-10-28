"""Strict transcription schema; no JSON repair or text correction."""
import json

def strict_parse(text):
    def pairs(xs):
        out={}
        for k,v in xs:
            if k in out:raise ValueError('DUPLICATE_KEY')
            out[k]=v
        return out
    try:
        value=json.loads(text,object_pairs_hook=pairs,parse_constant=lambda x:(_ for _ in ()).throw(ValueError('NONFINITE')))
    except (ValueError,TypeError):return None,{'json_valid':False,'schema_valid':False,'errors':['INVALID_JSON']}
    errors=[]
    def obj(v,keys,optional=()):
        if not isinstance(v,dict):errors.append('WRONG_TYPE');return False
        if set(v)-set(keys)-set(optional):errors.append('EXTRA_KEYS')
        if set(keys)-set(v):errors.append('MISSING_KEYS')
        return True
    def atom(v,index):
        if not obj(v,[index,'text','uncertain'],['merged'] if index=='column_index' else []):return
        if type(v.get(index)) is not int or v[index]<0:errors.append('WRONG_INDEX_TYPE')
        if 'merged' in v and type(v['merged']) is not bool:errors.append('WRONG_TYPE')
        if not isinstance(v.get('text'),str) or type(v.get('uncertain')) is not bool:errors.append('WRONG_TYPE')
    if not isinstance(value,dict):errors.append('WRONG_TYPE')
    elif value.get('type')=='table':
        obj(value,['type','rows'],['outside_table_lines']);rows=value.get('rows')
        if 'outside_table_lines' in value:
            outside=value['outside_table_lines']
            if not isinstance(outside,list):errors.append('WRONG_TYPE')
            else:
                for i,line in enumerate(outside):
                    atom(line,'index')
                    if isinstance(line,dict) and line.get('index')!=i:errors.append('NONCONTIGUOUS_LINE_INDEX')
        if not isinstance(rows,list):errors.append('WRONG_TYPE')
        else:
            for i,row in enumerate(rows):
                if not obj(row,['row_index','cells']):continue
                if type(row.get('row_index')) is not int or row['row_index']!=i:errors.append('NONCONTIGUOUS_ROW_INDEX')
                cells=row.get('cells')
                if not isinstance(cells,list):errors.append('WRONG_TYPE');continue
                for j,cell in enumerate(cells):
                    atom(cell,'column_index')
                    if isinstance(cell,dict) and cell.get('column_index')!=j:errors.append('NONCONTIGUOUS_COLUMN_INDEX')
    elif value.get('type')=='lines':
        obj(value,['type','lines']);lines=value.get('lines')
        if not isinstance(lines,list):errors.append('WRONG_TYPE')
        else:
            for i,line in enumerate(lines):
                atom(line,'index')
                if isinstance(line,dict) and line.get('index')!=i:errors.append('NONCONTIGUOUS_LINE_INDEX')
    else:errors.append('UNKNOWN_TYPE')
    return value,{'json_valid':True,'schema_valid':not errors,'errors':(['SCHEMA_MISMATCH']+sorted(set(errors))) if errors else []}


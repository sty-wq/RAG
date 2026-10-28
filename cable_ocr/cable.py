"""Small token parser; formatting never changes non-whitespace characters."""
import re

VOLTAGE=re.compile(r"(?P<voltage>\d+(?:\.\d+)?/\d+(?:\.\d+)?kV)")
GROUP=re.compile(r"\d+\*\d+(?:\.\d+)?")


def parse_cable(raw):
    matches=list(VOLTAGE.finditer(raw))
    base={'raw':raw,'status':'UNKNOWN','voltage':None,'cores':None,'boundary':None,'reasons':[]}
    if len(matches)!=1:
        if len(matches)>1: base.update(status='INVALID',reasons=['MULTIPLE_VOLTAGE_BOUNDARIES'])
        elif any(x in raw for x in ('kV','*','YJV','YJY','YUY')): base.update(status='PARTIAL',reasons=['VOLTAGE_NOT_PARSED'])
        return base
    m=matches[0]; prefix=raw[:m.start()]; tail=raw[m.end():]
    cores=tail.strip()
    base.update(voltage=m['voltage'],cores=cores,boundary=m.end(),prefix=prefix)
    if '\n' in raw or '\r' in raw:
        base.update(status='PARTIAL',reasons=['MULTILINE_MODEL']);return base
    if not prefix or not prefix.endswith('-') or not re.fullmatch(r'[A-Za-z0-9-]+',prefix):
        base.update(status='PARTIAL',reasons=['PREFIX_FORM_UNKNOWN']);return base
    if not cores:
        base.update(status='PARTIAL',reasons=['CORES_MISSING']);return base
    groups=cores.split('+')
    if not all(GROUP.fullmatch(x) for x in groups):
        base.update(status='INVALID',reasons=['CORE_GROUP_INVALID']);return base
    base.update(status='VALID',groups=[{'count':x.split('*')[0],'section':x.split('*')[1]} for x in groups])
    # A hint, never a replacement or a claim that an unseen prefix is invalid.
    if 'YUY' in prefix: base['reasons'].append('OBSERVED_CONFUSION_YUY_REQUIRES_REVIEW')
    return base


def normalize(raw,field):
    result={'text':raw,'status':'unchanged','rule_id':None,'edits':[],'parser':None}
    if field!='规格型号': return result
    parsed=parse_cable(raw);result['parser']=parsed
    if parsed['status']!='VALID':
        result['status']='review_required';return result
    boundary=parsed['boundary']
    # Only insert one space when there is no whitespace at a unique boundary.
    if boundary<len(raw) and not raw[boundary].isspace():
        text=raw[:boundary]+' '+raw[boundary:]
        again=parse_cable(text)
        if ''.join(text.split())==''.join(raw.split()) and again['voltage']==parsed['voltage'] and again['cores']==parsed['cores']:
            result.update(text=text,status='safe_formatting',rule_id='SPACE_BETWEEN_VOLTAGE_AND_CORES',
                          edits=[{'raw_start':boundary,'raw_end':boundary,'insert':' '}])
    return result

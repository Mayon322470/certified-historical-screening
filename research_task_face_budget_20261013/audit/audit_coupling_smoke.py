"""Small independent symbolic and proof replay for shared-beta front end."""
from pathlib import Path
from fractions import Fraction as F
from copy import deepcopy
import hashlib,json,sys,time

sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT.parent/'research_zhao2026_comparison_20261012'
sys.path[:0]=[str(ROOT/'audit'),str(ROOT/'implementation'),str(ROOT/'theory'),str(OLD/'qa')]
import coupling_faces as cf
import audit_coupling_faces as audit
import audit_rank1_ma as am
import audit_rank1_zhao as az
import independent_ma_reference as ref


def encode(x):
    if isinstance(x,F):return str(x)
    if isinstance(x,dict):return {k:encode(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [encode(v) for v in x]
    return x


def main():
    started=time.perf_counter()
    cases=json.loads((ROOT/'protocol/development/public_fixtures.json').read_text())
    symbolic=range_tests=0
    for case in cases:
        meta=cf.prepare(case);faces=audit.reference_faces(case)
        assert len(meta['faces'])==len(faces)==12
        for actual,(name,family,expected) in zip(meta['faces'],faces):
            assert actual['name']==name and actual['family']==family and audit.canonical(actual)==expected
            symbolic+=1
            for lo,hi in ((F(1,2),F(2)),(F(9,10),F(11,10)),(F(1),F(1))):
                for order in range(6):
                    value=cf.evaluate(actual,lo,hi,order)
                    assert ref.contained(audit.qrange(expected,lo,hi,order),(value.lower(),value.upper()))
                    range_tests+=1
    case=dict(name='independent_joint_inconsistency',initial=['0','0','0'],past=[['1','0']],
              prior=['1/2','2'],future=[['1','0']],H='1',rho='0',limit='100',
              response=dict(kind='normalized_global_cubic_v1',epsilon_x='10',L='2'),
              packet=[['9/10','11/10'],['-11/10','-9/10'],['-10','10']])
    certificate=cf.reject_cell(cf.prepare(case),F(1,2),F(2))
    assert certificate is not None and certificate['face_family']=='same_beta'
    assert audit.check_certificate(case,certificate)
    mutations=[]
    for label,key,value in [('nonpositive_claim','certified_lower',F(0)),('wrong_face','face_index',0),
                             ('wrong_parameter_map','q_interval',(F(0),F(2)))]:
        bad=deepcopy(certificate);bad[key]=value
        try:audit.check_certificate(case,bad)
        except (AssertionError,ValueError):mutations.append(label)
        else:raise AssertionError(label)
    bad=deepcopy(certificate);bad['expression']['poly'][2]+=1
    try:audit.check_certificate(case,bad)
    except AssertionError:mutations.append('wrong_expression')
    else:raise AssertionError('wrong_expression')
    boundary=deepcopy(case);boundary['packet']=[['0','1']]*3;boundary['response']['epsilon_x']='0'
    assert cf.reject_cell(cf.prepare(boundary),F(1,2),F(2)) is None
    proofs=json.loads((ROOT/'audit/RANK1_CONTRACT_PROOFS.json').read_text())
    for row in proofs:
        result=(az.check_contract if row['method'].startswith('zhao') else am.check_contract)(row['case'],row['certificate'])
        assert result is None if row['result'] is None else result==tuple(map(F,row['result']))
    sources=[Path(__file__),ROOT/'audit/audit_coupling_faces.py',ROOT/'implementation/coupling_faces.py',
             ROOT/'audit/audit_rank1_ma.py',ROOT/'audit/audit_rank1_zhao.py',ROOT/'audit/RANK1_CONTRACT_PROOFS.json',
             OLD/'qa/independent_ma_reference.py']
    report=dict(status='PASS',scope='PREFREEZE_SMALL_INDEPENDENT_COUPLING_AND_SERIALIZATION_AUDIT',
                symbolic_face_reconstructions=symbolic,derivative_range_comparisons=range_tests,
                same_beta_only_contradiction_checked=True,closed_boundary_not_deleted=True,
                proof_mutations_rejected=mutations,saved_contracts_deserialized_and_replayed=len(proofs),
                source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                elapsed_seconds=time.perf_counter()-started,formal_generator_called=False,
                formal_data_read=False,performance_or_novelty_proved=False)
    path=ROOT/'audit/COUPLING_AND_SERIALIZATION_AUDIT.json'
    with path.open('x',encoding='utf-8') as f:json.dump(encode(report),f,ensure_ascii=False,indent=2)
    with (ROOT/'audit/COUPLING_INDEPENDENT_PROOF.json').open('x',encoding='utf-8') as f:
        json.dump(encode(dict(case=case,certificate=certificate)),f,ensure_ascii=False,indent=2)
    print(json.dumps(encode(report),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

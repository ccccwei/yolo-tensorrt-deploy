import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np

out=Path(__file__).resolve().parent
root=out.parents[1]
sys.path.insert(0,str(root/'tests'))
from topk_cases import validate_detections

selector=out/'selector'
report={'cases':{},'commands':[]}
for case in ['unique','fp16_rounded_ties','high_confidence_ties','threshold_multiclass']:
    scores=np.fromfile(selector/case/'scores.bin',dtype=np.float32).reshape(1,8400,80)
    outputs={}
    records={}
    for label in ['control','grouped']:
        work=selector/(case+'_'+label+'_gpu')
        work.mkdir(exist_ok=False)
        command=['/usr/local/TensorRT/bin/trtexec','--loadEngine='+str(selector/(label+'.engine')),
                 '--loadInputs=boxes:'+str(selector/'boxes.bin')+',scores:'+str(selector/case/'scores.bin'),
                 '--noDataTransfers','--useCudaGraph','--iterations=3','--warmUp=0','--duration=0',
                 '--dumpRawBindingsToFile']
        result=subprocess.run(command,cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=60)
        (work/'stdout.txt').write_text(result.stdout)
        if result.returncode: raise RuntimeError(result.stdout[-1800:])
        paths=list(work.glob('output0.output.*.raw'))
        if len(paths)!=1: raise RuntimeError('Expected one output binding')
        value=np.fromfile(paths[0],dtype=np.float32).reshape(1,300,6)
        outputs[label]=value
        records[label]=validate_detections(scores,value)
        records[label].update(output=str(paths[0].relative_to(out)),sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest())
        report['commands'].append(command)
    records['raw_identical']=bool(np.array_equal(outputs['control'],outputs['grouped']))
    records['equal_rows']=int(np.count_nonzero(np.all(outputs['control']==outputs['grouped'],axis=2)))
    sets=[set(tuple(row) for row in outputs[label][0]) for label in ['control','grouped']]
    records['equal_row_sets']=sets[0]==sets[1]
    records['common_rows']=len(sets[0]&sets[1])
    report['cases'][case]=records
    print(case+': selection contract PASS; raw identical='+str(records['raw_identical'])+', common rows='+str(records['common_rows']),flush=True)
with (out/'selector_gpu.json').open('x') as target: json.dump(report,target,indent=2)

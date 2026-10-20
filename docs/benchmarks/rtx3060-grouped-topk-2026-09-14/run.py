#!/usr/bin/env python3
"""Compare four newly built engines using GPU-only and unchanged 11 pipeline timing."""
import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

LABELS = ['control_a', 'grouped_a', 'grouped_b', 'control_b']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def rows(path):
    with path.open(newline='') as source:
        return list(csv.DictReader(source))

def stats(values):
    values = sorted(values)
    return {'n':len(values), 'mean':sum(values)/len(values),
            **{f'p{p}':values[math.ceil(p/100*len(values))-1] for p in [50,95,99]}}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engines', nargs=4, type=Path, required=True, metavar=('CONTROL_A','GROUPED_A','GROUPED_B','CONTROL_B'))
    parser.add_argument('--inputs', type=Path, required=True, help='Existing bus.bin / zidane.bin FP32 inputs from project preprocessing')
    parser.add_argument('--images', type=Path, default=Path('images'))
    parser.add_argument('--runtime', type=Path, default=Path('build/release/src/yolo_registered_input'))
    parser.add_argument('--trtexec', default='/usr/local/TensorRT/bin/trtexec')
    parser.add_argument('--output', type=Path, required=True, help='New output directory')
    args = parser.parse_args()
    engines = dict(zip(LABELS,(p.resolve() for p in args.engines)))
    inputs,images,runtime,out = (p.resolve() for p in [args.inputs,args.images,args.runtime,args.output])
    for name in ['bus','zidane']:
        if (inputs/(name+'.bin')).stat().st_size != 3*640*640*4:
            raise ValueError('Expected FP32 [1,3,640,640] input')
    out.mkdir(parents=True,exist_ok=False)
    gpu_cmd=['nvidia-smi','--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu','--format=csv']
    files=list(engines.values())+[runtime,Path(__file__).resolve()]+[inputs/(n+'.bin') for n in ['bus','zidane']]+[images/(n+'.jpg') for n in ['bus','zidane']]
    metadata={'sha256':{str(p):sha(p) for p in files},'runs':[],
              'protocol':'3 rounds; each image 1000 measured. Engine:500ms warmup; pipeline11:200 warmups/image, REGISTERED=1. Sequential runs.'}
    totals={label:{'gpu':[],'pipeline':{},'wall_ms':0.0,'samples':0} for label in LABELS}
    expected_detections=None
    expected_outputs={}
    def run(command,work,saved,tag):
        record={'name':tag,'command':command,'gpu_before':subprocess.check_output(gpu_cmd,text=True)}
        result=subprocess.run(command,cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=180)
        (saved/'stdout.txt').write_text(result.stdout)
        record.update(returncode=result.returncode,gpu_after=subprocess.check_output(gpu_cmd,text=True))
        metadata['runs'].append(record)
        (out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(tag+' failed: '+result.stdout[-2000:])
        return record
    with tempfile.TemporaryDirectory(prefix='yolo-topk-benchmark-') as temporary:
        temp=Path(temporary)
        # The pipeline sees exactly the same two images, even if the source folder grows.
        fixture=temp/'images'
        fixture.mkdir()
        for name in ['bus.jpg','zidane.jpg']:
            shutil.copyfile(images/name,fixture/name)
        for round_index in range(3):
            order=LABELS[round_index:]+LABELS[:round_index]
            if round_index%2: order=list(reversed(order))
            for label in order:
                for name in ['bus','zidane']:
                    tag=f'r{round_index+1}_{label}_{name}_engine'
                    saved=out/tag; saved.mkdir()
                    work=temp/tag; work.mkdir()
                    command=[args.trtexec,'--loadEngine='+str(engines[label]),'--loadInputs=images:'+str(inputs/(name+'.bin')),
                             '--noDataTransfers','--useCudaGraph','--infStreams=1','--warmUp=500','--duration=0','--iterations=1000',
                             '--exportTimes='+str(work/'times.json'),'--exportOutput='+str(saved/'output.json')]
                    run(command,work,saved,tag)
                    trace=json.loads((work/'times.json').read_text())
                    if len(trace)!=1000 or any(t['h2dMs']!=0 or t['d2hMs']!=0 or not math.isfinite(t['computeMs']) or t['computeMs']<=0 for t in trace):
                        raise RuntimeError('Bad GPU timing rows')
                    with gzip.open(saved/'times.json.gz','wb') as target:
                        target.write((work/'times.json').read_bytes())
                    values=json.loads((saved/'output.json').read_text())[0]['values']
                    if name in expected_outputs and values!=expected_outputs[name]:
                        raise RuntimeError('Saved engine output differs: '+tag)
                    expected_outputs[name]=values
                    totals[label]['gpu'].extend(t['computeMs'] for t in trace)
                    print(tag+f": GPU P50 {stats([t['computeMs'] for t in trace])['p50']:.6f} ms",flush=True)
                tag=f'r{round_index+1}_{label}_pipeline11'
                saved=out/tag; saved.mkdir()
                work=temp/tag; work.mkdir()
                command=[str(runtime),str(engines[label]),str(fixture),'1000','200','1']
                run(command,work,saved,tag)
                result=work/'results/11_registered_input'
                for path in result.glob('*.csv'): shutil.copyfile(path,saved/path.name)
                trace=rows(saved/'timings.csv')
                ids={(r['image'],int(r['iteration'])) for r in trace}
                if len(trace)!=2000 or ids!={(name,i) for name in ['bus.jpg','zidane.jpg'] for i in range(1,1001)}:
                    raise RuntimeError('Bad pipeline sample IDs')
                for field in ['staging_ms','gpu_ms','postprocess_ms','total_ms']:
                    values=[float(r[field]) for r in trace]
                    if not all(math.isfinite(v) and v>=0 for v in values): raise RuntimeError('Invalid timing')
                    totals[label]['pipeline'].setdefault(field,[]).extend(values)
                throughput=rows(saved/'throughput.csv')[0]
                wall=float(throughput['wall_ms'])
                if int(throughput['samples'])!=2000 or wall<=0 or not math.isclose(float(throughput['fps']),2000000/wall,rel_tol=1e-7):
                    raise RuntimeError('Bad throughput')
                totals[label]['samples']+=2000; totals[label]['wall_ms']+=wall
                detections=(saved/'detections.csv').read_bytes()
                if expected_detections is not None and detections!=expected_detections:
                    raise RuntimeError('Saved pipeline detections differ: '+tag)
                expected_detections=detections
                print(tag+f": pipeline P50 {stats([float(t['total_ms']) for t in trace])['p50']:.6f} ms",flush=True)
    summary={label:{'engine_gpu_ms':stats(item['gpu']),
                    'pipeline_ms':{k:stats(v) for k,v in item['pipeline'].items()},
                    'pipeline_fps':item['samples']*1000/item['wall_ms'], 'pipeline_wall_ms':item['wall_ms']}
             for label,item in totals.items()}
    if any(sha(Path(path))!=digest for path,digest in metadata['sha256'].items()):
        raise RuntimeError('An input/engine/runtime changed while benchmarking')
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('Saved 24000 engine-only + 24000 pipeline samples; all final outputs identical.',flush=True)

if __name__=='__main__':
    main()

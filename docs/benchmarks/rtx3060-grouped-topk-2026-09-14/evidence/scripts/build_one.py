"""Build a new experiment engine, recording command, log, cache and model hashes."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

out = Path(__file__).resolve().parent
source, label = Path(sys.argv[1]).resolve(), sys.argv[2]
engine = out / (label + '.engine')
cache = out / 'shared.cache'
if engine.exists() or (out / (label + '.build.json')).exists():
    raise SystemExit('Refusing to overwrite an existing engine/build record')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
command = ['/usr/local/TensorRT/bin/trtexec', '--onnx=' + str(source), '--saveEngine=' + str(engine),
           '--fp16', '--memPoolSize=workspace:1024', '--skipInference',
           '--profilingVerbosity=detailed', '--builderOptimizationLevel=3', '--timingCacheFile=' + str(cache),
           '--exportLayerInfo=' + str(out / (label + '.layers.json'))]
gpu = ['nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,utilization.gpu', '--format=csv']
record = {'command':command, 'source_sha256':sha(source), 'cache_before':sha(cache),
          'gpu_before':subprocess.check_output(gpu,text=True), 'started_unix':time.time()}
print('Building ' + label + ' from ' + str(source), flush=True)
start = time.monotonic()
with (out / (label + '.build.log')).open('x') as log:
    done = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
record.update(returncode=done.returncode, wall_s=time.monotonic()-start,
              engine_sha256=sha(engine), cache_after=sha(cache),
              gpu_after=subprocess.check_output(gpu,text=True))
(out / (label + '.build.json')).write_text(json.dumps(record,indent=2)+'\n')
print(label + ': return=' + str(done.returncode) + ', wall_s=' + str(round(record['wall_s'],2)), flush=True)
if done.returncode:
    print((out / (label + '.build.log')).read_text()[-4000:])
raise SystemExit(done.returncode)

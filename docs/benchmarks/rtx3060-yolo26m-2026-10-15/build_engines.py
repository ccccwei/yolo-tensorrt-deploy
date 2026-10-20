#!/usr/bin/env python3
"""Build a control/grouped pair sequentially from independent empty timing caches."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trtexec', default='/usr/local/TensorRT/bin/trtexec')
    args = parser.parse_args()
    models, output = args.models.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    for label in ('control', 'grouped'):
        if not (models / (label + '.onnx')).is_file():
            raise FileNotFoundError(label + '.onnx')
        if list(output.glob(label + '.*')):
            raise FileExistsError('Refusing to overwrite build artifacts for ' + label)
    gpu = ['nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu', '--format=csv']
    for label in ('control', 'grouped'):
        source = models / (label + '.onnx')
        engine, cache = output / (label + '.engine'), output / (label + '.cache')
        command = [args.trtexec, '--onnx=' + str(source), '--saveEngine=' + str(engine),
                   '--fp16', '--memPoolSize=workspace:1024', '--skipInference',
                   '--profilingVerbosity=detailed', '--builderOptimizationLevel=3',
                   '--timingCacheFile=' + str(cache),
                   '--exportLayerInfo=' + str(output / (label + '.layers.json'))]
        record = {'label': label, 'command': command, 'source_sha256': sha(source),
                  'initial_cache_sha256': sha(cache), 'initial_cache': 'empty',
                  'started_utc': datetime.now(timezone.utc).isoformat(),
                  'gpu_before': subprocess.check_output(gpu, text=True)}
        start = time.monotonic()
        print('Building ' + label, flush=True)
        with (output / (label + '.build.log')).open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=1200)
        record.update(returncode=result.returncode, wall_s=time.monotonic() - start,
                      engine_sha256=sha(engine), final_cache_sha256=sha(cache),
                      gpu_after=subprocess.check_output(gpu, text=True))
        (output / (label + '.build.json')).write_text(json.dumps(record, indent=2) + '\n')
        print(f'{label}: return={result.returncode}, wall_s={record["wall_s"]:.1f}', flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()

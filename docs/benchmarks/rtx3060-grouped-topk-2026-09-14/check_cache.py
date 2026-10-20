#!/usr/bin/env python3
"""Supplementary engine-only check: two builds start from identical cache copies."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = sorted(values)
    return {'n': len(values), 'mean': sum(values) / len(values),
            **{f'p{p}': values[math.ceil(p / 100 * len(values)) - 1] for p in [50, 95, 99]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['control', 'grouped', 'cache', 'input', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    model = {'control': args.control.resolve(), 'grouped': args.grouped.resolve()}
    cache, tensor = args.cache.resolve(), args.input.resolve()
    metadata = {'base_cache_sha256': sha(cache), 'input_sha256': sha(tensor),
                'script_sha256': sha(Path(__file__)), 'commands': [], 'builds': {}}
    gpu = ['nvidia-smi', '--query-gpu=temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu', '--format=csv']
    executable = '/usr/local/TensorRT/bin/trtexec'

    def run(command, tag):
        record = {'tag': tag, 'command': command,
                  'gpu_before': subprocess.check_output(gpu, text=True)}
        with (out / (tag + '.log')).open('x') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
        record.update(returncode=result.returncode, gpu_after=subprocess.check_output(gpu, text=True))
        metadata['commands'].append(record)
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
        if result.returncode:
            raise RuntimeError(tag + ' failed; see log')

    for label, source in model.items():
        local_cache = out / (label + '.cache')
        shutil.copyfile(cache, local_cache)
        metadata['builds'][label] = {'source_sha256': sha(source), 'cache_before': sha(local_cache)}
        run([executable, '--onnx=' + str(source), '--saveEngine=' + str(out / (label + '.engine')),
             '--fp16', '--memPoolSize=workspace:1024', '--skipInference',
             '--profilingVerbosity=detailed', '--builderOptimizationLevel=3',
             '--timingCacheFile=' + str(local_cache), '--exportLayerInfo=' + str(out / (label + '.layers.json'))], label + '_build')
        metadata['builds'][label].update(cache_after=sha(local_cache), engine_sha256=sha(out / (label + '.engine')))
        print('Built ' + label + ' from independent base-cache copy', flush=True)

    samples = {label: [] for label in model}
    expected = None
    for round_id, order in enumerate([['control', 'grouped'], ['grouped', 'control']], 1):
        for label in order:
            tag = f'r{round_id}_{label}'
            run([executable, '--loadEngine=' + str(out / (label + '.engine')),
                 '--loadInputs=images:' + str(tensor), '--useCudaGraph', '--noDataTransfers',
                 '--infStreams=1', '--warmUp=500', '--duration=0', '--iterations=1000',
                 '--exportTimes=' + str(out / (tag + '.times.json')),
                 '--exportOutput=' + str(out / (tag + '.output.json'))], tag)
            trace = json.loads((out / (tag + '.times.json')).read_text())
            assert len(trace) == 1000
            assert all(t['h2dMs'] == t['d2hMs'] == 0 and math.isfinite(t['computeMs']) and t['computeMs'] > 0 for t in trace)
            samples[label].extend(t['computeMs'] for t in trace)
            values = json.loads((out / (tag + '.output.json')).read_text())[0]['values']
            assert expected is None or values == expected
            expected = values
            print(tag + ': ' + str(stats([t['computeMs'] for t in trace])), flush=True)
    summary = {label: stats(values) for label, values in samples.items()}
    summary['p50_latency_reduction_percent'] = (1 - summary['grouped']['p50'] / summary['control']['p50']) * 100
    summary['inspector_json_equal'] = json.loads((out / 'control.layers.json').read_text()) == json.loads((out / 'grouped.layers.json').read_text())
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    # Diagnostic only: never pool instrumented layer timings into the measurements above.
    for label in model:
        run([executable, '--loadEngine=' + str(out / (label + '.engine')),
             '--loadInputs=images:' + str(tensor), '--useCudaGraph', '--noDataTransfers',
             '--infStreams=1', '--warmUp=200', '--duration=0', '--iterations=500',
             '--separateProfileRun', '--dumpProfile', '--exportProfile=' + str(out / (label + '.profile.json'))], label + '_profile')
    assert sha(cache) == metadata['base_cache_sha256']
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

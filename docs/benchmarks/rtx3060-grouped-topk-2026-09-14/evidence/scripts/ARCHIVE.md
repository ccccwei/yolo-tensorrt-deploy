# Original experiment scripts

These six files are byte-for-byte evidence copies from
`artifacts/grouped-topk-IAqxawtr/`. Their original paths and SHA-256 hashes are
recorded in [`../manifest.json`](../manifest.json).

They preserve the scripts actually used for the experiment, including original
absolute paths and assumptions about their parent directory. They are **not
relocatable entry points**: copying them into this archive does not make them
directly runnable here. Follow the experiment report's reproduction instructions
and resolve the original dependencies and working directories before running
any script. Some scripts build or run GPU engines and create output files.

The accompanying evidence stores model outputs, comparison results, build logs,
layer descriptions, and fixture metadata. Model files, serialized engines,
images, input tensors, timing caches, executables, failed attempts, and the
separate `fixed_cache` experiment are deliberately excluded. Compressed evidence
uses deterministic gzip encoding; the manifest records both stored and original
content hashes.

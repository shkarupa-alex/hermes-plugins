# Russian ASR release corpus

The corpus intentionally contains one real VK voice message supplied by the
repository owner for ASR regression testing. The committed manifest records
its immutable SHA-256, duration, reference transcript, provenance, and usage
permission. No VK token, attachment URL, or access key is stored in the
repository.

`baseline.json` is the accepted result for the pinned default model. The
release workflow reruns recognition and rejects WER regressions greater than
two absolute percentage points or keyword-recall regressions greater than one
absolute percentage point.

# Pre-checkpoint interruption

On 19 October 2026, neither the recorded supervisor PID 83608 nor trainer PID
13964 was present in `ps`; the former execution session 23314 was also missing.
No other `models.train` or `models.suite` process was live. The stale state file
still said running, which was not treated as authoritative process evidence.

The original run directory had config.json, invocations.jsonl, an empty
history.jsonl, and no last.pt/best.pt. The last original log entry was epoch 1,
step 200. There was no logged exception, so the interruption cause is unknown.

This directory preserves that attempt before restarting epoch 1 using the
unchanged size, seed, learning rate, corpus, microbatch and accumulation settings.
The completed 100K condition was retained and skipped by the queue's verifier.

Original final progress lines:

```text
epoch=1 step=50 train_loss=1.1595 train_accuracy=0.3581
epoch=1 step=100 train_loss=1.1385 train_accuracy=0.3459
epoch=1 step=150 train_loss=1.1282 train_accuracy=0.3446
epoch=1 step=200 train_loss=1.1249 train_accuracy=0.3394
```

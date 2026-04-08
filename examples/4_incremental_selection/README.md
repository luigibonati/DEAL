## Incremental selection 

In addition to the standard mode which requires to specify a threshold, one can also run in an incremental mode, that is, running DEAL in multiple iterations with progressively
lower thresholds, so you can stop when a target number of selected structures is reached instead of picking a single threshold upfront. 

### Run incremental DEAL with CLI

Incremental selection is now integrated in the CLI.

Run directly with defaults (`max_iterations=10`, with a decay `threshold_factor=0.75`):

```bash
deal --file input/input_fcu.xyz.gz --max-selected 50
```

**Output example**
```
[DEAL] Running in incremental mode with max_selected = 50.

[DEAL] Iteration 1 (threshold: 0.75)
[DEAL] Examined:   200 | Selected:     8 | Speed:   0.46 s/step | Elapsed:   104.98 s
[DEAL] New selected: 8 

[DEAL] Iteration 2 (threshold: 0.562)
[DEAL] Examined:   192 | Selected:    11 | Speed:   0.39 s/step | Elapsed:   199.48 s
[DEAL] New selected: 3 

[DEAL] Iteration 3 (threshold: 0.422)
[DEAL] Examined:   189 | Selected:    19 | Speed:   0.44 s/step | Elapsed:   340.25 s
[DEAL] New selected: 8 

[DEAL] Iteration 4 (threshold: 0.316)
[DEAL] Examined:   181 | Selected:    32 | Speed:   0.42 s/step | Elapsed:   577.46 s
[DEAL] New selected: 13 

[DEAL] Iteration 5 (threshold: 0.237)
[DEAL] Examined:   168 | Selected:    46 | Speed:   0.44 s/step | Elapsed:   909.93 s
[DEAL] New selected: 14 

[DEAL] Iteration 6 (threshold: 0.178)
[DEAL] Examined:   154 | Selected:    62 | Speed:   0.46 s/step | Elapsed:  1389.57 s
[DEAL] New selected: 16 

[DEAL] Stopping incremental mode: max_selected is reached.
```

### Customize input

If you want to customize the settings, create a YAML file `input.yaml`:

```yaml
data:
  files: "input/input_fcu.xyz.gz"
  shuffle: true
  seed: 42

deal:
  max_selected: 50
  max_iterations: 10
  threshold_factor: 0.75
  output_prefix: "deal_incremental"
```

and run: 

```bash
deal -c input.yaml
```

### Deprecated Python script

This functionality was first implemented as example in `incremental_DEAL.py`. This is kept as a deprecated/legacy example showing how to implement
incremental selection manually with `configure_run(...)` and in-memory `images`.

You can still run it with:

```bash
python incremental_DEAL.py
```

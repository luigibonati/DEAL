## Incremental selection 

Incremental selection means running DEAL in multiple iterations with progressively
lower thresholds, so you can stop when a target number of selected structures is
reached instead of picking a single threshold upfront. 

### Run incremental DEAL with CLI

Incremental selection is now integrated in the CLI.

Run directly with defaults (`max_iterations=10`, starting from `threshold_factor=0.75`):

```bash
deal --file input/input_fcu.xyz --max-selected 50
```

If you want to customize the settings, `b_selection`, save the YAML below as `input.yaml` and run:

```yaml
data:
  files: "input/input_fcu.xyz"

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

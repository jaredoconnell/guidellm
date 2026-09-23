# Trace replay profiling runbook

Investigation artifacts in this directory. Not product code. Two passes stay
separate: Pass A measures send-time accuracy; Pass B attributes CPU (and
distorts those send times).

## First test (local, Pass A, mock, OTEL stall fixture)

This is the smallest run that exercises the full pipeline: zero-latency mock
server, `otel_stall.jsonl` (gap then `quick_burst`), JSON report, delay
classification.

From the repo root, with `.venv` already set up:

```bash
bash scripts/trace_profile/run_pass_a_local.sh
```

WEKA stall fixture instead:

```bash
bash scripts/trace_profile/run_pass_a_local.sh weka
```

What you should see:

- Mock server on `127.0.0.1:8000`
- `guidellm run --profile kind=replay` against `otel_stall.jsonl` or `weka_stall.jsonl`
- Report at `scripts/trace_profile/out/<format>_stall_mock.json`
- CSV + stdout summary at `scripts/trace_profile/out/<format>_stall_mock_compare.csv`

Read the summary line `dispatch_delay client_late`. Single-digit-ms p99 on a
fast mock means the client kept up. Late `burst_after_gap` rows with
`concurrency_bound` and `sleeping_future_count > 0` support the
slot-held-by-future-sleep hypothesis.

Manual equivalent (if you want to watch the mock in another terminal):

```bash
.venv/bin/guidellm mock-server \
  --host 127.0.0.1 --port 8000 \
  --ttft-ms 0 --itl-ms 0 --output-tokens 1 --request-latency 0

.venv/bin/guidellm run \
  --backend kind=openai_http,target=http://127.0.0.1:8000 \
  --profile kind=replay,time_scale=1.0 \
  --data kind=otel,source.kind=json_file,source.path=otel_stall.jsonl,time_scale=1.0 \
  --disable-progress \
  --output kind=json,path=scripts/trace_profile/out/otel_stall_mock.json

.venv/bin/python scripts/trace_profile/compare_replay_timings.py \
  --benchmarks scripts/trace_profile/out/otel_stall_mock.json \
  --trace otel_stall.jsonl \
  --format otel \
  --backend mock \
  --output scripts/trace_profile/out/otel_stall_mock_compare.csv
```

Leave `GUIDELLM_PROFILE` unset for Pass A.

## Pass A against a real vLLM (local or in-cluster)

Same client flags, different target. Keep `--disable-progress` and write JSON.

```bash
.venv/bin/guidellm run \
  --backend kind=openai_http,target=http://REPLACE_VLLM:8000 \
  --profile kind=replay,time_scale=1.0 \
  --data kind=otel,source.kind=json_file,source.path=otel_stall.jsonl,time_scale=1.0 \
  --disable-progress \
  --output kind=json,path=scripts/trace_profile/out/otel_stall_vllm.json

.venv/bin/python scripts/trace_profile/compare_replay_timings.py \
  --benchmarks scripts/trace_profile/out/otel_stall_vllm.json \
  --trace otel_stall.jsonl --format otel --backend vllm \
  --output scripts/trace_profile/out/otel_stall_vllm_compare.csv
```

Compare `client_late` p99 mock vs vLLM:

- High on mock, low on vLLM → mock was the limiter
- High on both → client scheduler; go to Pass B
- Low on mock, high on vLLM → serve queueing (`server_bound`)

## Pass B (line / function CPU, local)

Do this only after Pass A. Profiles go next to the report; send delays in this
run are contaminated.

```bash
export PYTHONPATH="$PWD/scripts/trace_profile${PYTHONPATH:+:$PYTHONPATH}"
export GUIDELLM_PROFILE=cprofile   # or pyinstrument, or line
export GUIDELLM_PROFILE_DIR="$PWD/scripts/trace_profile/out/profiles"
export GUIDELLM__MAX_WORKER_PROCESSES=1
mkdir -p "$GUIDELLM_PROFILE_DIR"

# optional, not in the repo:
# .venv/bin/pip install pyinstrument line_profiler

# start mock-server as in Pass A, then:
.venv/bin/guidellm run \
  --backend kind=openai_http,target=http://127.0.0.1:8000 \
  --profile kind=replay,time_scale=1.0 \
  --data kind=otel,source.kind=json_file,source.path=otel_stall.jsonl,time_scale=1.0 \
  --disable-progress \
  --metrics kind=generative,sample_size=0 \
  --output kind=json,path=scripts/trace_profile/out/otel_stall_profiled.json
```

Each process writes `pid-<pid>.txt` (and `.prof` / `.html` / `.lprof.txt`).
Rank cProfile by `tottime`, not `cumtime` (`asyncio.sleep` dominates wall time).

`line` wraps scheduler/deserializer/HTTP methods listed in `sitecustomize.py`.

## OpenShift

1. Stage traces onto the PVC (create the PVC first if needed):

   ```bash
   oc apply -f scripts/trace_profile/guidellm-trace-profile-job.yaml
   # wait for PVC, then copy into a throwaway pod, or:
   oc create configmap guidellm-trace-profile \
     --from-file=sitecustomize.py=scripts/trace_profile/sitecustomize.py \
     --from-file=compare_replay_timings.py=scripts/trace_profile/compare_replay_timings.py
   ```

   Copy JSONL to `traces/otel.jsonl` (and `traces/weka.jsonl`) on the PVC. The
   Jobs mount PVC `subPath: traces` at `/traces`.

2. Mock Job needs no GPU. vLLM Job: set
   `GUIDELLM__SPEC__BACKEND__TARGET` to the in-cluster Service URL.

3. `oc logs -f job/guidellm-trace-profile-mock -c guidellm`

4. Copy results: `/results/pass-a-mock/compare.csv` (and `pass-a-vllm`).

5. Pass B on OpenShift: uncomment `GUIDELLM_PROFILE=cprofile` in the Job,
   optionally `pip install --target "$HOME/.local" pyinstrument line_profiler`
   at the start of the guidellm container script (home is an emptyDir). Do not
   use py-spy; restricted-v2 has no `SYS_PTRACE`.

Swap `--data kind=weka,...path=/traces/weka.jsonl` for WEKA. Pin the image tag
to the GuideLLM build you are investigating; `:v0.7.3` in the YAML is a
placeholder.

## Classification columns

See `compare_replay_timings.py`. Delay is `request_start - targeted_start`.
`targeted_start` is already `start_time + profile.time_scale * relative_timestamp`
(data `time_scale` is already in `relative_timestamp`). Trace join is a greedy
offset match and is only a sanity check; classification uses request timings
and DAG `parent_node_ids`.

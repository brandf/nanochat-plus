#!/bin/bash

# The $10 tier of nanochat
# Designed for a single A100 (80GB) leased at ~$1.70/hour, leaving ~6 hours of wall time.
# Mirrors speedrun.sh/run1000.sh but pares everything down to a 12-layer model.

# -----------------------------------------------------------------------------
# Environment / tooling setup
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR
# Python toolchain via uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate
# wandb defaults (dummy skips logging)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer + dataset
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
uv run maturin develop --release --manifest-path rustbpe/Cargo.toml
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# We only need ~1B characters to bootstrap the tokenizer for this tiny model.
TARGET_TOKENIZER_SHARDS=4
TARGET_PRETRAIN_SHARDS=24
DATA_DIR="$NANOCHAT_BASE_DIR/base_data"
count_shards() {
    if [ -d "$DATA_DIR" ]; then
        find "$DATA_DIR" -maxdepth 1 -name 'shard_*.parquet' 2>/dev/null | wc -l | tr -d ' '
    else
        echo 0
    fi
}
CURRENT_SHARDS=$(count_shards)
if [ "$CURRENT_SHARDS" -lt "$TARGET_TOKENIZER_SHARDS" ]; then
    python -m nanochat.dataset -n $TARGET_TOKENIZER_SHARDS
else
    echo "Found $CURRENT_SHARDS shards (>= $TARGET_TOKENIZER_SHARDS); skipping tokenizer bootstrap download."
fi
CURRENT_SHARDS=$(count_shards)
# Kick off the rest of the pretraining shards (24 total -> ~6B chars -> ~1.2B tokens @4.8 chars/token).
if [ "$CURRENT_SHARDS" -lt "$TARGET_PRETRAIN_SHARDS" ]; then
    python -m nanochat.dataset -n $TARGET_PRETRAIN_SHARDS &
    DATASET_DOWNLOAD_PID=$!
else
    echo "Found $CURRENT_SHARDS shards (>= $TARGET_PRETRAIN_SHARDS); dataset already prepared."
    DATASET_DOWNLOAD_PID=""
fi

TOKENIZER_DIR="$NANOCHAT_BASE_DIR/tokenizer"
TOKENIZER_PKL="$TOKENIZER_DIR/tokenizer.pkl"
TOKEN_BYTES="$TOKENIZER_DIR/token_bytes.pt"
if [ ! -f "$TOKENIZER_PKL" ] || [ ! -f "$TOKEN_BYTES" ]; then
    python -m scripts.tok_train --max_chars=1000000000
    python -m scripts.tok_eval
else
    echo "Tokenizer cache found in $TOKENIZER_DIR; skipping tok_train/tok_eval."
fi

# Base model needs all shards available
echo "Waiting for dataset download to finish..."
if [ -n "$DATASET_DOWNLOAD_PID" ]; then
    wait $DATASET_DOWNLOAD_PID
fi

# -----------------------------------------------------------------------------
# Base model (pretraining)

# Scaling notes for this $10 run:
# - depth=12 => d_model=768 and roughly 1.2e8 parameters (GPT-2 small sized).
# - Each token costs ~6.4 * params ≈ 7.7e8 FLOPs (matching base_train's estimator).
# - target_param_data_ratio=8 => ~9.6e8 tokens => ~7.4e17 total FLOPs (~1.9h of steady A100 @1.1e14 FLOPs/s).
# - VRAM estimate: activations dominate, roughly ~(layers * batch * seq * hidden) scaled by attention caches.
#   Past single-GPU nanochat runs show device_batch_size≈40 drives ~65-70GB allocated (per torch.cuda.max_memory_allocated),
#   leaving just enough headroom for tokenization buffers and optimizer state sync on an 80GB A100.
# - Add tokenizer/mid/sft/evals and we stay below 6h, i.e. <$10 at $1.70/hour.

NPROC_PER_NODE=1

torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_train -- \
    --depth=12 \
    --device_batch_size=40 \
    --total_batch_size=327680 \
    --target_param_data_ratio=8 \
    --eval_every=200 \
    --eval_tokens=2097152 \
    --core_metric_every=-1 \
    --sample_every=400 \
    --warmup_ratio=0.02 \
    --warmdown_ratio=0.2 \
    --final_lr_frac=0.05 \
    --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_loss
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.base_eval

# -----------------------------------------------------------------------------
# Midtraining (tool-use + multi-choice adapters)
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.mid_train -- \
    --device_batch_size=8 \
    --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i mid

# -----------------------------------------------------------------------------
# Supervised finetuning
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_sft -- \
    --device_batch_size=4 \
    --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC_PER_NODE -m scripts.chat_eval -- -i sft

# -----------------------------------------------------------------------------
# Final report + chat entrypoints
python -m nanochat.report generate
# python -m scripts.chat_cli -p "Why is this run only $10?"
# python -m scripts.chat_web

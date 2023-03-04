from jax import random
import jax.numpy as jnp
from nanogpt_jax import NanoGPT
from nanogpt_jax.utils import (
    vocab_size,
    get_batch,
    train_step,
    estimate_loss,
    decode,
    generate,
)
from tqdm import tqdm
import optax

# batch_size = 64
# block_size = 256
# n_emb = 384
# n_head = 6
# n_layer = 6
# max_iters = 5000
# eval_interval = 500
# eval_iters = 50
# dropout_rate = 0.2

# Use following hyper-parameters for local testing
batch_size = 8
block_size = 16
n_emb = 36
n_head = 1
n_layer = 1
max_iters = 5001
eval_interval = 100
eval_iters = 10
dropout_rate = 0.2


jax_key = random.PRNGKey(10061)
model = NanoGPT(
    jax_key, vocab_size, n_emb, block_size, n_head, n_layer, dropout_rate=dropout_rate
)

optim = optax.adam(3e-4)
optim_state = optim.init(model)

# training
train_key = random.PRNGKey(59131)
train_key = random.split(train_key, 1)
for i in tqdm(range(max_iters)):
    train_key = random.split(train_key[0], 2)
    inp, targets = get_batch("train", train_key[0], batch_size, block_size)
    model, optim_state, loss = train_step(
        model, train_key[1], inp, targets, optim, optim_state, is_training=True
    )

    if i % eval_interval == 0:
        losses = estimate_loss(model, train_key, batch_size, block_size, eval_iters)
        print(
            f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        print("---------------------------------------------------------------------")
        context = jnp.zeros((1, 1), dtype=jnp.int32)
        gen = generate(model, block_size, vocab_size, context, tok_num=100)
        gen = gen[0].tolist()
        output = decode(gen)
        print(output)
        print("---------------------------------------------------------------------")

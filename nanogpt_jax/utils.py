import jax.numpy as jnp
from jax import random
import numpy as np
import equinox as eqx
from tqdm import tqdm
import jax

with open("data/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join([itos[i] for i in l])

data = jnp.asarray(encode(text), dtype=jnp.float32)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


def get_batch(split, key, batch_size, block_size):
    # generate a small batch
    data = train_data if split == "train" else val_data
    ix = random.randint(key, (batch_size,), 0, len(data) - block_size)
    x = jnp.stack([data[i : i + block_size] for i in ix], axis=0, dtype=jnp.int16)
    y = jnp.stack([data[i + 1 : i + block_size + 1] for i in ix], dtype=jnp.int16)
    return x, y


def jcross_entropy(logits, targets):
    n = len(targets)
    logits_maxes = logits.max(1, keepdims=True)
    norm_logits = logits - logits_maxes  # subtract max for numerical stability
    counts = jnp.exp(norm_logits)
    counts_sum = counts.sum(axis=1, keepdims=True)
    counts_sum_inv = counts_sum**-1
    probs = counts * counts_sum_inv
    logprobs = jnp.log(probs)
    loss = -logprobs[jnp.arange(n), targets].mean()
    return loss


def estimate_loss(model, key, batch_size, block_size, eval_iters):
    out = {}
    for split in ["train", "val"]:
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            key = random.split(key[0])
            X, Y = get_batch(split, key[0], batch_size, block_size)
            losses[k] = eval_loss_fn(model, key[0], X, Y, is_training=True)
        out[split] = losses.mean()
    return out


def eval_loss_fn(model, key, inp, targets, is_training):
    logits = model(inp, key, is_training)
    B, T, C = logits.shape
    logits = jnp.reshape(logits, (B * T, C))
    targets = jnp.reshape(targets, (B * T))
    loss = jcross_entropy(logits, targets)
    return loss


@eqx.filter_value_and_grad
def loss_fn(model, key, inp, targets, is_training):
    logits = model(inp, key, is_training)
    B, T, C = logits.shape
    logits = jnp.reshape(logits, (B * T, C))
    targets = jnp.reshape(targets, (B * T))
    loss = jcross_entropy(logits, targets)
    return loss


@eqx.filter_jit()
def train_step(model, key, inp, targets, optim, optim_state, is_training):
    loss, grads = loss_fn(model, key, inp, targets, is_training)
    updates, optim_state = optim.update(grads, optim_state)
    model = eqx.apply_updates(model, updates)
    return model, optim_state, loss


def generate(model, block_size, vocab_size, idx, tok_num):
    train_key = random.PRNGKey(517351)
    train_key = random.split(train_key, 1)
    source = jnp.arange(vocab_size)
    # idx is (B,T) array of indices in the current context

    for index in tqdm(range(tok_num)):
        train_key = random.split(train_key[0], 2)
        # crop idx to the last block_size tokens
        idx_cond = idx[:, -block_size:]
        logits = model(idx_cond, train_key[0], is_training=False)
        logits = logits[:, -1, :]  # (B,C)
        probs = jax.nn.softmax(logits, axis=-1)
        # sample from the distribution
        idx_next = jnp.asarray(
            random.choice(train_key[1], a=source, p=probs[0])
        ).reshape(1, 1)
        # add samples index to the running sequence
        idx = jnp.concatenate((idx, idx_next), axis=1)
    return idx

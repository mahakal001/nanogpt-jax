import jax
import jax.numpy as jnp
from jax import random
from jax import jit
from typing import List, Tuple
import optax
import equinox as eqx
from equinox import static_field


class Head(eqx.Module):
    query: jnp.ndarray
    key: jnp.ndarray
    value: jnp.ndarray
    tril: jnp.ndarray = static_field()
    dropout_rate: float = static_field()

    def __init__(self, jax_key, head_size, n_emb, block_size, dropout_rate):
        keys = random.split(jax_key, 3)
        self.query = random.uniform(keys[0], (n_emb, head_size)) * (1 / n_emb**0.5)
        self.key = random.uniform(keys[1], (n_emb, head_size)) * (1 / n_emb**0.5)
        self.value = random.uniform(keys[2], (n_emb, head_size)) * (1 / n_emb**0.5)
        self.tril = jnp.tril(jnp.ones((block_size, block_size)))
        self.dropout_rate = dropout_rate

    def __call__(self, x, key, is_training):
        B, T, C = x.shape

        k = x @ self.key  # B, T, H
        q = x @ self.query  # B, T, H
        wei = q @ k.transpose(0, 2, 1)  # B, T, T
        wei = jnp.where(self.tril[:T, :T] == 0, -jax.numpy.inf, wei)  # B, T, T
        wei = jax.nn.softmax(wei, axis=-1)  # B, T, T
        if is_training:
            wei *= random.bernoulli(key, 1 - self.dropout_rate, wei.shape)  # dropout
            wei /= 1 - self.dropout_rate
        # perform the weighted aggregation of the values
        v = x @ self.value  #  B, T, H
        out = wei @ v  # B, T, H
        return out


class MultiHeadAttention(eqx.Module):
    heads: List[eqx.Module]
    proj: jnp.ndarray
    dropout_rate: float = static_field()

    def __init__(self, jax_key, head_size, n_emb, block_size, num_heads, dropout_rate):
        keys = random.split(jax_key, 1 + num_heads)
        self.heads = [
            Head(keys[index], head_size, n_emb, block_size, dropout_rate)
            for index in range(num_heads)
        ]
        self.proj = random.uniform(keys[-1], (n_emb, n_emb)) * (1 / n_emb**0.5)
        self.dropout_rate = dropout_rate

    def __call__(self, x, key, is_training):
        keys = random.split(key, 1 + len(self.heads))
        out = jnp.concatenate(
            [h(x, k, is_training) for h, k in zip(self.heads, keys)], axis=-1
        )  # B, T, E
        out = out @ self.proj  # B, T, E
        if is_training:
            out *= random.bernoulli(
                keys[-1], 1 - self.dropout_rate, out.shape
            )  # dropout
            out /= 1 - self.dropout_rate
        return out


class FeedForward(eqx.Module):
    l1: jnp.ndarray
    l2: jnp.ndarray
    dropout_rate: float = static_field()

    def __init__(self, jax_key, n_emb, dropout_rate):
        keys = random.split(jax_key, 2)
        self.l1 = random.uniform(keys[0], (n_emb, 4 * n_emb)) * (1 / n_emb**0.5)
        self.l2 = random.uniform(keys[1], (4 * n_emb, n_emb)) * (1 / (4 * n_emb) ** 0.5)
        self.dropout_rate = dropout_rate

    def __call__(self, x, key, is_training):
        x = x @ self.l1
        x = jax.nn.relu(x)
        x = x @ self.l2
        if is_training:
            x *= random.bernoulli(key, 1 - self.dropout_rate, x.shape)  # dropout
            x /= 1 - self.dropout_rate

        return x


class Block(eqx.Module):
    sa: eqx.Module
    ffwd: eqx.Module
    ln1: eqx.Module
    ln2: eqx.Module

    """ Transformer block: communiction followed by computation """

    def __init__(self, jax_key, n_embd, block_size, n_head, dropout_rate):
        # n_embd: embedding dimension, n_head: the number of heads we would like
        keys = random.split(jax_key, 2)
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(
            keys[0], head_size, n_embd, block_size, n_head, dropout_rate
        )
        self.ffwd = FeedForward(keys[1], n_embd, dropout_rate)
        self.ln1 = LayerNorm(n_embd)
        self.ln2 = LayerNorm(n_embd)

    def __call__(self, x, key, is_training):
        key1, key2 = random.split(key, 2)
        x = x + self.sa(self.ln1(x), key1, is_training)
        x = x + self.ffwd(self.ln2(x), key2, is_training)
        return x


class LayerNorm(eqx.Module):
    gamma: jnp.ndarray
    beta: jnp.ndarray
    eps: float = static_field()

    def __init__(self, dim, eps=1e-5):
        self.gamma = jnp.ones(dim)
        self.beta = jnp.zeros(dim)
        self.eps = eps

    def __call__(self, x):
        xmean = jnp.mean(x, axis=-1, keepdims=True)
        xvar = jnp.var(x, axis=-1, keepdims=True)
        inv = self.gamma * jax.lax.rsqrt(xvar + self.eps)
        return inv * (x - xmean) + self.beta


class NanoGPT(eqx.Module):
    tok_embedding: jnp.ndarray
    pos_embedding: jnp.ndarray
    lm_head: jnp.ndarray
    blocks: List[eqx.Module]
    ln_f: eqx.Module

    def __init__(
        self, jax_key, vocab_size, n_emb, block_size, n_head, n_layer, dropout_rate
    ):
        keys = random.split(jax_key, 3 + n_layer)
        self.tok_embedding = random.uniform(keys[0], (vocab_size, n_emb)) * (
            1 / vocab_size**0.5
        )
        self.pos_embedding = random.uniform(keys[1], (block_size, n_emb)) * (
            1 / block_size**0.5
        )
        self.lm_head = random.uniform(keys[2], (n_emb, vocab_size)) * (1 / n_emb**0.5)
        self.blocks = [
            Block(
                jax_key=keys[-index],
                n_embd=n_emb,
                n_head=n_head,
                block_size=block_size,
                dropout_rate=dropout_rate,
            )
            for index in range(n_layer)
        ]
        self.ln_f = LayerNorm(n_emb)

    def __call__(self, idx, key, is_training):
        B, T = idx.shape
        # idx and targets are both (B,T) tensor of integers

        tok_emb = self.tok_embedding[idx]  # (B, T, C)
        pos_emb = self.pos_embedding[jnp.arange(T)]  # (T, C)
        x = pos_emb + tok_emb  # (B, T, C)

        keys = random.split(key, len(self.blocks))
        for block, k in zip(self.blocks, keys):
            x = block(x, k, is_training)
        x = self.ln_f(x)
        logits = x @ self.lm_head

        return logits

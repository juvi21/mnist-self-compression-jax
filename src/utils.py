from typing import Any
import jax.numpy as jnp
import optax
from flax.training import train_state

class TrainState(train_state.TrainState):
    batch_stats: Any

    @classmethod
    def create(cls, *, apply_fn, params, tx, batch_stats, **kwargs):
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            batch_stats=batch_stats,
            **kwargs,
        )

def qbits_fn(params: dict) -> jnp.ndarray:
    def count_bits(layer_params: dict) -> jnp.ndarray:
        weight = layer_params['weight']
        b = jnp.maximum(layer_params['b'], 0.1)
        return jnp.sum(b) * jnp.prod(jnp.array(weight.shape[1:]))
    
    return sum(
        count_bits(layer_params)
        for layer_name, layer_params in params.items()
        if 'QConv2d_' in layer_name
    )

def cross_entropy_loss(logits: jnp.ndarray, labels: jnp.ndarray) -> jnp.ndarray:
    return optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()

def compute_metrics(logits: jnp.ndarray, labels: jnp.ndarray) -> dict:
    loss = cross_entropy_loss(logits, labels)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == labels)
    return {'loss': loss, 'accuracy': accuracy}
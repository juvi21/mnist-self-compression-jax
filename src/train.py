from functools import partial
import jax
import jax.numpy as jnp
from .utils import cross_entropy_loss, compute_metrics, qbits_fn

@partial(jax.jit, static_argnums=(2,))
def train_step(state, batch, weight_count: int):
    def loss_fn(params):
        logits, mut = state.apply_fn({'params': params, 'batch_stats': state.batch_stats}, 
                                     batch['image'], train=True, mutable=['batch_stats'])
        loss = cross_entropy_loss(logits, batch['label'])
        Q = qbits_fn(params) / weight_count
        return loss + 0.05 * Q, (logits, mut)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, mut)), grads = grad_fn(state.params)
    
    new_state = state.apply_gradients(grads=grads, batch_stats=mut['batch_stats'])
    metrics = compute_metrics(logits, batch['label'])
    metrics['Q'] = qbits_fn(new_state.params) / weight_count
    
    return new_state, metrics

@jax.jit
def eval_step(state, batch):
    logits = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats},
                            batch['image'], train=False)
    return compute_metrics(logits, batch['label'])
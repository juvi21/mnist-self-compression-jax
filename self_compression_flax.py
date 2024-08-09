from typing import Sequence, Any
from functools import partial
import time
from jax.tree_util import tree_map 
import jax
import jax.numpy as jnp
from jax.random import PRNGKey
import flax.linen as nn
from flax.training import train_state
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt

from examples import mnist

class CustomTrainState(train_state.TrainState):
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

class QConv2d(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int | Sequence[int]
    
    @nn.compact
    def __call__(self, x):
        k_size = self.kernel_size if isinstance(self.kernel_size, Sequence) else (self.kernel_size, self.kernel_size)
        scale = 1 / jnp.sqrt(self.in_channels * jnp.prod(jnp.array(k_size)))
        weight = self.param('weight', 
                            nn.initializers.uniform(scale),
                            (self.out_channels, self.in_channels, *k_size))
        e = self.param('e', nn.initializers.constant(-8), (self.out_channels,))
        b = self.param('b', nn.initializers.constant(2), (self.out_channels,))

        b_positive = jnp.maximum(b, 0.1)
        qw = jnp.clip(
            2**-e[:, None, None, None] * weight,
            -2**(b_positive[:, None, None, None]-1),
            2**(b_positive[:, None, None, None]-1) - 1
        )
        w = jax.lax.stop_gradient(qw.round() - qw) + qw

        return jax.lax.conv_general_dilated(
            x, 2**e[:, None, None, None] * w, (1, 1), 'SAME',
            dimension_numbers=('NHWC', 'OIHW', 'NHWC')
        )

class Model(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray:
        def conv_block(x: jnp.ndarray, in_channels: int, out_channels: int, kernel_size: int) -> jnp.ndarray:
            x = QConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size)(x)
            x = nn.relu(x)
            return x

        x = conv_block(x, in_channels=1, out_channels=32, kernel_size=5)
        x = conv_block(x, in_channels=32, out_channels=32, kernel_size=5)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = conv_block(x, in_channels=32, out_channels=64, kernel_size=3)
        x = conv_block(x, in_channels=64, out_channels=64, kernel_size=3)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(10)(x)
        return x

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

def main():
    train_images, train_labels, test_images, test_labels = mnist()

    X_train = jnp.asarray(train_images).reshape(-1, 28, 28, 1) / 255.0
    X_test = jnp.asarray(test_images).reshape(-1, 28, 28, 1) / 255.0
    Y_train = jnp.asarray(train_labels.argmax(axis=1))
    Y_test = jnp.asarray(test_labels.argmax(axis=1))

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    model = Model()
    variables = model.init(subkey, jnp.ones((1, 28, 28, 1)))
    weight_count = sum(p.size for p in jax.tree_util.tree_leaves(variables['params']))

    tx = optax.adam(learning_rate=1e-3)
    state = CustomTrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        batch_stats=variables['batch_stats']
    )

    num_epochs = 600
    batch_size = 512
    steps_per_epoch = X_train.shape[0] // batch_size

    all_metrics = []
    for epoch in range(num_epochs):
        start_time = time.time()
        epoch_metrics = []
        
        pbar = tqdm(range(steps_per_epoch), total=steps_per_epoch, 
                    desc=f"Epoch {epoch+1}/{num_epochs}")
        for step in pbar:
            key, subkey = jax.random.split(key)
            idx = jax.random.permutation(subkey, X_train.shape[0])[:batch_size]
            batch = {'image': X_train[idx], 'label': Y_train[idx]}
            state, metrics = train_step(state, batch, weight_count)
            epoch_metrics.append(metrics)
            
            model_bytes = metrics['Q'] * weight_count / 8
            pbar.set_description(f"loss: {metrics['loss']:6.2f}  bytes: {model_bytes:.1f}  acc: {metrics['accuracy']:5.2f}")
        
        avg_metrics = {
            key: sum(m[key] for m in epoch_metrics) / len(epoch_metrics)
            for key in epoch_metrics[0].keys()
        }
        
        print(f"\nEpoch {epoch+1}: loss: {avg_metrics['loss']:.4f}, "
              f"accuracy: {avg_metrics['accuracy']:.4f}, "
              f"Q: {avg_metrics['Q']:.4f}")
        print(f"Epoch time: {time.time() - start_time:.2f} seconds")

        test_metrics = []
        for i in range(0, X_test.shape[0], batch_size):
            batch = {'image': X_test[i:i+batch_size], 'label': Y_test[i:i+batch_size]}
            metrics = eval_step(state, batch)
            test_metrics.append(metrics)
        
        avg_test_metrics = {
            key: sum(m[key] for m in test_metrics) / len(test_metrics)
            for key in test_metrics[0].keys()
        }
        print(f"Test accuracy: {avg_test_metrics['accuracy']:.4f}")
        
        all_metrics.extend(epoch_metrics)

    print(f"\nFinal model size: {qbits_fn(state.params) / 8:.1f} bytes")

    # Debugging: Print out some information about all_metrics
    print(f"Total number of metric entries: {len(all_metrics)}")
    print(f"Type of first entry in all_metrics: {type(all_metrics[0])}")
    print(f"Keys in first entry of all_metrics: {all_metrics[0].keys() if isinstance(all_metrics[0], dict) else 'Not a dict'}")

    plt.figure(figsize=(12, 6))
    try:
        q_values = [m['Q'] * weight_count / 8 for m in all_metrics]
        plt.plot(q_values, color="red", label="Model Size (bytes)")
    except TypeError as e:
        print(f"Error plotting Q values: {e}")
        print(f"First few Q values: {all_metrics[:5]}")
    
    try:
        accuracy_values = [m['accuracy'] * 100 for m in all_metrics]
        plt.twinx()
        plt.plot(accuracy_values, color="blue", label="Training Accuracy (%)")
    except TypeError as e:
        print(f"Error plotting accuracy values: {e}")
        print(f"First few accuracy values: {all_metrics[:5]}")

    plt.title("Model Size vs Training Accuracy Over Training")
    plt.xlabel("Iteration")
    plt.ylabel("Metric Value")
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

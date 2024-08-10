import time
import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm
from src.datasets import mnist

from src.layers import Model
from src.utils import TrainState, qbits_fn
from src.train import train_step, eval_step
from src.plot import plot_training_results

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
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        batch_stats=variables['batch_stats']
    )

    num_epochs = 500
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

    plot_training_results(all_metrics, weight_count)

if __name__ == "__main__":
    main()
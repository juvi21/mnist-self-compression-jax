# This was the very first version. It trains but I guess I have it more 'jax' in the current version.
# Maybe this is useful for someone ... so I keep it for now.
import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from tqdm import trange
from jax.tree_util import tree_leaves
import matplotlib.pyplot as plt

from examples import mnist

class QConv2d(nn.Module):
    in_channels: int
    out_channels: int
    kernel_size: int | tuple[int, int]
    
    def setup(self):
        k_size = self.kernel_size if isinstance(self.kernel_size, tuple) else (self.kernel_size, self.kernel_size)
        scale = 1 / jnp.sqrt(self.in_channels * jnp.prod(jnp.array(k_size)))
        self.weight = self.param('weight', 
                                 nn.initializers.uniform(scale),
                                 (self.out_channels, self.in_channels, *k_size))
        self.e = self.param('e', nn.initializers.constant(-8), (self.out_channels,))
        self.b = self.param('b', nn.initializers.constant(2), (self.out_channels,))

    def qweight(self):
        b_positive = jnp.maximum(self.b, 0.1)
        qw = jnp.minimum(
            jnp.maximum(2**-self.e[:, None, None, None] * self.weight, -2**(b_positive[:, None, None, None]-1)),
            2**(b_positive[:, None, None, None]-1) - 1
        )
        return qw

    @nn.compact
    def __call__(self, x):
        qw = self.qweight()
        w = jax.lax.stop_gradient(qw.round() - qw) + qw
        return jax.lax.conv_general_dilated(
            x, 2**self.e[:, None, None, None] * w, (1, 1), 'SAME',
            dimension_numbers=('NHWC', 'OIHW', 'NHWC')
        )

class Model(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool = False):
        x = QConv2d(1, 32, 5)(x)
        x = nn.relu(x)
        x = QConv2d(32, 32, 5)(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = QConv2d(32, 64, 3)(x)
        x = nn.relu(x)
        x = QConv2d(64, 64, 3)(x)
        x = nn.relu(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(10)(x)
        return x

def qbits_fn(params):
    total_bits = 0
    for layer_name, layer_params in params.items():
        if 'QConv2d_' in layer_name:
            weight = layer_params['weight']
            b = layer_params['b']
            b_positive = jnp.maximum(b, 0.1)
            bits = jnp.sum(b_positive) * jnp.prod(jnp.array(weight.shape[1:]))
            total_bits += bits
    return total_bits

train_images, train_labels, test_images, test_labels = mnist()

# mean, std = 0.1307, 0.3081
# Standart way of norm in MNIST playing around with mean and std.
X_train = jnp.array(train_images).reshape(-1, 28, 28, 1) / 255.0
X_test = jnp.array(test_images).reshape(-1, 28, 28, 1) / 255.0
#X_train = (jnp.array(train_images).reshape(-1, 28, 28, 1) - mean) / std
Y_train = jnp.array(train_labels.argmax(axis=1))
#X_test = (jnp.array(test_images).reshape(-1, 28, 28, 1) - mean) / std
Y_test = jnp.array(test_labels.argmax(axis=1))

@jax.jit
def loss_fn(params, batch_stats, batch):
    inputs, labels = batch
    variables = {'params': params, 'batch_stats': batch_stats}
    logits, updated_vars = model.apply(variables, inputs, mutable=['batch_stats'], train=True)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    Q = qbits_fn(params) / weight_count
    return loss + 0.05 * Q, (updated_vars['batch_stats'], loss, Q)

@jax.jit
def update_step(opt_state, params, batch_stats, batch):
    (loss, (updated_batch_stats, loss_value, Q)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch_stats, batch)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return opt_state, params, updated_batch_stats, loss_value, Q

@jax.jit
def compute_accuracy(params, batch_stats, inputs, targets):
    variables = {'params': params, 'batch_stats': batch_stats}
    logits = model.apply(variables, inputs, train=False)
    return jnp.mean(jnp.argmax(logits, axis=-1) == targets)

key = jax.random.PRNGKey(0)
model = Model()
variables = model.init(key, jnp.ones((1, 28, 28, 1)))
params, batch_stats = variables['params'], variables['batch_stats']

weight_count = sum(p.size for p in tree_leaves(params))

optimizer = optax.adam(learning_rate=1e-3)
opt_state = optimizer.init(params)

losses, test_accs, bytes_used = [], [], []
test_acc = 0.0

for i in (t := trange(100000)):
    key, subkey = jax.random.split(jax.random.PRNGKey(i))
    idx = jax.random.randint(subkey, (512,), 0, X_train.shape[0])
    batch = (X_train[idx], Y_train[idx])
    
    opt_state, params, batch_stats, loss, Q = update_step(opt_state, params, batch_stats, batch)
    
    model_bytes = Q * weight_count / 8
    
    if i % 10 == 9:
        test_acc = compute_accuracy(params, batch_stats, X_test, Y_test).item()
    
    losses.append(loss)
    test_accs.append(test_acc * 100)

    bytes_used.append(model_bytes)
    
    t.set_description(f"loss: {loss:6.2f}  bytes: {model_bytes:.1f}  acc: {test_acc:5.2f}%")

print(f"Final test accuracy: {test_accs[-1]:.2f}%")
print(f"Final model size: {bytes_used[-1]:.1f} bytes")

fig, ax1 = plt.subplots(figsize=(12,6))

ax1.set_xlabel("Iteration")
ax1.set_ylabel("Model Size (bytes)")
ax1.yaxis.label.set_color("red")
ax1.plot(bytes_used, color="red")

ax2 = ax1.twinx()
ax2.plot(test_accs, color="blue")
ax2.set_ylim(0, 100)
ax2.set_ylabel("Test Accuracy (%)")
ax2.yaxis.label.set_color("blue")

plt.title("Model Size vs Test Accuracy Over Training")
plt.tight_layout()
plt.show()

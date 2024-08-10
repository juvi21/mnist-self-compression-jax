from typing import Sequence
import jax
import jax.numpy as jnp
import flax.linen as nn

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
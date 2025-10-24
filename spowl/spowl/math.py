import jax
import jax.numpy as jnp


def soft_ce(pred, target, num_bins, vmin, vmax):
    """Computes the cross entropy loss between predictions and soft targets."""
    pred = jax.nn.log_softmax(pred, axis=-1)
    target = two_hot(target, num_bins, vmin, vmax)
    return -(target * pred).sum(-1, keepdims=True)


def two_hot(x, num_bins, vmin, vmax):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    bin_size = (vmax - vmin) / (num_bins-1)
    if num_bins == 0:
        return x
    elif num_bins == 1:
        return symlog(x)
    x = jnp.squeeze(jnp.clip(symlog(x), vmin, vmax), 1)
    bin_idx = jnp.floor((x - vmin) / bin_size).astype(jnp.int32)
    bin_offset = (x - vmin) / bin_size - bin_idx
    soft_two_hot = jnp.zeros((x.shape[0], num_bins))
    soft_two_hot = soft_two_hot.at[jnp.arange(x.shape[0]), bin_idx].set(1 - bin_offset)
    soft_two_hot = soft_two_hot.at[jnp.arange(x.shape[0]), (bin_idx + 1) % num_bins].set(bin_offset)
    return soft_two_hot


def log_std(x, low, dif):
    return low + 0.5 * dif * (jnp.tanh(x) + 1)


def _gaussian_residual(eps, log_std):
    return -0.5 * jnp.power(eps, 2) - log_std


def _gaussian_logprob(residual):
    return residual - 0.5 * jnp.log(2 * jnp.pi)


def gaussian_logprob(eps, log_std, size=None):
    """Compute Gaussian log probability."""
    residual = _gaussian_residual(eps, log_std).sum(-1, keepdims=True)
    if size is None:
        size = eps.shape[-1]
    return _gaussian_logprob(residual) * size


def _squash(pi):
    return jnp.log(jax.nn.relu(1 - jnp.pow(pi, 2)) + 1e-6)


def squash(mu, pi, log_pi):
    """Apply squashing function."""
    mu = jnp.tanh(mu)
    pi = jnp.tanh(pi)
    log_pi -= _squash(pi).sum(-1, keepdims=True)
    return mu, pi, log_pi


def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def two_hot_inv(x, num_bins, vmin, vmax):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    if num_bins == 0:
        return x
    elif num_bins == 1:
        return symexp(x)
    DREG_BINS = jnp.linspace(vmin, vmax, num_bins)
    x = jax.nn.softmax(x, axis=-1)
    x = jnp.sum(x * DREG_BINS, axis=-1, keepdims=True)
    return symexp(x)


def bootstrap_mean_ci(key, sample, n_samples, ci):
    bootstraped = jax.random.choice(key, sample, (n_samples, len(sample)))
    bootstrapped = jnp.mean(bootstraped, 1)
    low = jnp.percentile(bootstrapped, (100-ci) / 2)
    high = jnp.percentile(bootstrapped, (100+ci) / 2)
    return low, high
    
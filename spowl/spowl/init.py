import jax
import equinox as eqx


def trunc_init(weight: jax.Array, key: jax.random.PRNGKey) -> jax.Array:
    return 0.02 * jax.random.truncated_normal(key, shape=weight.shape, lower=-2, upper=2)


def init_linear(model, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [
        x.weight for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)
    ]
    get_biases = lambda m: [
        x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)
    ]
    weights = get_weights(model)
    biases = get_biases(model)

    new_weights = [
        trunc_init(weight, subkey) for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    new_biases = [jax.numpy.zeros_like(bias) for bias in biases]
    new_model = eqx.tree_at(get_biases, new_model, new_biases)
    return new_model

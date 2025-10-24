import jax
import equinox as eqx
import equinox.nn as enn


@eqx.filter_vmap(in_axes=(0, None, None, None, None, None))
def ens(key, in_dim, mlp_dims, out_dim, act, dropout):
    return mlp(key, in_dim, mlp_dims, out_dim, act, dropout)


class SimNorm(eqx.Module):
    """
    Simplicial normalization.
    Adapted from https://arxiv.org/abs/2204.00616.
    """
    dim: int
    
    def __init__(self, dim):
        self.dim = dim
    
    def __call__(self, x, key=None):
        shp = x.shape
        new_sh = (*shp[:-1], -1, self.dim)
        x = x.reshape(new_sh)
        x = jax.nn.softmax(x, axis=-1)
        return x.reshape(shp)


class NormedLinear(eqx.Module):
    """
    Linear layer with LayerNorm, activation, and optionally dropout.
    """
    linear: enn.Linear
    ln: enn.LayerNorm
    dropout: enn.Dropout
    in_features: int
    out_features: int
    act: eqx.Module

    def __init__(self, in_dim, out_dim, act=enn.Lambda(jax.nn.mish), dropout=0., key=None):
        self.linear = enn.Linear(in_dim, out_dim, key=key)
        self.ln = enn.LayerNorm(out_dim)
        self.act = act
        self.dropout = enn.Dropout(dropout) if dropout else None
        self.in_features = in_dim
        self.out_features = out_dim

    def __call__(self, x, key):
        x = self.linear(x)
        if self.dropout:
            x = self.dropout(x, key=key)
        x = self.ln(x)
        return self.act(x)


def mlp(key, in_dim, mlp_dims, out_dim, act=None, dropout=0.):
    """
    Basic building block of SPOWL.
    MLP with LayerNorm, Mish activations, and optionally dropout.
    """
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    mlp = []
    for i in range(len(dims) - 2):
        key, subkey = jax.random.split(key)
        mlp.append(NormedLinear(dims[i], dims[i+1], dropout=dropout*(i==0), key=subkey))
    mlp.append(NormedLinear(dims[-2], dims[-1], act=act, key=key) if act else enn.Linear(dims[-2], dims[-1], key=key))
    return enn.Sequential(mlp)


def enc(obs_shape, num_enc_layers, enc_dim, latent_dim, simnorm_dim, key, out={}):
    out = mlp(key, obs_shape, max(num_enc_layers-1, 1)*[enc_dim], latent_dim, act=SimNorm(simnorm_dim))
    return out

import jax.numpy as jnp

def mellow_transform(x, n_actions, use_bias: bool, use_scale: bool):
    logn_actions = jnp.where(n_actions == 1, 1.0, jnp.log(n_actions)) # do not transform the inputs with no actions 
     
    if use_scale:
        # renormalization of the entropy
        x = x / logn_actions
        logn_actions = 1.0 # entropies have been rescaled and so the debiasing constant
    
    if use_bias:
        # mellow shift of entropy
        x = x - logn_actions
    return x
import logging
import jax
import jax.numpy as jnp
from jax.experimental.stax import Dense, serial, Tanh
from deq import Deq, Module, forward


def test_forward_solver():
    ndim = 10
    W = jax.random.normal(jax.random.PRNGKey(0), (ndim, ndim)) / jnp.sqrt(ndim)
    x = jax.random.normal(jax.random.PRNGKey(1), (ndim,))
    f = lambda W, x, z: jnp.tanh(jnp.dot(W, z) + x)

    z_star = forward(lambda z: f(W, x, z), z_init=jnp.zeros_like(x))
    logging.debug(z_star)
    return


def test_deq_fwd():
    ndim = 10
    x = jax.random.normal(jax.random.PRNGKey(1), (ndim,))

    solver = forward
    f = Module(serial(Dense(ndim), Tanh))

    d = Deq(solver, f)
    out_shape, params = d.init(jax.random.PRNGKey(2), x.shape)
    print(out_shape, params)

    d.apply(params, x)
    y = jax.jit(d.apply)(params, x)
    print(y.shape)

    jax.grad(lambda p: sum(d.apply(params, x)))(1.0)
    return


test_deq_fwd()
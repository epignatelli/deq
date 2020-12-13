from typing import Callable, Tuple, Any, NamedTuple
import functools
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import stax
from jax.nn.initializers import normal


Params = Any
RNGKey = jnp.ndarray
Shape = Tuple[int, ...]
Function = Callable[[jnp.ndarray], jnp.ndarray]
Solver = Callable[[Function, jnp.ndarray], jnp.ndarray]


class Module(
    NamedTuple(
        "Module",
        [
            ("init", Callable[[RNGKey, Shape], Tuple[Shape, Params]]),
            ("apply", Callable[[Params, jnp.ndarray], jnp.ndarray]),
        ],
    )
):
    def __new__(self, init_apply_tuple):
        print(init_apply_tuple)
        return super().__new__(self, *init_apply_tuple)


def module(module_maker):
    @functools.wraps(module_maker)
    def fabricate_module(*args, **kwargs):
        init_apply = module_maker(*args, **kwargs)
        return Module(init_apply)

    return fabricate_module


@module
def Deq(solver: Solver, module: Module) -> Module:
    def init_fn(rng: jax.random.PRNGKey, in_shape: Shape) -> Tuple[Shape, Params]:
        return module.init(rng, in_shape)

    def f(params, x):
        return module.apply(params, x)

    @jax.custom_vjp
    def apply_fn(params, x):
        z_star = solver(lambda z: f(params, z), z_init=jnp.zeros_like(x))
        return z_star

    def apply_fn_fwd(params, x):
        z_star = apply_fn(params, x)
        return z_star, (params, x, z_star)

    def apply_fn_bwd(res: Tuple, z_star_bar: jnp.ndarray) -> jnp.ndarray:
        params, x, z_star = res
        _, vjp_a = jax.vjp(lambda params: f(params, z_star), params, x)
        _, vjp_z = jax.vjp(lambda z: f(params, z), z_star)
        return vjp_a(
            solver(lambda u: vjp_z(u)[0] + z_star_bar, z_init=jnp.zeros_like(x))
        )

    apply_fn.defvjp(apply_fn_fwd, apply_fn_bwd)
    return init_fn, apply_fn
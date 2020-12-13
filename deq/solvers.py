from typing import Callable

import jax
import jax.numpy as jnp


def forward(f: Callable, z_init: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    def cond(state):
        z, fz = state
        return jnp.linalg.norm(z - fz) > eps

    def update(state):
        z, fz = state
        return fz, f(fz)

    state = (z_init, f(z_init))
    z, fz = jax.lax.while_loop(cond, update, state)
    return fz


def newton(f: Callable, z_init: jnp.ndarray) -> jnp.ndarray:
    f_root = lambda z: f(z) - z
    g = lambda z: z - jnp.linalg.solve(jax.jacobian(f_root)(z), f_root(z))
    return forward(g, z_init)


def anderson(
    f: Callable,
    z_init: jnp.ndarray,
    m: int = 5,
    lam: float = 1e-4,
    max_iter: int = 50,
    tol: float = 1e-5,
    beta: float = 1.0,
) -> jnp.ndarray:
    x0 = z_init
    x1 = f(x0)
    x2 = f(x1)
    X = jnp.concatenate([jnp.stack([x0, x1]), jnp.zeros((m - 2, *jnp.shape(x0)))])
    F = jnp.concatenate([jnp.stack([x1, x2]), jnp.zeros((m - 2, *jnp.shape(x0)))])

    def step(n, k, X, F):
        G = F[:n] - X[:n]
        GTG = jnp.tensordot(G, G, [list(range(1, G.ndim))] * 2)
        H = jnp.block(
            [[jnp.zeros((1, 1)), jnp.ones((1, n))], [jnp.ones((n, 1)), GTG]]
        ) + lam * jnp.eye(n + 1)
        alpha = jnp.linalg.solve(H, jnp.zeros(n + 1).at[0].set(1))[1:]

        xk = beta * jnp.dot(alpha, F[:n]) + (1 - beta) * jnp.dot(alpha, X[:n])
        X = X.at[k % m].set(xk)
        F = F.at[k % m].set(f(xk))
        return X, F

    # unroll the first m steps
    for k in range(2, m):
        X, F = step(k, k, X, F)
        res = jnp.linalg.norm(F[k] - X[k]) / (1e-5 + jnp.linalg.norm(F[k]))
        if res < tol or k + 1 >= max_iter:
            return X[k], k

    # run the remaining steps in a lax.while_loop
    def body_fun(carry):
        k, X, F = carry
        X, F = step(m, k, X, F)
        return k + 1, X, F

    def cond_fun(carry):
        k, X, F = carry
        kmod = (k - 1) % m
        res = jnp.linalg.norm(F[kmod] - X[kmod]) / (1e-5 + jnp.linalg.norm(F[kmod]))
        return (k < max_iter) & (res >= tol)

    k, X, F = jax.lax.while_loop(cond_fun, body_fun, (k + 1, X, F))
    return X[(k - 1) % m], k


def broyden(f: Callable, z_init: jnp.ndarray) -> jnp.ndarray:
    def sherman_morrison(m, u, v):
        return jnp.matmul(v.T, m)


def gauss_seidel(f: Callable, z_init: jnp.ndarray) -> jnp.ndarray:
    pass


def jacobi(f: Callable, z_init: jnp.ndarray) -> jnp.ndarray:
    pass

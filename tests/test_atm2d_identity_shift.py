"""K42: a BE step must not silently discard its time-derivative coefficient."""
import pytest
import torch

import kintera as kt
from kintera.atm2d.newton.coupled import newton_implicit_step


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda":
        if not torch.cuda.is_available():
            pytest.skip("CUDA unavailable")
        from kintera import kintera as native
        if not hasattr(native, "cuda_csr_solve_cusolver"):
            pytest.skip("CUDA sparse binding unavailable")
    return request.param


def closed_column(dtype, device):
    kw = dict(dtype=dtype, device=device)
    return kt.AtmState2D(
        x1f=torch.arange(4, **kw), x2f=torch.tensor([0., 1.], **kw),
        temperature=torch.full((1, 3), 250., **kw),
        pressure=torch.full((1, 3), 1e4, **kw),
        concentration=torch.ones((1, 3, 1), **kw),
    )


@pytest.mark.parametrize("dtype,dt", [(torch.float64, 1e18), (torch.float32, 1e8)])
def test_newton_rejects_lost_identity_shift(device, dtype, dt):
    """Previously CUDA converged with 98.44%/70.94% mass loss; CPU LU raised."""
    state = closed_column(dtype, device)
    before = state.concentration.clone()
    with pytest.raises(RuntimeError, match="identity shift"):
        newton_implicit_step(
            state, dt, kzz=torch.ones((1, 3), dtype=dtype, device=device),
            source_terms=None, transport_form="c_diffusion", max_iterations=4,
        )
    torch.testing.assert_close(state.concentration, before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("dt", [0.1, 1.0, 10.0])
def test_ordinary_newton_preserves_closed_constant_state(device, dtype, dt):
    state = closed_column(dtype, device)
    before = state.concentration.clone()
    result = newton_implicit_step(
        state, dt, kzz=torch.ones((1, 3), dtype=dtype, device=device),
        source_terms=None, transport_form="c_diffusion", max_iterations=4,
    )
    assert result.converged
    assert torch.isfinite(result.concentration).all()
    # Forward error against the exact constant-state solution, not a solve residual.
    tol = 2e-5 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(result.concentration, before, rtol=tol, atol=0)
    torch.testing.assert_close(result.concentration.sum(), before.sum(), rtol=tol, atol=0)
    torch.testing.assert_close(state.concentration, before, rtol=0, atol=0)


@pytest.mark.parametrize("dtype,kzz", [(torch.float64, 1e20), (torch.float32, 1e10)])
def test_natural_form_also_rejects_lost_identity_shift(device, dtype, kzz):
    # dt < 1 takes the I-dt*L branch; large transport can still erase the I.
    state = closed_column(dtype, device)
    with pytest.raises(RuntimeError, match="identity shift"):
        kt.build_implicit_step_system(
            state, torch.full((1, 3), kzz, dtype=dtype, device=device), 0.1,
            transport_form="c_diffusion",
        )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_small_representable_shift_without_transport_is_allowed(device, dtype):
    # Small shift alone is not grounds to reject a perfectly conditioned system.
    state = closed_column(dtype, device)
    matrix, rhs = kt.build_implicit_step_system(
        state, torch.zeros((1, 3), dtype=dtype, device=device), 1e30,
        transport_form="c_diffusion",
    )
    result = kt.solve_sparse_system(matrix, rhs)
    torch.testing.assert_close(result, state.concentration)


def test_underflowed_shift_without_stored_diagonal_is_rejected(device):
    state = closed_column(torch.float32, device)
    with pytest.raises(RuntimeError, match="identity shift"):
        kt.build_implicit_step_system(
            state, torch.zeros((1, 3), dtype=torch.float32, device=device), 1e100,
            transport_form="c_diffusion",
        )

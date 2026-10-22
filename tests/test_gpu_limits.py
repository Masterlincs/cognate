import torch

from mtgemb.cli import gpu_limits


def test_no_env_no_limits():
    assert gpu_limits(env={}) == {"mem_gb": None, "rest": 0.0, "hook": None}


def test_rest_sleeps_in_proportion_to_the_update():
    now, slept = [100.0], []
    lim = gpu_limits(env={"MTGEMB_GPU_REST": "0.5"}, clock=lambda: now[0], sleep=slept.append)
    try:
        w = torch.nn.Parameter(torch.ones(2))
        opt = torch.optim.SGD([w], lr=0.1)
        for took in (4.0, 200.0):  # an ordinary update, then one with an evaluation in it
            now[0] += took
            w.grad = torch.ones(2)
            opt.step()
    finally:
        lim["hook"].remove()
    assert slept == [2.0, 60.0]  # half the update time, capped at a minute
    assert torch.allclose(w.detach(), torch.full((2,), 0.8))  # the step itself is unchanged

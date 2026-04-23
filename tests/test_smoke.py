
from examples.generate_synthetic_neck import make_necked_tube
from spectral_membranes.pipeline import run_dual_operator_pipeline

def test_smoke():
    mesh = make_necked_tube(n_theta=16, n_z=24)
    results = run_dual_operator_pipeline(mesh, k=8)
    graph = results["graph"]
    assert graph.lambda2 is not None
    assert graph.lambda2 > 0

import torch
from pytorch3d import transforms



from scipy.stats import binomtest
def binomial_test(success, n, confidence = 0.95):
    result = binomtest(k=success, n=max(n,1))
    mid = result.proportion_estimate
    low = result.proportion_ci(confidence_level=confidence, method='exact').low
    high = result.proportion_ci(confidence_level=confidence, method='exact').high

    result_str = f"{100*success/max(n,1):.1f}% ({success} / {n});   ({100*confidence:.0f}% CI: {low*100:.1f}%~{high*100:.1f}%)"

    return mid, low, high, result_str


@torch.jit.script
def normalize_quaternion(q):
    return transforms.standardize_quaternion(q/torch.norm(q, dim=-1, keepdim=True))

def check_irreps_sorted(irreps):
    max_deg = 0
    for irrep in irreps:
        deg = int(irrep.ir[0])
        if deg < max_deg:
            return False
        else:
            max_deg = deg
    return True

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio



def calc_mse(gt, pred):
    """Compute Mean Squared Error (MSE)"""
    return np.mean((gt - pred) ** 2)


def calc_nmse(gt, pred):
    """Compute Normalized Mean Squared Error (NMSE)"""
    return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2


def calc_psnr(gt, pred, maxval=None):
    """Compute Peak Signal to Noise Ratio metric (PSNR)"""
    if maxval is None:
        maxval = gt.max() - gt.min()
    return peak_signal_noise_ratio(gt, pred, data_range=maxval)


def calc_ssim(gt, pred, maxval=None):
    """Compute Structural Similarity Index Metric (SSIM)"""
    if not gt.ndim == pred.ndim:
        raise ValueError("Ground truth dimensions does not match pred.")

    maxval = gt.max() - gt.min() if maxval is None else maxval
    ssim = structural_similarity(gt, pred, data_range=maxval)

    return ssim
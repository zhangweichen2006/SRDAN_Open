import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils

class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights


class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)


################################################################################
# Helper functions to compute variances based on kernel matrices
################################################################################


class RBFMMDLoss(nn.Module):
    def __init__(self, sigma_list: list = [0.01, 0.1, 1, 10, 100]):
        """
        Args:
            sigma_list: sigma_list.
        """
        super(RBFMMDLoss, self).__init__()
        self.sigma_list = sigma_list

    def forward(self, X, Y, biased=True):
        K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
        # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
        return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)
        

min_var_est = 1e-8


# Consider linear time MMD with a linear kernel:
# K(f(x), f(y)) = f(x)^Tf(y)
# h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
#             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
#
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def linear_mmd2(f_of_X, f_of_Y):
    loss = 0.0
    delta = f_of_X - f_of_Y
    #pdb.set_trace()
    loss = torch.mean((delta[:-1] * delta[1:]).sum(1))
    return loss


# Consider linear time MMD with a polynomial kernel:
# K(f(x), f(y)) = (alpha*f(x)^Tf(y) + c)^d
# f_of_X: batch_size * k
# f_of_Y: batch_size * k
def poly_mmd2(f_of_X, f_of_Y, d=2, alpha=1.0, c=2.0):
    K_XX = (alpha * (f_of_X[:-1] * f_of_X[1:]).sum(1) + c)
    K_XX_mean = torch.mean(K_XX.pow(d))

    K_YY = (alpha * (f_of_Y[:-1] * f_of_Y[1:]).sum(1) + c)
    K_YY_mean = torch.mean(K_YY.pow(d))

    K_XY = (alpha * (f_of_X[:-1] * f_of_Y[1:]).sum(1) + c)
    K_XY_mean = torch.mean(K_XY.pow(d))

    K_YX = (alpha * (f_of_Y[:-1] * f_of_X[1:]).sum(1) + c)
    K_YX_mean = torch.mean(K_YX.pow(d))

    return K_XX_mean + K_YY_mean - K_XY_mean - K_YX_mean


def _mix_rbf_kernel(X, Y, sigma_list):
    assert(X.size(0) == Y.size(0))
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)


def mix_rbf_mmd2(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)


def mix_rbf_mmd2_and_ratio(X, Y, sigma_list, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigma_list)
    # return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)
    return _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=biased)



def _mmd2(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    return mmd2


def _mmd2_and_ratio(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    mmd2, var_est = _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=const_diagonal, biased=biased)
    loss = mmd2 / torch.sqrt(torch.clamp(var_est, min=min_var_est))
    return loss, mmd2, var_est


def _mmd2_and_variance(K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    m = K_XX.size(0)    # assume X, Y are same shape

    # Get the various sums of kernels that we'll use
    # Kts drop the diagonal, but we don't need to compute them explicitly
    if const_diagonal is not False:
        diag_X = diag_Y = const_diagonal
        sum_diag_X = sum_diag_Y = m * const_diagonal
        sum_diag2_X = sum_diag2_Y = m * const_diagonal**2
    else:
        diag_X = torch.diag(K_XX)                       # (m,)
        diag_Y = torch.diag(K_YY)                       # (m,)
        sum_diag_X = torch.sum(diag_X)
        sum_diag_Y = torch.sum(diag_Y)
        sum_diag2_X = diag_X.dot(diag_X)
        sum_diag2_Y = diag_Y.dot(diag_Y)

    Kt_XX_sums = K_XX.sum(dim=1) - diag_X             # \tilde{K}_XX * e = K_XX * e - diag_X
    Kt_YY_sums = K_YY.sum(dim=1) - diag_Y             # \tilde{K}_YY * e = K_YY * e - diag_Y
    K_XY_sums_0 = K_XY.sum(dim=0)                     # K_{XY}^T * e
    K_XY_sums_1 = K_XY.sum(dim=1)                     # K_{XY} * e

    Kt_XX_sum = Kt_XX_sums.sum()                       # e^T * \tilde{K}_XX * e
    Kt_YY_sum = Kt_YY_sums.sum()                       # e^T * \tilde{K}_YY * e
    K_XY_sum = K_XY_sums_0.sum()                       # e^T * K_{XY} * e

    Kt_XX_2_sum = (K_XX ** 2).sum() - sum_diag2_X      # \| \tilde{K}_XX \|_F^2
    Kt_YY_2_sum = (K_YY ** 2).sum() - sum_diag2_Y      # \| \tilde{K}_YY \|_F^2
    K_XY_2_sum  = (K_XY ** 2).sum()                    # \| K_{XY} \|_F^2

    if biased:
        mmd2 = ((Kt_XX_sum + sum_diag_X) / (m * m)
            + (Kt_YY_sum + sum_diag_Y) / (m * m)
            - 2.0 * K_XY_sum / (m * m))
    else:
        mmd2 = (Kt_XX_sum / (m * (m - 1))
            + Kt_YY_sum / (m * (m - 1))
            - 2.0 * K_XY_sum / (m * m))

    var_est = (
        2.0 / (m**2 * (m - 1.0)**2) * (2 * Kt_XX_sums.dot(Kt_XX_sums) - Kt_XX_2_sum + 2 * Kt_YY_sums.dot(Kt_YY_sums) - Kt_YY_2_sum)
        - (4.0*m - 6.0) / (m**3 * (m - 1.0)**3) * (Kt_XX_sum**2 + Kt_YY_sum**2)
        + 4.0*(m - 2.0) / (m**3 * (m - 1.0)**2) * (K_XY_sums_1.dot(K_XY_sums_1) + K_XY_sums_0.dot(K_XY_sums_0))
        - 4.0*(m - 3.0) / (m**3 * (m - 1.0)**2) * (K_XY_2_sum) - (8 * m - 12) / (m**5 * (m - 1)) * K_XY_sum**2
        + 8.0 / (m**3 * (m - 1.0)) * (
            1.0 / m * (Kt_XX_sum + Kt_YY_sum) * K_XY_sum
            - Kt_XX_sums.dot(K_XY_sums_1)
            - Kt_YY_sums.dot(K_XY_sums_0))
        )
    return mmd2, var_est

# def pairwise_distance(x, y):

#     if not len(x.shape) == len(y.shape) == 2:
#         raise ValueError('Both inputs should be matrices.')

#     if x.shape[1] != y.shape[1]:
#         raise ValueError('The number of features should be the same.')

#     x = x.view(x.shape[0], x.shape[1], 1)
#     y = torch.transpose(y, 0, 1)
#     output = torch.sum((x - y) ** 2, 1)
#     output = torch.transpose(output, 0, 1)

#     return output

# def gaussian_kernel_matrix(x, y, sigmas):

#     sigmas = sigmas.view(sigmas.shape[0], 1)
#     beta = 1. / (2. * sigmas)
#     dist = pairwise_distance(x, y).contiguous()
#     dist_ = dist.view(1, -1)
#     s = torch.matmul(beta, dist_)

#     return torch.sum(torch.exp(-s), 0).view_as(dist)

# def maximum_mean_discrepancy(x, y, kernel= gaussian_kernel_matrix):

#     cost = torch.mean(kernel(x, x))
#     cost += torch.mean(kernel(y, y))
#     cost -= 2 * torch.mean(kernel(x, y))

#     return cost

# def mmd_loss(source_features, target_features):

#     sigmas = [
#         1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
#         1e3, 1e4, 1e5, 1e6
#     ]
#     gaussian_kernel = partial(
#         gaussian_kernel_matrix, sigmas = torch.cuda.FloatTensor(sigmas)
#     )
#     loss_value = maximum_mean_discrepancy(source_features, target_features, kernel= gaussian_kernel)
#     loss_value = loss_value

#     return loss_value


# class MultipleKernelMaximumMeanDiscrepancy(nn.Module):
#     """
#     Parameters:
#         - **kernels** (tuple(`nn.Module`)): kernel functions.
#         - **linear** (bool): whether use the linear version of DAN. Default: False
#         - **quadratic_program** (bool): whether use quadratic program to solve :math:`\beta`. Default: False
#     Inputs: z_s, z_t
#         - **z_s** (tensor): activations from the source domain, :math:`z^s`
#         - **z_t** (tensor): activations from the target domain, :math:`z^t`
#     Shape:
#         - Inputs: :math:`(minibatch, *)`  where * means any dimension
#         - Outputs: scalar
#     .. note::
#         Activations :math:`z^{s}` and :math:`z^{t}` must have the same shape.
#     .. note::
#         The kernel values will add up when there are multiple kernels.
#     Examples::
#         - from dalib.modules.kernels import GaussianKernel
#         - feature_dim = 1024
#         - batch_size = 10
#         - kernels = (GaussianKernel(alpha=0.5), GaussianKernel(alpha=1.), GaussianKernel(alpha=2.))
#         - loss = MultipleKernelMaximumMeanDiscrepancy(kernels)
#         - # features from source domain and target domain
#         - z_s, z_t = torch.randn(batch_size, feature_dim), torch.randn(batch_size, feature_dim)
#         - output = loss(z_s, z_t)
#     """

#     def __init__(self, kernels: Sequence[nn.Module], linear: Optional[bool] = False,
#                  quadratic_program: Optional[bool] = False):
#         super(MultipleKernelMaximumMeanDiscrepancy, self).__init__()
#         self.kernels = kernels
#         self.index_matrix = None
#         self.linear = linear
#         self.quadratic_program = quadratic_program

#     def forward(self, z_s: torch.Tensor, z_t: torch.Tensor) -> torch.Tensor:
#         features = torch.cat([z_s, z_t], dim=0)
#         batch_size = int(z_s.size(0))
#         self.index_matrix = _update_index_matrix(batch_size, self.index_matrix, self.linear).to(z_s.device)

#         if not self.quadratic_program:
#             kernel_matrix = sum([kernel(features) for kernel in self.kernels])  # Add up the matrix of each kernel
#             # Add 2 / (n-1) to make up for the value on the diagonal
#             # to ensure loss is positive in the non-linear version
#             loss = (kernel_matrix * self.index_matrix).sum() + 2. / float(batch_size - 1)
#         else:
#             kernel_values = [(kernel(features) * self.index_matrix).sum() + 2. / float(batch_size - 1) for kernel in self.kernels]
#             loss = optimal_kernel_combinations(kernel_values)
#         return loss


# def _update_index_matrix(batch_size: int, index_matrix: Optional[torch.Tensor] = None,
#                          linear: Optional[bool] = True) -> torch.Tensor:
#     r"""
#     Update the `index_matrix` which convert `kernel_matrix` to loss.
#     If `index_matrix` is a tensor with shape (2 x batch_size, 2 x batch_size), then return `index_matrix`.
#     Else return a new tensor with shape (2 x batch_size, 2 x batch_size).
#     """
#     if index_matrix is None or index_matrix.size(0) != batch_size * 2:
#         index_matrix = torch.zeros(2 * batch_size, 2 * batch_size)
#         if linear:
#             for i in range(batch_size):
#                 s1, s2 = i, (i + 1) % batch_size
#                 t1, t2 = s1 + batch_size, s2 + batch_size
#                 index_matrix[s1, s2] = 1. / float(batch_size)
#                 index_matrix[t1, t2] = 1. / float(batch_size)
#                 index_matrix[s1, t2] = -1. / float(batch_size)
#                 index_matrix[s2, t1] = -1. / float(batch_size)
#         else:
#             for i in range(batch_size):
#                 for j in range(batch_size):
#                     if i != j:
#                         index_matrix[i][j] = 1. / float(batch_size * (batch_size - 1))
#                         index_matrix[i + batch_size][j + batch_size] = 1. / float(batch_size * (batch_size - 1))
#             for i in range(batch_size):
#                 for j in range(batch_size):
#                     index_matrix[i][j + batch_size] = -1. / float(batch_size * batch_size)
#                     index_matrix[i + batch_size][j] = -1. / float(batch_size * batch_size)
#     return index_matrix

# class GaussianKernel(nn.Module):
#     """Gaussian Kernel Matrix
#     Parameters:
#         - sigma (float, optional): bandwidth :math:`\sigma`. Default: None
#         - track_running_stats (bool, optional): If ``True``, this module tracks the running mean of :math:`\sigma^2`.
#           Otherwise, it won't track such statistics and always uses fix :math:`\sigma^2`. Default: ``True``
#         - alpha (float, optional): :math:`\alpha` which decides the magnitude of :math:`\sigma^2` when track_running_stats is set to ``True``
#     Inputs:
#         - X (tensor): input group :math:`X`
#     Shape:
#         - Inputs: :math:`(minibatch, F)` where F means the dimension of input features.
#         - Outputs: :math:`(minibatch, minibatch)`
#     """

#     def __init__(self, sigma: Optional[float] = None, track_running_stats: Optional[bool] = True,
#                  alpha: Optional[float] = 1.):
#         super(GaussianKernel, self).__init__()
#         assert track_running_stats or sigma is not None
#         self.sigma_square = torch.tensor(sigma * sigma) if sigma is not None else None
#         self.track_running_stats = track_running_stats
#         self.alpha = alpha

#     def forward(self, X: torch.Tensor) -> torch.Tensor:
#         l2_distance_square = ((X.unsqueeze(0) - X.unsqueeze(1)) ** 2).sum(2)

#         if self.track_running_stats:
#             self.sigma_square = self.alpha * torch.mean(l2_distance_square.detach())

#         return torch.exp(-l2_distance_square / (2 * self.sigma_square))


# def optimal_kernel_combinations(kernel_values: List[torch.Tensor]) -> torch.Tensor:
#     # use quadratic program to get optimal kernel
#     num_kernel = len(kernel_values)
#     kernel_values_numpy = array([float(k.detach().cpu().data.item()) for k in kernel_values])
#     if np.all(kernel_values_numpy <= 0):
#         beta = solve_qp(
#             P=-np.eye(num_kernel),
#             q=np.zeros(num_kernel),
#             A=kernel_values_numpy,
#             b=np.array([-1.]),
#             G=-np.eye(num_kernel),
#             h=np.zeros(num_kernel),
#         )
#     else:
#         beta = solve_qp(
#             P=np.eye(num_kernel),
#             q=np.zeros(num_kernel),
#             A=kernel_values_numpy,
#             b=np.array([1.]),
#             G=-np.eye(num_kernel),
#             h=np.zeros(num_kernel),
#         )
#     beta = beta / beta.sum(axis=0) * num_kernel  # normalize
#     return sum([k * b for (k, b) in zip(kernel_values, beta)])

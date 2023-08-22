import numba
import numpy as np


@numba.jit(
    numba.float32[:](
        numba.float32[:],
        numba.float32[:]
    ), nopython=True
)
def bbox_intersection(lhs_xywh, rhs_xywh):
    x1 = max(lhs_xywh[0], rhs_xywh[0])
    y1 = max(lhs_xywh[1], rhs_xywh[1])
    width = min(lhs_xywh[0] + lhs_xywh[2],  rhs_xywh[0] + rhs_xywh[2]) - x1
    height = min(lhs_xywh[1] + lhs_xywh[3], rhs_xywh[1] + rhs_xywh[3]) - y1

    if width <= 0 or height <= 0:
        return np.zeros((4,), dtype=np.float32)

    return np.array([x1, y1, width, height], dtype=np.float32)


@numba.jit(
    numba.types.List(dtype=numba.types.Tuple([numba.float32, numba.int32]))(
        numba.float32[:],
        numba.float32,
        numba.int32,
    ), nopython=True
)
def get_max_score_index(scores, threshold, top_k):
    """Get max scores with corresponding indices.
       scores: a set of scores.
       threshold: only consider scores higher than the threshold.
       top_k: if -1, keep all; otherwise, keep at most top_k.
       score_index_vec: store the sorted (score, index) pair.
    """
    indices = np.argwhere(scores > threshold)
    score_index_vec = [(scores[i], i) for i in indices]
    # Sort the score pair according to the scores in descending order
    score_index_vec.sort(reverse=True)

    # Keep top_k scores if needed.
    if top_k > 0:
        return score_index_vec[:top_k]

    return score_index_vec


@numba.jit(
    numba.float32(numba.float32[:]), nopython=True
)
def area(xywh_box):
    return xywh_box[2] * xywh_box[3]


@numba.jit(
    numba.float32(
        numba.float32[:],
        numba.float32[:]
    ), nopython=True
)
def jaccardDistance(a_xywh, b_xywh):
    Aa = area(a_xywh)
    Ab = area(b_xywh)
    eps = np.finfo(np.float32).eps

    if ((Aa + Ab) <= eps):
        # jaccard_index = 1 -> distance = 0
        return 0

    inter = bbox_intersection(a_xywh, b_xywh)

    Aab = area(inter)
    # distance = 1 - jaccard_index
    return np.float32(1) - Aab / (Aa + Ab - Aab)


@numba.jit(
    numba.float32(
        numba.float32[:],
        numba.float32[:]
    ), nopython=True
)
def rect_overlap(a_xywh, b_xywh):
    return np.float32(1) - jaccardDistance(a_xywh, b_xywh)


@numba.jit(
    numba.types.List(dtype=numba.int32)(
        numba.float32[:, :],
        numba.float32[:],
        numba.float32,
        numba.float32,
        numba.float32,
        numba.int32,
    )
)
def nms(bboxes_xywh, scores, score_threshold, nms_threshold, eta, top_k):
    """See: https://github.com/opencv/opencv/blob/ca0bd70cde431b1dd211254011dd9bcf965f582f/modules/dnn/src/nms.cpp
    """
    score_index_vec = get_max_score_index(scores, score_threshold, top_k)
    indices = []

    # Do nms.
    adaptive_threshold = nms_threshold

    for _, idx in score_index_vec:
        keep = True

        for kept_idx in indices:
            if not keep:
                break

            overlap = rect_overlap(bboxes_xywh[idx], bboxes_xywh[kept_idx])
            keep = overlap <= adaptive_threshold

        if keep:
            indices.append(idx)

            if eta < 1 and adaptive_threshold > 0.5:
                adaptive_threshold *= eta

    return indices

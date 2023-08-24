import numpy as np

cimport numpy as cnp
cimport cython

cnp.import_array()

FLOAT_ELEM_DTYPE = np.float32
INDEX_DTYPE = np.int32

ctypedef cnp.float32_t FLOAT_ELEM_DTYPE_COMPILE
ctypedef cnp.int32_t INDEX_DTYPE_COMPILE

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
cpdef cnp.ndarray[FLOAT_ELEM_DTYPE_COMPILE, ndim=1] bbox_intersection(
    cnp.ndarray[FLOAT_ELEM_DTYPE_COMPILE, ndim=1] lhs_xywh, 
    cnp.ndarray[FLOAT_ELEM_DTYPE_COMPILE, ndim=1] rhs_xywh):
    cdef float x1 = max(lhs_xywh[0], rhs_xywh[0])
    cdef float y1 = max(lhs_xywh[1], rhs_xywh[1])
    cdef float width = min(lhs_xywh[0] + lhs_xywh[2],  rhs_xywh[0] + rhs_xywh[2]) - x1
    cdef float height = min(lhs_xywh[1] + lhs_xywh[3], rhs_xywh[1] + rhs_xywh[3]) - y1

    if width <= 0 or height <= 0:
        return np.zeros((4,), dtype=FLOAT_ELEM_DTYPE)

    return np.array([x1, y1, width, height], dtype=FLOAT_ELEM_DTYPE)

@cython.boundscheck(False) 
@cython.wraparound(False)  
cpdef list[tuple[float, int]] get_max_score_index(
    cnp.ndarray[FLOAT_ELEM_DTYPE_COMPILE, ndim=1] scores, 
    float threshold, 
    int top_k):
    """Get max scores with corresponding indices.
       scores: a set of scores.
       threshold: only consider scores higher than the threshold.
       top_k: if -1, keep all; otherwise, keep at most top_k.
       score_index_vec: store the sorted (score, index) pair.
    """
    indices = np.argwhere(scores > threshold)
    cdef list[tuple[float, int]] score_index_vec = [(float(scores[i]), int(i)) for i in indices]
    # Sort the score pair according to the scores in descending order
    score_index_vec.sort(reverse=True)

    # Keep top_k scores if needed.
    if top_k > 0:
        return score_index_vec[:top_k]

    return score_index_vec

@cython.boundscheck(False) 
@cython.wraparound(False) 
cpdef inline float area(
    cnp.ndarray[FLOAT_ELEM_DTYPE_COMPILE, ndim=1] xywh_box):
    return xywh_box[2] * xywh_box[3]


@cython.boundscheck(False) 
@cython.wraparound(False)  
cpdef double jaccard_distance(
    cnp.ndarray[FLOAT_ELEM_DTYPE_COMPILE, ndim=1] a_xywh, 
    cnp.ndarray[FLOAT_ELEM_DTYPE_COMPILE, ndim=1] b_xywh):
    
    cdef double Aa = area(a_xywh)
    cdef double Ab = area(b_xywh)
    eps = np.finfo(np.double).eps

    if (Aa + Ab) <= eps:
        # jaccard_index = 1 -> distance = 0
        return 0

    cdef cnp.ndarray[FLOAT_ELEM_DTYPE_COMPILE, ndim=1] inter = bbox_intersection(a_xywh, b_xywh)

    cdef double Aab = area(inter)
    # distance = 1 - jaccard_index
    return 1 - Aab / (Aa + Ab - Aab)

@cython.boundscheck(False) 
@cython.wraparound(False)  
cpdef double rect_overlap(
    cnp.ndarray[FLOAT_ELEM_DTYPE_COMPILE, ndim=1] a_xywh, 
    cnp.ndarray[FLOAT_ELEM_DTYPE_COMPILE, ndim=1] b_xywh):
    return 1 - jaccard_distance(a_xywh, b_xywh)

@cython.boundscheck(False) 
@cython.wraparound(False)  
def boxes_nms(
    cnp.ndarray[FLOAT_ELEM_DTYPE_COMPILE, ndim=2] bboxes_xywh, 
    cnp.ndarray[FLOAT_ELEM_DTYPE_COMPILE, ndim=1] scores, 
    float score_threshold, 
    float nms_threshold, 
    float eta, 
    int top_k):
    """See: https://github.com/opencv/opencv/blob/ca0bd70cde431b1dd211254011dd9bcf965f582f/modules/dnn/src/nms.cpp
    """
    score_index_vec = get_max_score_index(scores, score_threshold, top_k)
    
    cdef list[int] indices = []

    # Do nms.
    cdef float adaptive_threshold = nms_threshold

    for _, idx in score_index_vec:
        keep = True

        for kept_idx in indices:
            if not keep:
                break

            overlap = rect_overlap(bboxes_xywh[idx].reshape(-1), bboxes_xywh[kept_idx].reshape(-1))
            keep = overlap <= adaptive_threshold

        if keep:
            indices.append(idx)

            if eta < 1 and adaptive_threshold > 0.5:
                adaptive_threshold *= eta

    return np.array(indices, dtype=INDEX_DTYPE)

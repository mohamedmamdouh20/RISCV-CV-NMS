#pragma once

#include "types.hpp"

#include <cstdint>
#include <algorithm>

namespace ref
{
    // Scalar educational NonMaxSuppression. Same contract and algorithm as
    // vec::NonMaxSuppression, including odd-even sort (corners only, y1 <= y2
    // and x1 <= x2, no box-format conversion). See lib/include/NMS.hpp.
    int NonMaxSuppression(const float* boxes,
                          const float* scores,
                          int num_batches,
                          int num_classes,
                          int spatial_dim,
                          int64_t max_output_boxes_per_class,
                          float iou_threshold,
                          const float* score_threshold,
                          CenterPointBox center_point_box,
                          int64_t* selected_indices);
}

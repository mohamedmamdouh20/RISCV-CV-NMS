#include "NMS.hpp"

static float calculate_iou(
    const float* a,
    const float* b,
    CenterPointBox center_point_box)
{
    float ax1, ay1, ax2, ay2;
    float bx1, by1, bx2, by2;

    if (center_point_box == CenterPointBox::Corners)
    {
        ay1 = a[0];
        ax1 = a[1];
        ay2 = a[2];
        ax2 = a[3];

        by1 = b[0];
        bx1 = b[1];
        by2 = b[2];
        bx2 = b[3];
    }
    else
    {
        // Center-point format: [x_center, y_center, width, height]
        ax1 = a[0] - a[2] / 2.0f;
        ay1 = a[1] - a[3] / 2.0f;
        ax2 = a[0] + a[2] / 2.0f;
        ay2 = a[1] + a[3] / 2.0f;

        bx1 = b[0] - b[2] / 2.0f;
        by1 = b[1] - b[3] / 2.0f;
        bx2 = b[0] + b[2] / 2.0f;
        by2 = b[1] + b[3] / 2.0f;
    }

    const float inter_x1 = std::max(ax1, bx1);
    const float inter_y1 = std::max(ay1, by1);
    const float inter_x2 = std::min(ax2, bx2);
    const float inter_y2 = std::min(ay2, by2);

    const float inter_w = std::max(0.0f, inter_x2 - inter_x1);
    const float inter_h = std::max(0.0f, inter_y2 - inter_y1);
    const float intersection = inter_w * inter_h;

    const float area_a =
        std::max(0.0f, ax2 - ax1) * std::max(0.0f, ay2 - ay1);

    const float area_b =
        std::max(0.0f, bx2 - bx1) * std::max(0.0f, by2 - by1);

    const float union_area = area_a + area_b - intersection;

    return union_area > 0.0f ? intersection / union_area : 0.0f;
}

static int score_suppression(
    int spatial_dim,
    const float *score_threshold,
    const float *scores_class,
    int *order,
    float *packed_scores)
{
    int candidate_count = 0;

    for (int i = 0; i < spatial_dim; ++i)
    {
        const float score = scores_class[i];

        if (score_threshold != nullptr && score < *score_threshold)
            continue;

        order[candidate_count] = i;
        packed_scores[candidate_count] = score;
        ++candidate_count;
    }

    return candidate_count;
}

static void EvenOddSort(int *order, float *scores, int count)
{
    for (int phase = 0; phase < count; ++phase)
    {
        const int start = phase % 2;

        for (int i = start; i + 1 < count; i += 2)
        {
            if (scores[i] < scores[i + 1])
            {
                std::swap(scores[i], scores[i + 1]);
                std::swap(order[i], order[i + 1]);
            }
        }
    }
}

static int greedy_IOU(
    const float *boxes,
    int spatial_dim,
    int batch_index,
    int class_index,
    const int *order,
    int order_count,
    float iou_threshold,
    int64_t max_output_boxes_per_class,
    CenterPointBox center_point_box,
    int64_t *selected_indices,
    int selected_count)
{
    bool *suppressed = new bool[order_count]();

    int selected = 0;

    for (int i = 0;
         i < order_count && selected < max_output_boxes_per_class;
         ++i)
    {
        if (suppressed[i])
            continue;

        const int box_i = order[i];

        selected_indices[selected_count * 3] = batch_index;
        selected_indices[selected_count * 3 + 1] = class_index;
        selected_indices[selected_count * 3 + 2] = box_i;

        ++selected;
        ++selected_count;

        const float *a =
            boxes + (batch_index * spatial_dim + box_i) * 4;

        for (int j = i + 1; j < order_count; ++j)
        {
            if (suppressed[j])
                continue;

            const int box_j = order[j];

            const float *b =
                boxes + (batch_index * spatial_dim + box_j) * 4;

            const float iou = calculate_iou(a, b, center_point_box);

            if (iou > iou_threshold)
                suppressed[j] = true;
        }
    }

    delete[] suppressed;
    return selected_count;
}

// Student exercise: scalar NonMaxSuppression.
// The contract is in test/reference/include/NMS.hpp (same as vec::). Match the
// ONNX golden fixtures used by NMS_test (test/unit_test/include/nms_onnx_cases.hpp).
// Keep this file scalar: no RVV intrinsics and no riscv_wrappers.hpp.
int ref::NonMaxSuppression(const float *boxes,                 // [[y1,x1,y2,x2],[y1,x1,y2,x2] ] ---> tensor with shape: [num_batches, spatial_dimension, 4]
                           const float *scores,                // tensor shape: [num_batches, num_classes, spatial_dimension]
                           int num_batches,                    // number of images processed together
                           int num_classes,                    // number of objects that the model can recognize
                           int spatial_dim,                    // the number of boxes
                           int64_t max_output_boxes_per_class, // max number of boxes per batch(image) per class(object detected)
                           float iou_threshold,
                           const float *score_threshold,
                           CenterPointBox center_point_box,
                           int64_t *selected_indices // this is the output buffer --> [batch,class,box]
)
{
    // 1) check that the number of boxes != 0 and  num_batches != 0 and num_classes != 0 and max_output_boxes_per_class != 0
    if (num_batches == 0 || num_classes == 0 || max_output_boxes_per_class == 0 || spatial_dim == 0)
    {
        return 0;
    }

    // temp-vars used by the algorithm:

    // 1- order: array of box indicies but in order and its length is the num_boxes
    int *order = new int[spatial_dim];

    // 2- suppressed: array of the suppressed boxes in a class
    // int *suppressed = new int[spatial_dim]; // may be array of boolean (true --> suppressed, false --> not suppressed)

    // 3- packed_scores: array of the scores for the class and each element is float
    float *packed_scores = new float[spatial_dim];

    int selected_count = 0;

    // loop through batches
    for (int b = 0; b < num_batches; b++)
    {
        // loop through classes in each batch
        for (int c = 0; c < num_classes; c++)
        {
            // 1) score suppression
            // float *box_class = boxes + (b * spatial_dim + i) * 4;
            // float *scores_class = scores[(b * num_classes + c) * spatial_dim + i];
            const float *scores_class =
                scores + (b * num_classes + c) * spatial_dim;

            int order_count = score_suppression( // variable holding how many elements in the order array as a candidate counts to be sorted
                spatial_dim,
                score_threshold,
                scores_class,
                order,
                packed_scores);

            // 2) even-odd sorting
            EvenOddSort(order, packed_scores, order_count);

            // 3) Select boxes using greedy IoU
            selected_count = greedy_IOU(
                boxes,
                spatial_dim,
                b,
                c,
                order,
                order_count,
                iou_threshold,
                max_output_boxes_per_class,
                center_point_box,
                selected_indices,
                selected_count);
        }
    }

    // delete[] suppressed;
    delete[] order;
    delete[] packed_scores;

    return selected_count;
}

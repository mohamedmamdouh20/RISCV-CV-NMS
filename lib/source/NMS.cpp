#include "NMS.hpp"
#include "riscv_wrappers.hpp"

static void vec_suppress_overlapping(
    const float *box_y1,
    const float *box_x1,
    const float *box_y2,
    const float *box_x2,
    const float *box_areas,
    int selected_pos,
    int start,
    int candidate_count,
    uint8_t *suppressed,
    float iou_threshold)
{
    // The selected box is already represented in SoA form.
    const float ay1 = box_y1[selected_pos];
    const float ax1 = box_x1[selected_pos];
    const float ay2 = box_y2[selected_pos];
    const float ax2 = box_x2[selected_pos];
    const float area_a = box_areas[selected_pos];

    for (int offset = start; offset < candidate_count;)
    {
        const size_t vl =
            rvv::SetVlF32M2(candidate_count - offset);

        // Load contiguous coordinate columns for the later candidates.
        const auto by1 = rvv::LoadF32M2(box_y1 + offset, vl);
        const auto bx1 = rvv::LoadF32M2(box_x1 + offset, vl);
        const auto by2 = rvv::LoadF32M2(box_y2 + offset, vl);
        const auto bx2 = rvv::LoadF32M2(box_x2 + offset, vl);
        const auto area_b = rvv::LoadF32M2(box_areas + offset, vl);

        // Intersection rectangle.
        const auto iy1 = rvv::MaxF32ScalarM2(by1, ay1, vl);
        const auto ix1 = rvv::MaxF32ScalarM2(bx1, ax1, vl);
        const auto iy2 = rvv::MinF32ScalarM2(by2, ay2, vl);
        const auto ix2 = rvv::MinF32ScalarM2(bx2, ax2, vl);

        const auto zero_v = rvv::SplatF32M2(0.0f, vl);

        const auto height = rvv::MaxF32M2(
            rvv::SubF32M2(iy2, iy1, vl), zero_v, vl);

        const auto width = rvv::MaxF32M2(
            rvv::SubF32M2(ix2, ix1, vl), zero_v, vl);

        const auto intersection =
            rvv::MulF32M2(height, width, vl);

        // union = area_a + area_b - intersection.
        const auto union_area = rvv::SubF32M2(
            rvv::AddF32ScalarM2(area_b, area_a, vl),
            intersection,
            vl);

        // Avoid division by zero, then explicitly set IoU to zero
        // wherever union_area <= 0.
        const auto positive_union =
            rvv::MaskF32GreaterThanScalarM2(
                union_area, 0.0f, vl);

        const auto safe_union = rvv::MergeF32M2(
            rvv::SplatF32M2(1.0f, vl),
            union_area,
            positive_union,
            vl);

        const auto raw_iou =
            rvv::DivF32M2(intersection, safe_union, vl);

        const auto iou = rvv::MergeF32M2(
            zero_v,
            raw_iou,
            positive_union,
            vl);

        // Suppress only candidates that are not already suppressed
        // and whose IoU strictly exceeds the threshold.
        const auto iou_too_large =
            rvv::MaskF32GreaterThanScalarM2(
                iou, iou_threshold, vl);

        const auto flags =
            rvv::LoadU8Mf2(suppressed + offset, vl);

        const auto not_yet_suppressed =
            rvv::MaskU8NotEqualScalarMf2(flags, 1, vl);

        const auto mark_suppressed =
            rvv::AndMaskM2(
                not_yet_suppressed, iou_too_large, vl);

        // Masked store: write 1 only to newly suppressed candidates.
        const auto ones = rvv::SplatU8Mf2(1, vl);

        rvv::StoreU8Mf2Masked(
            suppressed + offset,
            ones,
            mark_suppressed,
            vl);

        offset += static_cast<int>(vl);
    }
}

int vec::NonMaxSuppression(
    const float *boxes,
    const float *scores,
    int num_batches,
    int num_classes,
    int spatial_dim,
    int64_t max_output_boxes_per_class,
    float iou_threshold,
    const float *score_threshold,
    CenterPointBox center_point_box,
    int64_t *selected_indices)
{
    int *order = new int[spatial_dim];
    float *packed_scores = new float[spatial_dim];
    uint8_t *suppressed = new uint8_t[spatial_dim];

    float *box_y1 = new float[spatial_dim];
    float *box_x1 = new float[spatial_dim];
    float *box_y2 = new float[spatial_dim];
    float *box_x2 = new float[spatial_dim];
    float *box_areas = new float[spatial_dim];

    int selected_count = 0;

    for (int batch = 0; batch < num_batches; ++batch)
    {
        for (int cls = 0; cls < num_classes; ++cls)
        {
            const float *scores_class =
                scores + (batch * num_classes + cls) * spatial_dim;

            int candidate_count = 0;

            // Stage 1: score filtering.
            // Process scores in RVV vectors.
            const float threshold = *score_threshold;

            for (int offset = 0; offset < spatial_dim;)
            {
                const size_t vl =
                    rvv::SetVlF32M4(spatial_dim - offset);

                // Load scores for this chunk.
                const rvv::F32M4 score_vec =
                    rvv::LoadF32M4(scores_class + offset, vl);

                // Keep scores >= threshold.
                const rvv::MaskF32M4 keep =
                    rvv::MaskF32GreaterEqualScalarM4(
                        score_vec, threshold, vl);

                // Count how many candidates survive.
                const size_t kept =
                    rvv::CountTrueF32M4(keep, vl);

                // Compact the surviving scores.
                const rvv::F32M4 compact_scores =
                    rvv::CompressF32M4(score_vec, keep, vl);

                rvv::StoreF32M4(
                    packed_scores + candidate_count,
                    compact_scores,
                    kept);

                // Generate the original box indices.
                rvv::U32M4 indices = rvv::IotaU32M4(vl);
                indices = rvv::AddU32M4(
                    indices, static_cast<uint32_t>(offset), vl);

                // Compact indices using the same mask.
                const rvv::U32M4 compact_indices =
                    rvv::CompressU32M4(indices, keep, vl);

                rvv::StoreI32M4(
                    order + candidate_count,
                    rvv::BitcastU32ToI32M4(compact_indices),
                    kept);

                candidate_count += static_cast<int>(kept);

                offset += static_cast<int>(vl);
            }

            // Stage 2: descending score sort.

            // Stage 2: vectorized odd-even transposition sort.
            // Sort scores in descending order and keep box indices aligned.
            for (int phase = 0; phase < candidate_count; ++phase)
            {
                const int start = phase % 2;

                // Number of non-overlapping adjacent pairs in this phase.
                const int pair_count =
                    (candidate_count - start) / 2;

                for (int offset = 0; offset < pair_count;)
                {
                    const size_t vl =
                        rvv::SetVlF32M4(pair_count - offset);

                    // Each lane handles one pair:
                    // left positions:  start + 2 * lane
                    // right positions: start + 2 * lane + 1
                    const ptrdiff_t stride =
                        2 * static_cast<ptrdiff_t>(sizeof(float));

                    float *left_scores =
                        packed_scores + start + 2 * offset;
                    float *right_scores = left_scores + 1;

                    int *left_indices =
                        order + start + 2 * offset;
                    int *right_indices = left_indices + 1;

                    // Load the left and right scores of each pair.
                    const rvv::F32M4 left =
                        rvv::LoadStridedF32M4(
                            left_scores, stride, vl);

                    const rvv::F32M4 right =
                        rvv::LoadStridedF32M4(
                            right_scores, stride, vl);

                    // Swap when left score < right score.
                    const rvv::MaskF32M4 swap_mask =
                        rvv::MaskF32LessThanM4(left, right, vl);

                    // Select the larger score for the left position.
                    const rvv::F32M4 new_left =
                        rvv::MergeF32M4(
                            left, right, swap_mask, vl);

                    // Select the smaller score for the right position.
                    const rvv::F32M4 new_right =
                        rvv::MergeF32M4(
                            right, left, swap_mask, vl);

                    // Load the corresponding box indices.
                    const rvv::I32M4 left_id =
                        rvv::LoadStridedI32M4(
                            left_indices, stride, vl);

                    const rvv::I32M4 right_id =
                        rvv::LoadStridedI32M4(
                            right_indices, stride, vl);

                    // Apply the exact same swap mask to the indices.
                    const rvv::I32M4 new_left_id =
                        rvv::MergeI32M4(
                            left_id, right_id, swap_mask, vl);

                    const rvv::I32M4 new_right_id =
                        rvv::MergeI32M4(
                            right_id, left_id, swap_mask, vl);

                    // Store scores and indices back to their positions.
                    rvv::StoreStridedF32M4(
                        left_scores, stride, new_left, vl);

                    rvv::StoreStridedF32M4(
                        right_scores, stride, new_right, vl);

                    rvv::StoreStridedI32M4(
                        left_indices, stride, new_left_id, vl);

                    rvv::StoreStridedI32M4(
                        right_indices, stride, new_right_id, vl);

                    offset += static_cast<int>(vl);
                }
            }

            // intermediate step
            for (int i = 0; i < candidate_count; ++i)
            {
                const int box_index = order[i];

                const float *b =
                    boxes + (batch * spatial_dim + box_index) * 4;

                box_y1[i] = b[0];
                box_x1[i] = b[1];
                box_y2[i] = b[2];
                box_x2[i] = b[3];

                const float h = (b[2] > b[0]) ? (b[2] - b[0]) : 0.0f;
                const float w = (b[3] > b[1]) ? (b[3] - b[1]) : 0.0f;

                box_areas[i] = h * w;
            }

            // Stage 3: greedy NMS.
            for (int i = 0; i < candidate_count; ++i)
            {
                suppressed[i] = 0;
            }

            int selected_for_class = 0;

            for (int i = 0;
                 i < candidate_count &&
                 selected_for_class < max_output_boxes_per_class;
                 ++i)
            {
                if (suppressed[i] != 0)
                    continue;

                const int box_i = order[i];

                selected_indices[selected_count * 3] = batch;
                selected_indices[selected_count * 3 + 1] = cls;
                selected_indices[selected_count * 3 + 2] = box_i;

                ++selected_count;
                ++selected_for_class;

                // Scan only later candidates. Their coordinates are
                // contiguous in the SoA arrays and can be vector-loaded.
                vec_suppress_overlapping(
                    box_y1,
                    box_x1,
                    box_y2,
                    box_x2,
                    box_areas,
                    i,
                    i + 1,
                    candidate_count,
                    suppressed,
                    iou_threshold);
            }
        }
    }

    delete[] box_areas;
    delete[] box_x2;
    delete[] box_y2;
    delete[] box_x1;
    delete[] box_y1;

    delete[] suppressed;
    delete[] packed_scores;
    delete[] order;
    return selected_count;
}

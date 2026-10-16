#pragma once

// Readable names over GNU RVV C intrinsics. Kernels include this header and
// call rvv::* only. Add a wrapper here when a kernel needs a new intrinsic.

#if !defined(__riscv_vector)
#error "riscv_wrappers.hpp requires RISC-V Vector (compile with -march including v)"
#endif
#include <riscv_vector.h>

#include <cstddef>
#include <cstdint>

namespace rvv
{
    using F32M4 = vfloat32m4_t;
    using I32M4 = vint32m4_t;
    using U32M4 = vuint32m4_t;
    using MaskF32M4 = vbool8_t;
    using U8M4 = vuint8m4_t;

    // How many f32 (or i32) lanes to process this iteration (LMUL=4).
    inline size_t SetVlF32M4(size_t remaining)
    {
        return __riscv_vsetvl_e32m4(remaining);
    }

    inline size_t SetVlU8M4(size_t remaining)
    {
        return __riscv_vsetvl_e8m4(remaining);
    }

    inline I32M4 LoadI32M4(const int *src, size_t vl)
    {
        return __riscv_vle32_v_i32m4(src, vl);
    }

    inline F32M4 LoadF32M4(const float *src, size_t vl)
    {
        return __riscv_vle32_v_f32m4(src, vl);
    }

    inline U8M4 LoadU8M4(const uint8_t *src, size_t vl)
    {
        return __riscv_vle8_v_u8m4(src, vl);
    }

    inline void StoreI32M4(int *dst, I32M4 values, size_t vl)
    {
        __riscv_vse32_v_i32m4(dst, values, vl);
    }

    inline void StoreF32M4(float *dst, F32M4 values, size_t vl)
    {
        __riscv_vse32_v_f32m4(dst, values, vl);
    }

    inline void StoreU8M4(uint8_t *dst, U8M4 values, size_t vl)
    {
        __riscv_vse8_v_u8m4(dst, values, vl);
    }

    // Load/store with a byte stride between consecutive lanes (e.g. every
    // other f32: stride = 2 * sizeof(float)).
    inline F32M4 LoadStridedF32M4(const float *src, ptrdiff_t stride_bytes, size_t vl)
    {
        return __riscv_vlse32_v_f32m4(src, stride_bytes, vl);
    }

    inline I32M4 LoadStridedI32M4(const int *src, ptrdiff_t stride_bytes, size_t vl)
    {
        return __riscv_vlse32_v_i32m4(src, stride_bytes, vl);
    }

    inline void StoreStridedF32M4(float *dst, ptrdiff_t stride_bytes, F32M4 values, size_t vl)
    {
        __riscv_vsse32_v_f32m4(dst, stride_bytes, values, vl);
    }

    inline void StoreStridedI32M4(int *dst, ptrdiff_t stride_bytes, I32M4 values, size_t vl)
    {
        __riscv_vsse32_v_i32m4(dst, stride_bytes, values, vl);
    }

    inline U32M4 BitcastI32ToU32M4(I32M4 values)
    {
        return __riscv_vreinterpret_v_i32m4_u32m4(values);
    }

    inline U32M4 ShiftLeftU32M4(U32M4 values, size_t bits, size_t vl)
    {
        return __riscv_vsll_vx_u32m4(values, bits, vl);
    }

    // Indexed load: lane i reads base[byte_offsets[i] / 4] as f32.
    inline F32M4 GatherF32M4(const float *base, U32M4 byte_offsets, size_t vl)
    {
        return __riscv_vluxei32_v_f32m4(base, byte_offsets, vl);
    }

    inline MaskF32M4 MaskF32LessThanM4(F32M4 left, F32M4 right, size_t vl)
    {
        return __riscv_vmflt_vv_f32m4_b8(left, right, vl);
    }

    // Lane = mask ? when_true : when_false.
    inline F32M4 MergeF32M4(F32M4 when_false, F32M4 when_true, MaskF32M4 mask, size_t vl)
    {
        return __riscv_vmerge_vvm_f32m4(when_false, when_true, mask, vl);
    }

    inline I32M4 MergeI32M4(I32M4 when_false, I32M4 when_true, MaskF32M4 mask, size_t vl)
    {
        return __riscv_vmerge_vvm_i32m4(when_false, when_true, mask, vl);
    }

    inline U8M4 AddU8WrappingM4(U8M4 a, U8M4 b, size_t vl)
    {
        return __riscv_vadd_vv_u8m4(a, b, vl);
    }

    inline U8M4 AddU8SaturatingM4(U8M4 a, U8M4 b, size_t vl)
    {
        return __riscv_vsaddu_vv_u8m4(a, b, vl);
    }

    inline U8M4 SplatU8M4(uint8_t value, size_t vl)
    {
        return __riscv_vmv_v_x_u8m4(value, vl);
    }

    inline void FillU8(uint8_t *dst, uint8_t value, size_t count)
    {
        size_t remaining = count;
        size_t offset = 0;
        while (remaining > 0)
        {
            const size_t vl = SetVlU8M4(remaining);
            StoreU8M4(dst + offset, SplatU8M4(value, vl), vl);
            offset += vl;
            remaining -= vl;
        }
    }

    // Lane i holds i (0, 1, 2, ...).
    inline U32M4 IotaU32M4(size_t vl)
    {
        return __riscv_vid_v_u32m4(vl);
    }

    inline U32M4 AddU32M4(U32M4 values, uint32_t addend, size_t vl)
    {
        return __riscv_vadd_vx_u32m4(values, addend, vl);
    }

    inline I32M4 BitcastU32ToI32M4(U32M4 values)
    {
        return __riscv_vreinterpret_v_u32m4_i32m4(values);
    }

    inline MaskF32M4 MaskF32GreaterEqualScalarM4(F32M4 values, float scalar, size_t vl)
    {
        return __riscv_vmfge_vf_f32m4_b8(values, scalar, vl);
    }

    // Pack kept lanes to the front, preserving order.
    inline U32M4 CompressU32M4(U32M4 values, MaskF32M4 keep, size_t vl)
    {
        return __riscv_vcompress_vm_u32m4(values, keep, vl);
    }

    inline F32M4 CompressF32M4(F32M4 values, MaskF32M4 keep, size_t vl)
    {
        return __riscv_vcompress_vm_f32m4(values, keep, vl);
    }

    inline size_t CountTrueF32M4(MaskF32M4 mask, size_t vl)
    {
        return __riscv_vcpop_m_b8(mask, vl);
    }

    using F32M2 = vfloat32m2_t;
    using I32M2 = vint32m2_t;
    using U32M2 = vuint32m2_t;
    using MaskF32M2 = vbool16_t;
    using U8Mf2 = vuint8mf2_t;

    inline size_t SetVlF32M2(size_t remaining)
    {
        return __riscv_vsetvl_e32m2(remaining);
    }

    inline F32M2 LoadF32M2(const float *src, size_t vl)
    {
        return __riscv_vle32_v_f32m2(src, vl);
    }

    inline U8Mf2 LoadU8Mf2(const uint8_t *src, size_t vl)
    {
        return __riscv_vle8_v_u8mf2(src, vl);
    }

    inline void StoreU8Mf2Masked(uint8_t *dst, U8Mf2 values, MaskF32M2 mask, size_t vl)
    {
        __riscv_vse8_v_u8mf2_m(mask, dst, values, vl);
    }

    inline U8Mf2 SplatU8Mf2(uint8_t value, size_t vl)
    {
        return __riscv_vmv_v_x_u8mf2(value, vl);
    }

    inline MaskF32M2 MaskU8NotEqualScalarMf2(U8Mf2 values, uint8_t scalar, size_t vl)
    {
        return __riscv_vmsne_vx_u8mf2_b16(values, scalar, vl);
    }

    // Index of first true lane, or -1 if none.
    inline int FirstTrueM2(MaskF32M2 mask, size_t vl)
    {
        return static_cast<int>(__riscv_vfirst_m_b16(mask, vl));
    }

    inline F32M2 MinF32ScalarM2(F32M2 values, float scalar, size_t vl)
    {
        return __riscv_vfmin_vf_f32m2(values, scalar, vl);
    }

    inline F32M2 MaxF32ScalarM2(F32M2 values, float scalar, size_t vl)
    {
        return __riscv_vfmax_vf_f32m2(values, scalar, vl);
    }

    inline F32M2 SubF32M2(F32M2 left, F32M2 right, size_t vl)
    {
        return __riscv_vfsub_vv_f32m2(left, right, vl);
    }

    inline F32M2 MulF32M2(F32M2 left, F32M2 right, size_t vl)
    {
        return __riscv_vfmul_vv_f32m2(left, right, vl);
    }

    inline F32M2 AddF32ScalarM2(F32M2 values, float scalar, size_t vl)
    {
        return __riscv_vfadd_vf_f32m2(values, scalar, vl);
    }

    inline F32M2 DivF32M2(F32M2 numer, F32M2 denom, size_t vl)
    {
        return __riscv_vfdiv_vv_f32m2(numer, denom, vl);
    }

    inline F32M2 SplatF32M2(float value, size_t vl)
    {
        return __riscv_vfmv_v_f_f32m2(value, vl);
    }

    inline MaskF32M2 MaskF32GreaterThanScalarM2(F32M2 values, float scalar, size_t vl)
    {
        return __riscv_vmfgt_vf_f32m2_b16(values, scalar, vl);
    }

    inline MaskF32M2 AndMaskM2(MaskF32M2 a, MaskF32M2 b, size_t vl)
    {
        return __riscv_vmand_mm_b16(a, b, vl);
    }

    inline F32M2 MergeF32M2(F32M2 when_false, F32M2 when_true, MaskF32M2 mask, size_t vl)
    {
        return __riscv_vmerge_vvm_f32m2(when_false, when_true, mask, vl);
    }

    inline MaskF32M2 MaskF32LessThanScalarM2(F32M2 values, float scalar, size_t vl)
    {
        return __riscv_vmflt_vf_f32m2_b16(values, scalar, vl);
    }

    inline MaskF32M2 MaskF32LessEqualScalarM2(F32M2 values, float scalar, size_t vl)
    {
        return __riscv_vmfle_vf_f32m2_b16(values, scalar, vl);
    }

    inline MaskF32M2 MaskF32GreaterEqualScalarM2(F32M2 values, float scalar, size_t vl)
    {
        return __riscv_vmfge_vf_f32m2_b16(values, scalar, vl);
    }

    inline F32M2 LoadStridedF32M2(const float *src, ptrdiff_t stride_bytes, size_t vl)
    {
        return __riscv_vlse32_v_f32m2(src, stride_bytes, vl);
    }

    inline F32M2 MaxF32M2(F32M2 left, F32M2 right, size_t vl)
    {
        return __riscv_vfmax_vv_f32m2(left, right, vl);
    }

    inline F32M2 MinF32M2(F32M2 left, F32M2 right, size_t vl)
    {
        return __riscv_vfmin_vv_f32m2(left, right, vl);
    }

    inline void StoreF32M2(float *dst, F32M2 values, size_t vl)
    {
        __riscv_vse32_v_f32m2(dst, values, vl);
    }

}

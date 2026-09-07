#pragma once

#include <torch/extension.h>

void exl3_fat_gemm(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    int64_t K,
    bool mcg,
    bool mul1);

void exl3_fat_gemm_scatter(
    at::Tensor a,
    at::Tensor packed,
    at::Tensor out,
    at::Tensor svh,
    at::Tensor token_idx,
    at::Tensor route_weight,
    int64_t K,
    bool mcg,
    bool mul1);

void exl3_fat_gemm2(
    at::Tensor a,
    at::Tensor packed0,
    at::Tensor packed1,
    at::Tensor out,
    at::Tensor svh0,
    at::Tensor svh1,
    int64_t K,
    bool mcg,
    bool mul1);

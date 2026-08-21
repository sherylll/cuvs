/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuvs/core/export.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdspan.hpp>

#include <cstddef>
#include <cstdint>

namespace CUVS_EXPORT cuvs {
namespace preprocessing {
namespace quantize {
namespace bbq {

/**
 * @defgroup bbq Better Binary Quantization utilities
 * @{
 */

using code_layout = cuvs::neighbors::bbq_code_layout;

/** @} */  // end of bbq group

}  // namespace bbq
}  // namespace quantize
}  // namespace preprocessing
}  // namespace CUVS_EXPORT cuvs

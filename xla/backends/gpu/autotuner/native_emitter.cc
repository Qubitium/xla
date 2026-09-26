/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "xla/backends/gpu/autotuner/native_emitter.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/tsl/platform/status_macros.h"
#include "llvm/ADT/SmallSet.h"
#include "xla/autotuning.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

// A use and the value it consumes. The value can be a parameter of a fusion
// that consumes the original instruction.
using HloUse = std::pair<const HloInstruction*, const HloInstruction*>;
using HloValue = std::pair<const HloInstruction*, int64_t>;
constexpr int64_t kNotTupleElement = -1;

// Follow uses through fusion and tuple/GTE boundaries without treating those
// wrappers as operations in the RMSNorm pattern.
std::vector<HloUse> UsersThroughFusions(const HloInstruction* value) {
  std::vector<HloUse> uses;
  std::vector<HloValue> aliases{{value, kNotTupleElement}};
  std::vector<HloValue> visited;
  while (!aliases.empty()) {
    const auto [alias, tuple_index] = aliases.back();
    aliases.pop_back();
    if (std::find(visited.begin(), visited.end(),
                  HloValue{alias, tuple_index}) != visited.end()) {
      continue;
    }
    visited.emplace_back(alias, tuple_index);
    const HloComputation* parent = alias->parent();
    if (parent->IsFusionComputation() && parent->root_instruction() == alias) {
      aliases.emplace_back(parent->FusionInstruction(), tuple_index);
    }
    for (const HloInstruction* user : alias->users()) {
      if (user->opcode() == HloOpcode::kFusion) {
        const auto* fusion = Cast<const HloFusionInstruction>(user);
        for (int64_t i = 0; i < fusion->operand_count(); ++i) {
          if (fusion->operand(i) == alias) {
            aliases.emplace_back(fusion->fused_parameter(i), tuple_index);
          }
        }
      } else if (tuple_index != kNotTupleElement) {
        if (user->opcode() == HloOpcode::kGetTupleElement &&
            user->tuple_index() == tuple_index) {
          aliases.emplace_back(user, kNotTupleElement);
        }
      } else if (user->opcode() == HloOpcode::kTuple) {
        for (int64_t i = 0; i < user->operand_count(); ++i) {
          if (user->operand(i) == alias) {
            aliases.emplace_back(user, i);
          }
        }
      } else {
        uses.emplace_back(user, alias);
      }
    }
  }
  return uses;
}

const HloInstruction* OtherOperand(const HloInstruction* binary,
                                   const HloInstruction* operand) {
  if (binary->operand_count() != 2) {
    return nullptr;
  }
  if (binary->operand(0) == operand) {
    return binary->operand(1);
  }
  if (binary->operand(1) == operand) {
    return binary->operand(0);
  }
  return nullptr;
}

bool F32ScalarConstant(const HloInstruction* instr, float* value) {
  if (instr == nullptr) {
    return false;
  }
  if (instr->opcode() == HloOpcode::kBroadcast) {
    instr = instr->operand(0);
  }
  if (instr->opcode() != HloOpcode::kConstant ||
      instr->shape().element_type() != PrimitiveType::F32 ||
      instr->shape().dimensions_size() != 0) {
    return false;
  }
  *value = instr->literal().GetFirstElement<float>();
  return std::isfinite(*value);
}

// Map a fusion parameter back to its caller's operand. Converts preserve the
// source activation for this pattern.
const HloInstruction* ActivationSource(const HloInstruction* instr) {
  while (true) {
    if (instr->opcode() == HloOpcode::kConvert) {
      instr = instr->operand(0);
    } else if (instr->opcode() == HloOpcode::kParameter &&
               instr->parent()->IsFusionComputation()) {
      instr = instr->parent()->FusionInstruction()->operand(
          instr->parameter_number());
    } else {
      return instr;
    }
  }
}

bool HasRmsNormConsumer(const HloInstruction* reduction,
                        const HloInstruction* activation, int64_t hidden_size) {
  for (const auto& [mean, sum] : UsersThroughFusions(reduction)) {
    if (mean->opcode() != HloOpcode::kMultiply) {
      continue;
    }
    float scale;
    if (!F32ScalarConstant(OtherOperand(mean, sum), &scale) ||
        std::abs(scale - 1.0f / hidden_size) > 1e-5f / hidden_size) {
      continue;
    }
    for (const auto& [variance_eps, mean_value] : UsersThroughFusions(mean)) {
      if (variance_eps->opcode() != HloOpcode::kAdd) {
        continue;
      }
      float epsilon;
      if (!F32ScalarConstant(OtherOperand(variance_eps, mean_value),
                             &epsilon) ||
          epsilon < 0.0f) {
        continue;
      }
      for (const auto& [inv_rms, variance] :
           UsersThroughFusions(variance_eps)) {
        if (inv_rms->opcode() != HloOpcode::kRsqrt ||
            inv_rms->operand(0) != variance) {
          continue;
        }
        for (const auto& [broadcast, inv_rms_value] :
             UsersThroughFusions(inv_rms)) {
          if (broadcast->opcode() != HloOpcode::kBroadcast ||
              broadcast->operand(0) != inv_rms_value) {
            continue;
          }
          for (const auto& [normalized, inv_rms_b] :
               UsersThroughFusions(broadcast)) {
            if (normalized->opcode() != HloOpcode::kMultiply) {
              continue;
            }
            const HloInstruction* normalized_input =
                OtherOperand(normalized, inv_rms_b);
            if (normalized_input != nullptr &&
                ActivationSource(normalized_input) ==
                    ActivationSource(activation)) {
              return true;
            }
          }
        }
      }
    }
  }
  return false;
}

// Identify the square-sum and its normalization consumers. A square-sum by
// itself can also be an L2 norm or another unrelated reduction.
bool IsRmsNormReductionFusion(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }

  const auto* fusion = Cast<const HloFusionInstruction>(&instr);
  const HloComputation* computation = fusion->fused_instructions_computation();
  if (computation == nullptr) {
    return false;
  }

  for (const HloInstruction* op : computation->instructions()) {
    if (op->opcode() != HloOpcode::kReduce || op->operand_count() != 2 ||
        op->dimensions().size() != 1 ||
        op->shape().element_type() != PrimitiveType::F32 ||
        op->operand(0)->opcode() != HloOpcode::kMultiply ||
        op->to_apply()->root_instruction()->opcode() != HloOpcode::kAdd) {
      continue;
    }
    float zero;
    if (!F32ScalarConstant(op->operand(1), &zero) || zero != 0.0f) {
      continue;
    }

    const HloInstruction* square = op->operand(0);
    if (square->operand_count() != 2 ||
        square->operand(0) != square->operand(1)) {
      continue;
    }

    const HloInstruction* square_input = square->operand(0);
    const HloInstruction* activation = square_input;
    if (square_input->opcode() == HloOpcode::kConvert) {
      if (square_input->operand_count() != 1 ||
          square_input->shape().element_type() != PrimitiveType::F32) {
        continue;
      }
      activation = square_input->operand(0);
    }

    if (activation->opcode() != HloOpcode::kParameter ||
        !activation->shape().IsArray() ||
        activation->shape().dimensions_size() < 1 ||
        (activation->shape().element_type() != PrimitiveType::F16 &&
         activation->shape().element_type() != PrimitiveType::BF16 &&
         activation->shape().element_type() != PrimitiveType::F32) ||
        square_input->shape().element_type() != PrimitiveType::F32 ||
        square_input->shape().dimensions_size() !=
            activation->shape().dimensions_size()) {
      continue;
    }

    const int64_t hidden_axis = activation->shape().dimensions_size() - 1;
    if (op->dimensions()[0] == hidden_axis &&
        activation->shape().dimensions(hidden_axis) > 0 &&
        HasRmsNormConsumer(op, square_input,
                           activation->shape().dimensions(hidden_axis))) {
      return true;
    }
  }
  return false;
}

llvm::SmallSet<int64_t, 4> ComputeUnrollFactors(
    const HloInstruction& instr, int64_t default_unroll_factor,
    const se::DeviceDescription& device_description) {
  llvm::SmallSet<int64_t, 4> unroll_factors;
  unroll_factors.insert(default_unroll_factor);

  auto analysis = HloFusionAnalysis::Create(instr, device_description);
  int64_t num_elements = ShapeUtil::ElementsIn(analysis.first_result_shape());
  int64_t n_threads_max = analysis.device_info().threads_per_core_limit() *
                          analysis.device_info().core_count();
  if (num_elements >= n_threads_max &&
      !MayCausePerformanceDropIfUnrolled(analysis.fusion())) {
    int64_t max_unroll_factor = MaxUnrollFactor(&analysis);
    unroll_factors.insert(max_unroll_factor);
    if (max_unroll_factor > 1) {
      unroll_factors.insert(max_unroll_factor / 2);
    }
  }
  return unroll_factors;
}

}  // namespace

// Returns true if the given instruction is a fusion instruction that is
// supported by the native emitter backend.
//
// There is no guarantee that the native emitter backend can actually compile if
// it has a config for another backend, and we currently don't have an easy way
// to check that. Therefore, we only support fusions that are already set up to
// go through the native emitter.
bool NativeEmitterBackend::IsSupported(const HloInstruction& instr) {
  if (instr.opcode() != HloOpcode::kFusion) {
    return false;
  }
  auto fusion = Cast<HloFusionInstruction>(&instr);
  return fusion->fusion_kind() != HloInstruction::FusionKind::kCustom &&
         !IsRmsNormReductionFusion(instr);
}

absl::StatusOr<std::vector<std::unique_ptr<BackendConfig>>>
NativeEmitterBackend::GetSupportedConfigs(const HloInstruction& instr) {
  std::vector<std::unique_ptr<BackendConfig>> configs;
  if (!IsSupported(instr)) {
    return configs;
  }

  ASSIGN_OR_RETURN(std::unique_ptr<BackendConfig> default_config,
                   GetDefaultConfig(instr));
  if (!default_config->has_native_emitter()) {
    return absl::InternalError("Expected NativeEmitterBackendConfig.");
  }
  // Tune unroll factor for loops.
  if (debug_options().xla_gpu_native_emitter_tune_unroll_factor_for_loops() &&
      default_config->native_emitter().type() ==
          NativeEmitterType::NATIVE_EMITTER_TYPE_LOOP) {
    llvm::SmallSet<int64_t, 4> unroll_factors = ComputeUnrollFactors(
        instr, default_config->native_emitter().unroll_factor(),
        target_config().device_description);
    for (int64_t unroll_factor : unroll_factors) {
      auto config_ptr = std::make_unique<BackendConfig>();
      NativeEmitterBackendConfig* config = config_ptr->mutable_native_emitter();
      config->set_type(NativeEmitterType::NATIVE_EMITTER_TYPE_LOOP);
      config->set_unroll_factor(unroll_factor);
      configs.push_back(std::move(config_ptr));
    }
    return configs;
  }
  configs.push_back(std::move(default_config));
  return configs;
}

absl::StatusOr<std::unique_ptr<BackendConfig>>
NativeEmitterBackend::GetDefaultConfig(const HloInstruction& instr) {
  if (IsRmsNormReductionFusion(instr)) {
    return absl::NotFoundError(
        "NativeEmitter is disabled for RMSNorm reduction fusions.");
  }
  auto config = std::make_unique<BackendConfig>();
  NativeEmitterBackendConfig* native_emitter_config =
      config->mutable_native_emitter();
  if (IsSupported(instr) &&
      debug_options().xla_gpu_native_emitter_tune_unroll_factor_for_loops()) {
    se::DeviceDescription device_description =
        target_config().device_description;
    HloFusionAnalysis fusion_analysis =
        HloFusionAnalysis::Create(instr, device_description);
    if (fusion_analysis.emitter_fusion_kind() ==
        HloFusionAnalysis::EmitterFusionKind::kLoop) {
      native_emitter_config->set_type(
          NativeEmitterType::NATIVE_EMITTER_TYPE_LOOP);
      native_emitter_config->set_unroll_factor(
          ComputeLoopFusionConfig(fusion_analysis));
    }
  }
  return config;
}

absl::Status NativeEmitterBackend::ApplyConfig(HloInstruction& instr,
                                               const BackendConfig& config) {
  if (!config.has_native_emitter()) {
    return absl::InvalidArgumentError("Expected NativeEmitterBackendConfig.");
  }
  const NativeEmitterBackendConfig& native_emitter_fusion_config =
      config.native_emitter();
  auto fusion_instr = Cast<HloFusionInstruction>(&instr);
  if (native_emitter_fusion_config.type() ==
      NativeEmitterType::NATIVE_EMITTER_TYPE_LOOP) {
    fusion_instr->set_fusion_kind(HloInstruction::FusionKind::kLoop);
  } else if (native_emitter_fusion_config.type() !=
             NativeEmitterType::NATIVE_EMITTER_TYPE_INVALID) {
    fusion_instr->set_fusion_kind(HloInstruction::FusionKind::kInput);
  }
  ASSIGN_OR_RETURN(GpuBackendConfig gpu_backend_config,
                   instr.backend_config<GpuBackendConfig>());
  *gpu_backend_config.mutable_native_emitter_backend_config() =
      native_emitter_fusion_config;
  RETURN_IF_ERROR(fusion_instr->set_backend_config(gpu_backend_config));
  return absl::OkStatus();
}

}  // namespace xla::gpu

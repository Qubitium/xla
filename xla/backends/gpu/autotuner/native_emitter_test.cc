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

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/substitute.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/util/proto/proto_matchers.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::testing::UnorderedElementsAre;
using ::tsl::proto_testing::EqualsProto;

const char kReductionFusionHlo[] = R"(
HloModule m

%func (lhs: f32[], rhs: f32[]) -> f32[] {
  %rhs = f32[] parameter(1)
  %lhs = f32[] parameter(0)
  ROOT %sum = f32[] add(%lhs, %rhs)
}

%fused_reduce.clone (param_0: f32[32,4096,2048]) -> f32[32,2048] {
  %param_0 = f32[32,4096,2048]{2,1,0} parameter(0)
  %c0 = f32[] constant(0)
  ROOT %reduce = f32[32,2048]{1,0} reduce(%param_0, %c0), dimensions={1},
    to_apply=%func
}

ENTRY %entry_computation (p0: f32[32,4096,2048]) -> f32[32,2048] {
  %p0 = f32[32,4096,2048]{2,1,0} parameter(0)
  ROOT %reduce_fusion = f32[32,2048]{1,0} fusion(%p0), kind=kInput,
    calls=%fused_reduce.clone
})";

const char kDeepSeekV41RmsNormHlo[] = R"(
HloModule deepseek_v41_rmsnorm

%add (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %sum = f32[] add(%lhs, %rhs)
}

%rmsnorm (hidden: bf16[1,1,5120], weight: bf16[5120]) -> bf16[1,1,5120] {
  %hidden = bf16[1,1,5120]{2,1,0} parameter(0)
  %weight = bf16[5120]{0} parameter(1)
  %hidden_f32 = f32[1,1,5120]{2,1,0} convert(%hidden)
  %square = f32[1,1,5120]{2,1,0} multiply(%hidden_f32, %hidden_f32)
  %zero = f32[] constant(0)
  %sum = f32[1,1]{1,0} reduce(%square, %zero), dimensions={2}, to_apply=%add
  %mean_scale = f32[] constant(0.0001953125)
  %mean_scale_b = f32[1,1]{1,0} broadcast(%mean_scale), dimensions={}
  %mean = f32[1,1]{1,0} multiply(%sum, %mean_scale_b)
  %eps = f32[] constant(1e-20)
  %eps_b = f32[1,1]{1,0} broadcast(%eps), dimensions={}
  %variance_eps = f32[1,1]{1,0} add(%mean, %eps_b)
  %inv_rms = f32[1,1]{1,0} rsqrt(%variance_eps)
  %inv_rms_b = f32[1,1,5120]{2,1,0} broadcast(%inv_rms), dimensions={0,1}
  %normalized = f32[1,1,5120]{2,1,0} multiply(%hidden_f32, %inv_rms_b)
  %weight_f32 = f32[5120]{0} convert(%weight)
  %weight_b = f32[1,1,5120]{2,1,0} broadcast(%weight_f32), dimensions={2}
  %weighted = f32[1,1,5120]{2,1,0} multiply(%normalized, %weight_b)
  ROOT %out = bf16[1,1,5120]{2,1,0} convert(%weighted)
}

ENTRY %entry_computation (hidden: bf16[1,1,5120], weight: bf16[5120]) -> bf16[1,1,5120] {
  %hidden = bf16[1,1,5120]{2,1,0} parameter(0)
  %weight = bf16[5120]{0} parameter(1)
  ROOT %fusion = bf16[1,1,5120]{2,1,0} fusion(%hidden, %weight),
    kind=kInput, calls=%rmsnorm
})";

const char kPlain5120ReductionHlo[] = R"(
HloModule plain_5120_reduce

%add (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %sum = f32[] add(%lhs, %rhs)
}

%plain_reduce (hidden: bf16[1,1,5120]) -> f32[1,1] {
  %hidden = bf16[1,1,5120]{2,1,0} parameter(0)
  %hidden_f32 = f32[1,1,5120]{2,1,0} convert(%hidden)
  %zero = f32[] constant(0)
  ROOT %sum = f32[1,1]{1,0} reduce(%hidden_f32, %zero),
    dimensions={2}, to_apply=%add
}

ENTRY %entry_computation (hidden: bf16[1,1,5120]) -> f32[1,1] {
  %hidden = bf16[1,1,5120]{2,1,0} parameter(0)
  ROOT %fusion = f32[1,1]{1,0} fusion(%hidden),
    kind=kInput, calls=%plain_reduce
})";

const char kSquared5120ReductionHlo[] = R"(
HloModule squared_5120_reduce

%add (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %sum = f32[] add(%lhs, %rhs)
}

%squared_reduce (hidden: bf16[1,1,5120]) -> f32[1,1] {
  %hidden = bf16[1,1,5120]{2,1,0} parameter(0)
  %hidden_f32 = f32[1,1,5120]{2,1,0} convert(%hidden)
  %square = f32[1,1,5120]{2,1,0} multiply(%hidden_f32, %hidden_f32)
  %zero = f32[] constant(0)
  ROOT %sum = f32[1,1]{1,0} reduce(%square, %zero),
    dimensions={2}, to_apply=%add
}

ENTRY %entry_computation (hidden: bf16[1,1,5120]) -> f32[1,1] {
  %hidden = bf16[1,1,5120]{2,1,0} parameter(0)
  ROOT %fusion = f32[1,1]{1,0} fusion(%hidden),
    kind=kInput, calls=%squared_reduce
})";

const char kTupleSplitRmsNormHlo[] = R"(
HloModule tuple_split_rmsnorm

%add (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %sum = f32[] add(%lhs, %rhs)
}

%square_sum (hidden: bf16[1,1,5120]) -> (f32[1,1], bf16[1,1,5120]) {
  %hidden = bf16[1,1,5120]{2,1,0} parameter(0)
  %hidden_f32 = f32[1,1,5120]{2,1,0} convert(%hidden)
  %square = f32[1,1,5120]{2,1,0} multiply(%hidden_f32, %hidden_f32)
  %zero = f32[] constant(0)
  %sum = f32[1,1]{1,0} reduce(%square, %zero), dimensions={2}, to_apply=%add
  ROOT %tuple = (f32[1,1]{1,0}, bf16[1,1,5120]{2,1,0}) tuple(%sum, %hidden)
}

%normalize (hidden: bf16[1,1,5120], sum: f32[1,1]) -> f32[1,1,5120] {
  %hidden = bf16[1,1,5120]{2,1,0} parameter(0)
  %sum = f32[1,1]{1,0} parameter(1)
  %scale = f32[] constant(0.0001953125)
  %scale_b = f32[1,1]{1,0} broadcast(%scale), dimensions={}
  %mean = f32[1,1]{1,0} multiply(%sum, %scale_b)
  %eps = f32[] constant(1e-20)
  %eps_b = f32[1,1]{1,0} broadcast(%eps), dimensions={}
  %variance_eps = f32[1,1]{1,0} add(%mean, %eps_b)
  %inv_rms = f32[1,1]{1,0} rsqrt(%variance_eps)
  %inv_rms_b = f32[1,1,5120]{2,1,0} broadcast(%inv_rms), dimensions={0,1}
  %hidden_f32 = f32[1,1,5120]{2,1,0} convert(%hidden)
  ROOT %normalized = f32[1,1,5120]{2,1,0} multiply(%hidden_f32, %inv_rms_b)
}

ENTRY %entry_computation (hidden: bf16[1,1,5120]) -> f32[1,1,5120] {
  %hidden = bf16[1,1,5120]{2,1,0} parameter(0)
  %reduce_fusion = (f32[1,1]{1,0}, bf16[1,1,5120]{2,1,0}) fusion(%hidden),
    kind=kInput, calls=%square_sum
  %sum = f32[1,1]{1,0} get-tuple-element(%reduce_fusion), index=0
  ROOT %norm_fusion = f32[1,1,5120]{2,1,0} fusion(%hidden, %sum),
    kind=kLoop, calls=%normalize
})";

const char kL2NormHlo[] = R"(
HloModule l2_norm

%add (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %sum = f32[] add(%lhs, %rhs)
}

%normalize (hidden: bf16[1,1,5120]) -> f32[1,1,5120] {
  %hidden = bf16[1,1,5120]{2,1,0} parameter(0)
  %hidden_f32 = f32[1,1,5120]{2,1,0} convert(%hidden)
  %square = f32[1,1,5120]{2,1,0} multiply(%hidden_f32, %hidden_f32)
  %zero = f32[] constant(0)
  %sum = f32[1,1]{1,0} reduce(%square, %zero), dimensions={2}, to_apply=%add
  %eps = f32[] constant(1e-20)
  %eps_b = f32[1,1]{1,0} broadcast(%eps), dimensions={}
  %sum_eps = f32[1,1]{1,0} add(%sum, %eps_b)
  %inv_norm = f32[1,1]{1,0} rsqrt(%sum_eps)
  %inv_norm_b = f32[1,1,5120]{2,1,0} broadcast(%inv_norm), dimensions={0,1}
  ROOT %normalized = f32[1,1,5120]{2,1,0} multiply(%hidden_f32, %inv_norm_b)
}

ENTRY %entry_computation (hidden: bf16[1,1,5120]) -> f32[1,1,5120] {
  %hidden = bf16[1,1,5120]{2,1,0} parameter(0)
  ROOT %norm_fusion = f32[1,1,5120]{2,1,0} fusion(%hidden),
    kind=kInput, calls=%normalize
})";

const char kCenteredVarianceHlo[] = R"(
HloModule centered_variance

%add (lhs: f32[], rhs: f32[]) -> f32[] {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %sum = f32[] add(%lhs, %rhs)
}

%normalize (hidden: f32[1,1,5120], center: f32[1,1]) -> f32[1,1,5120] {
  %hidden = f32[1,1,5120]{2,1,0} parameter(0)
  %center = f32[1,1]{1,0} parameter(1)
  %center_b = f32[1,1,5120]{2,1,0} broadcast(%center), dimensions={0,1}
  %centered = f32[1,1,5120]{2,1,0} subtract(%hidden, %center_b)
  %square = f32[1,1,5120]{2,1,0} multiply(%centered, %centered)
  %zero = f32[] constant(0)
  %sum = f32[1,1]{1,0} reduce(%square, %zero), dimensions={2}, to_apply=%add
  %scale = f32[] constant(0.0001953125)
  %scale_b = f32[1,1]{1,0} broadcast(%scale), dimensions={}
  %variance = f32[1,1]{1,0} multiply(%sum, %scale_b)
  %eps = f32[] constant(1e-5)
  %eps_b = f32[1,1]{1,0} broadcast(%eps), dimensions={}
  %variance_eps = f32[1,1]{1,0} add(%variance, %eps_b)
  %inv_std = f32[1,1]{1,0} rsqrt(%variance_eps)
  %inv_std_b = f32[1,1,5120]{2,1,0} broadcast(%inv_std), dimensions={0,1}
  ROOT %normalized = f32[1,1,5120]{2,1,0} multiply(%centered, %inv_std_b)
}

ENTRY %entry_computation (hidden: f32[1,1,5120], center: f32[1,1]) -> f32[1,1,5120] {
  %hidden = f32[1,1,5120]{2,1,0} parameter(0)
  %center = f32[1,1]{1,0} parameter(1)
  ROOT %norm_fusion = f32[1,1,5120]{2,1,0} fusion(%hidden, %center),
    kind=kInput, calls=%normalize
})";

const char kAddKernelHlo[] = R"(
HloModule m

%fused_add (p0: f32[$0], p1: f32[$0]) -> f32[$0] {
  %p0 = f32[$0]{1,0} parameter(0)
  %p1 = f32[$0]{1,0} parameter(1)
  ROOT %add = f32[$0]{1,0} add(%p0, %p1)
}

ENTRY %entry_computation (p0: f32[$0], p1: f32[$0]) -> f32[$0] {
  %p0 = f32[$0]{1,0} parameter(0)
  %p1 = f32[$0]{1,0} parameter(1)
  ROOT %loop_fusion = f32[$0]{1,0} fusion(%p0, %p1), kind=kLoop,
    calls=%fused_add
})";

const char kCustomFusionHlo[] = R"(
HloModule m

%fused_add_and_sub (p0: f32[32,16], p1: f32[32,16]) -> (f32[32,16], f32[32,16]) {
  %p0 = f32[32,16]{1,0} parameter(0)
  %p1 = f32[32,16]{1,0} parameter(1)
  %add = f32[32,16]{1,0} add(%p0, %p1)
  %sub = f32[32,16]{1,0} subtract(%p0, %p1)
  ROOT %tuple = (f32[32,16]{1,0}, f32[32,16]{1,0}) tuple(%add, %sub)
}

ENTRY %entry_computation (p0: f32[32,16], p1: f32[32,16]) -> (f32[32,16], f32[32,16]) {
  %p0 = f32[32,16]{1,0} parameter(0)
  %p1 = f32[32,16]{1,0} parameter(1)
  ROOT %reduce_fusion = (f32[32,16]{1,0}, f32[32,16]{1,0}) fusion(%p0, %p1), kind=kCustom,
    calls=%fused_add_and_sub,
    backend_config={ "fusion_backend_config": {
      "kind":"__triton",
      "block_level_fusion_config":{
        "num_warps":"1","output_tiles":[{"sizes":["1","4"]}],
        "num_ctas":1,"num_stages":1,"is_tma_allowed":false
      }
    }}
})";

class NativeEmitterBackendTest : public HloHardwareIndependentTestBase {
 protected:
  NativeEmitterBackendTest()
      : platform_(PlatformUtil::GetDefaultPlatform().value()),
        stream_executor_(platform_->ExecutorForDevice(0).value()),
        target_config_(stream_executor_),
        compiler_(Compiler::GetForPlatform(platform_->id()).value()),
        backend_(&debug_options_, compiler_.get(), &target_config_) {
    debug_options_.set_xla_gpu_libnvjitlink_mode(
        xla::DebugOptions::LIB_NV_JIT_LINK_MODE_DISABLED);
  }

  DebugOptions debug_options_;
  se::Platform* platform_;
  se::StreamExecutor* stream_executor_;
  Compiler::GpuTargetConfig target_config_;
  std::unique_ptr<Compiler> compiler_;
  NativeEmitterBackend backend_;
};

TEST_F(NativeEmitterBackendTest, GetDefaultConfig) {
  ASSERT_OK_AND_ASSIGN(auto reduction_module,
                       ParseAndReturnVerifiedModule(kReductionFusionHlo));
  auto fusion = reduction_module->entry_computation()->root_instruction();
  // Call GetDefaultConfig on the fusion instruction.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                       backend_.GetDefaultConfig(*(fusion)));
  // Verify the returned config is a native emitter config.
  ASSERT_TRUE(config->has_native_emitter());
  NativeEmitterBackendConfig native_emitter_config = config->native_emitter();
}

TEST_F(NativeEmitterBackendTest, GetSupportedConfigs) {
  ASSERT_OK_AND_ASSIGN(auto reduction_module,
                       ParseAndReturnVerifiedModule(kReductionFusionHlo));
  auto fusion = reduction_module->entry_computation()->root_instruction();
  // Call GetSupportedConfigs on the fusion instruction.
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*(fusion)));
  // There should only be a single config for the native emitter backend.
  ASSERT_EQ(configs.size(), 1);
  // Verify the returned config is a native emitter config.
  ASSERT_TRUE(configs[0]->has_native_emitter());
}

TEST_F(NativeEmitterBackendTest, RejectsDeepSeekV41RmsNorm) {
  // DeepSeek V4.1 Flash production RMSNorm contract:
  // BF16 [1,1,5120], hidden_size=5120, rms_norm_eps=1e-20.
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kDeepSeekV41RmsNormHlo));
  auto* fusion = module->entry_computation()->root_instruction();

  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));
  EXPECT_TRUE(configs.empty());
  EXPECT_THAT(backend_.GetDefaultConfig(*fusion),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(NativeEmitterBackendTest, AllowsNonRmsNorm5120Reduction) {
  // Same production-sized BF16 geometry, but no x*x square.
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kPlain5120ReductionHlo));
  auto* fusion = module->entry_computation()->root_instruction();

  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));
  ASSERT_EQ(configs.size(), 1);
  EXPECT_TRUE(configs[0]->has_native_emitter());
}

TEST_F(NativeEmitterBackendTest, AllowsNonRmsNormSquareReduction) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kSquared5120ReductionHlo));
  auto* fusion = module->entry_computation()->root_instruction();

  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));
  ASSERT_EQ(configs.size(), 1);
  EXPECT_TRUE(configs[0]->has_native_emitter());
  ASSERT_OK_AND_ASSIGN(auto default_config, backend_.GetDefaultConfig(*fusion));
  EXPECT_TRUE(default_config->has_native_emitter());
}

TEST_F(NativeEmitterBackendTest, RejectsTupleSplitRmsNormReduction) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kTupleSplitRmsNormHlo));
  auto* fusion =
      module->entry_computation()->GetInstructionWithName("reduce_fusion");
  ASSERT_NE(fusion, nullptr);
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));
  EXPECT_TRUE(configs.empty());
  EXPECT_THAT(backend_.GetDefaultConfig(*fusion),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(NativeEmitterBackendTest, AllowsL2Norm) {
  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kL2NormHlo));
  auto* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));
  ASSERT_EQ(configs.size(), 1);
  EXPECT_TRUE(configs[0]->has_native_emitter());
}

TEST_F(NativeEmitterBackendTest, AllowsCenteredVariance) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kCenteredVarianceHlo));
  auto* fusion = module->entry_computation()->root_instruction();
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*fusion));
  ASSERT_EQ(configs.size(), 1);
  EXPECT_TRUE(configs[0]->has_native_emitter());
}

TEST_F(NativeEmitterBackendTest, GetDefaultConfigForLoopFusion) {
  debug_options_.set_xla_gpu_native_emitter_tune_unroll_factor_for_loops(true);
  ASSERT_OK_AND_ASSIGN(
      auto loop_module,
      ParseAndReturnVerifiedModule(absl::Substitute(kAddKernelHlo, "32,16")));
  auto fusion = loop_module->entry_computation()->root_instruction();
  // Call GetDefaultConfig on the fusion instruction.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                       backend_.GetDefaultConfig(*(fusion)));
  // Verify the returned config is a native emitter config.
  ASSERT_TRUE(config->has_native_emitter());
  EXPECT_THAT(
      config->native_emitter(), EqualsProto(R"pb(type: NATIVE_EMITTER_TYPE_LOOP
                                                 unroll_factor: 1)pb"));
}

TEST_F(NativeEmitterBackendTest, GetSupportedConfigsForLoopFusion) {
  debug_options_.set_xla_gpu_native_emitter_tune_unroll_factor_for_loops(true);
  ASSERT_OK_AND_ASSIGN(auto loop_module,
                       ParseAndReturnVerifiedModule(
                           absl::Substitute(kAddKernelHlo, "1024,4096")));
  auto fusion = loop_module->entry_computation()->root_instruction();
  // Call GetSupportedConfigs on the fusion instruction.
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*(fusion)));
  // Verify the returned configs.
  std::vector<NativeEmitterBackendConfig> native_configs;
  for (const auto& config : configs) {
    ASSERT_TRUE(config->has_native_emitter());

    native_configs.push_back(config->native_emitter());
  }
  HloFusionAnalysis analysis =
      HloFusionAnalysis::Create(*fusion, target_config_.device_description);
  int64_t max_unroll_factor = MaxUnrollFactor(&analysis);
  ASSERT_GT(max_unroll_factor, 1);
  EXPECT_THAT(
      native_configs,
      UnorderedElementsAre(
          EqualsProto(absl::Substitute(R"pb(type: NATIVE_EMITTER_TYPE_LOOP
                                            unroll_factor: $0)pb",
                                       max_unroll_factor / 2)),
          EqualsProto(absl::Substitute(R"pb(type: NATIVE_EMITTER_TYPE_LOOP
                                            unroll_factor: $0)pb",
                                       max_unroll_factor))));
}

TEST_F(NativeEmitterBackendTest,
       GetSupportedConfigsDoesNotSupportKCustomFusions) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kCustomFusionHlo));
  auto fusion_instruction = module->entry_computation()->root_instruction();
  // Call GetSupportedConfigs on the fusion instruction.
  ASSERT_OK_AND_ASSIGN(std::vector<std::unique_ptr<BackendConfig>> configs,
                       backend_.GetSupportedConfigs(*(fusion_instruction)));
  // GetSupportedConfigs should return an empty vector as it doesn't support the
  // fusion.
  ASSERT_TRUE(configs.empty());
}

TEST_F(NativeEmitterBackendTest, ApplyConfig) {
  ASSERT_OK_AND_ASSIGN(auto reduction_module,
                       ParseAndReturnVerifiedModule(kReductionFusionHlo));
  auto fusion = reduction_module->entry_computation()->root_instruction();
  // Call ApplyConfig on the fusion instruction.
  NativeEmitterBackendConfig native_emitter_config;
  BackendConfig config;
  *config.mutable_native_emitter() = native_emitter_config;
  ASSERT_THAT(backend_.ApplyConfig(*(fusion), config), absl_testing::IsOk());
  // Verify the fusion instruction is now a kInput fusion.
  ASSERT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kInput);
  // Verify the fusion instruction has a native emitter backend config.
  ASSERT_TRUE(fusion->has_backend_config());
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig gpu_backend_config,
                       fusion->backend_config<GpuBackendConfig>());
  ASSERT_TRUE(gpu_backend_config.has_native_emitter_backend_config());
}

TEST_F(NativeEmitterBackendTest, ApplyConfigFailsForUnsupportedConfig) {
  ASSERT_OK_AND_ASSIGN(auto reduction_module,
                       ParseAndReturnVerifiedModule(kReductionFusionHlo));
  auto fusion = reduction_module->entry_computation()->root_instruction();
  BlockLevelFusionConfig block_level_fusion_config;
  BackendConfig config;
  *config.mutable_block_level() = block_level_fusion_config;
  ASSERT_THAT(backend_.ApplyConfig(*(fusion), config),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(NativeEmitterBackendTest, CompileForDefaultConfig) {
  ASSERT_OK_AND_ASSIGN(auto reduction_module,
                       ParseAndReturnVerifiedModule(kReductionFusionHlo));
  auto fusion = reduction_module->entry_computation()->root_instruction();
  // Call GetDefaultConfig on the fusion instruction.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                       backend_.GetDefaultConfig(*(fusion)));
  // Attempt to compile the fusion using the retrieved backend config.
  auto maybe_executable = backend_.Compile(*fusion, *config);
  // Verify that compilation succeeded and returned a valid executable.
  EXPECT_THAT(maybe_executable, absl_testing::IsOk());
}

class MockCompiler : public Compiler {
 public:
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<Executable>>, RunBackend,
              (std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
               const CompileOptions& options),
              (override));
  MOCK_METHOD(se::Platform::Id, PlatformId, (), (const, override));
  MOCK_METHOD(absl::StatusOr<std::unique_ptr<HloModule>>, RunHloPasses,
              (std::unique_ptr<HloModule> module, se::StreamExecutor* executor,
               const CompileOptions& options),
              (override));
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<Executable>>>, Compile,
              (std::unique_ptr<HloModule> hlo_module,
               std::vector<se::StreamExecutor*> stream_execs,
               const CompileOptions& options),
              (override));
  MOCK_METHOD(absl::StatusOr<std::vector<std::unique_ptr<CompiledModule>>>,
              CompileAheadOfTime,
              (std::unique_ptr<HloModule> hlo_module,
               const AotCompilationOptions& options),
              (override));
  MOCK_METHOD(HloCostAnalysis::ShapeSizeFunction, ShapeSizeBytesFunction, (),
              (const, override));
};

TEST_F(NativeEmitterBackendTest, CompileSetsIsAutotuningCompilationOption) {
  ASSERT_OK_AND_ASSIGN(auto reduction_module,
                       ParseAndReturnVerifiedModule(kReductionFusionHlo));
  auto fusion = reduction_module->entry_computation()->root_instruction();
  MockCompiler mock_compiler;
  NativeEmitterBackend backend(&debug_options_, &mock_compiler,
                               &target_config_);
  // Call GetDefaultConfig on the fusion instruction.
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<BackendConfig> config,
                       backend.GetDefaultConfig(*(fusion)));
  EXPECT_CALL(
      mock_compiler,
      RunBackend(
          testing::_, testing::_,
          testing::Field(&Compiler::CompileOptions::embed_hlo_module, false)))
      .WillOnce(testing::Return(std::unique_ptr<Executable>()));
  // Attempt to compile the fusion using the retrieved backend config.
  EXPECT_THAT(backend.Compile(*fusion, *config), absl_testing::IsOk());
}

}  // namespace
}  // namespace xla::gpu

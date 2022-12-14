#===============================================================================
# Copyright 2021-2022 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

set(OBJ_LIB ${LIB_NAME}_backend_dnnl_unit_test)

add_library(${OBJ_LIB} OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/test_compile.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_compiled_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fused_op.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_layout_id.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_logical_tensor.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_op_schema.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_partition.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_scratchpad.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_subgraph_pass.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/test_thread_local_cache.cpp
    ${GRAPH_TEST_THREAD}
    )

target_include_directories(${OBJ_LIB}
    PRIVATE
    ${PROJECT_BINARY_DIR}/third_party/oneDNN/include
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN/include
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN/
    ${PROJECT_SOURCE_DIR}/third_party/oneDNN/src/
    )
    
set_property(GLOBAL APPEND PROPERTY ${LIB_NAME}_UNIT_TEST_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)
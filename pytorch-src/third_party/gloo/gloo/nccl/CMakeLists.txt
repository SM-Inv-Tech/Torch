if(USE_CUDA)
    set(GLOO_NCCL_SRCS
      "${CMAKE_CURRENT_SOURCE_DIR}/nccl.cu"
      )

    set(GLOO_NCCL_HDRS
      "${CMAKE_CURRENT_SOURCE_DIR}/nccl.h"
      )

    list(APPEND GLOO_CUDA_SRCS ${GLOO_NCCL_SRCS})
    list(APPEND GLOO_CUDA_HDRS ${GLOO_NCCL_HDRS})
    set(GLOO_CUDA_SRCS ${GLOO_CUDA_SRCS} PARENT_SCOPE)
    set(GLOO_CUDA_HDRS ${GLOO_CUDA_HDRS} PARENT_SCOPE)
endif()

if(USE_ROCM)
    set(GLOO_NCCL_SRCS
      "${HIPIFY_OUTPUT_DIR}/nccl/nccl.hip"
      )

    set(GLOO_NCCL_HDRS
      "${HIPIFY_OUTPUT_DIR}/nccl/nccl.h"
      )

    list(APPEND GLOO_HIP_SRCS ${GLOO_NCCL_SRCS})
    list(APPEND GLOO_HIP_HDRS ${GLOO_NCCL_HDRS})
    set(GLOO_HIP_SRCS ${GLOO_HIP_SRCS} PARENT_SCOPE)
    set(GLOO_HIP_HDRS ${GLOO_HIP_HDRS} PARENT_SCOPE)
endif()

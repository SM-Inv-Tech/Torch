ADD_EXECUTABLE(cpp_tutorials_8
        main.cpp
        ConvReluBnImpl.cpp
        PlainCNNImpl.cpp)
TARGET_LINK_LIBRARIES(cpp_tutorials_8 ${TORCH_LIBRARIES})
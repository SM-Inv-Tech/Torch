ADD_EXECUTABLE(cpp_tutorials_11
        main.cpp
        VGGImpl.cpp)
TARGET_LINK_LIBRARIES(cpp_tutorials_11
        ${TORCH_LIBRARIES}
        ${OpenCV_LIBRARIES}
        stdc++fs)
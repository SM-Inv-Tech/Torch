ADD_EXECUTABLE(cpp_tutorials_10
        main.cpp
        ImageDataset.cpp)
TARGET_LINK_LIBRARIES(cpp_tutorials_10 ${TORCH_LIBRARIES} ${OpenCV_LIBRARIES} stdc++fs)
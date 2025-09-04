set(PROJECT_SOURCES
    "main.cpp"

    "DL/Model.cpp"
)

add_executable(PROJECT_EXE ${PROJECT_SOURCES})

set_target_properties(PROJECT_EXE PROPERTIES
    OUTPUT_NAME "main"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

target_include_directories(PROJECT_EXE PRIVATE "${CMAKE_SOURCE_DIR}/inc")

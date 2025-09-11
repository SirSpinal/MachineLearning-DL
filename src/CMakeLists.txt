set(PROJECT_SOURCES
    "main.cpp"

    "DL/ActFunc/ActFunc.cpp"
    "DL/Batch.cpp"
    "DL/Cost.cpp"
    "DL/InitFunc/ActInit.cpp"
    "DL/InitFunc/Functions.cpp"
    "DL/Metaparams.cpp"
    "DL/MLModel/BasicModel.cpp"
    "DL/ParamMatrix.cpp"
    "DL/Propagation.cpp"
    
    "util/PrintModel.cpp"
)

add_executable(PROJECT_EXE ${PROJECT_SOURCES})

set_target_properties(PROJECT_EXE PROPERTIES
    OUTPUT_NAME "main"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/bin"
)

target_include_directories(PROJECT_EXE PRIVATE "${CMAKE_SOURCE_DIR}/inc")

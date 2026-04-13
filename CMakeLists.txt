#
# PUSCH IQ Dataset Capture Plugin
#
# Builds libreceiver_pusch_capture.so — loaded via:
#   --loader.receiver.shlibversion _pusch_capture
#

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})

add_library(receiver_pusch_capture MODULE
    src/nr_pusch_capture.c
)
target_compile_options(receiver_pusch_capture PRIVATE
    -Wno-deprecated-declarations
    -Wno-error=unused-variable
    -Wno-error=unused-result
)
target_link_libraries(receiver_pusch_capture PRIVATE pthread)

add_dependencies(nr-softmodem receiver_pusch_capture)
add_dependencies(nr-uesoftmodem receiver_pusch_capture)
add_dependencies(receiver_pusch_capture generate_T)

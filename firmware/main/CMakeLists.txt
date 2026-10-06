set(model_file "${CMAKE_CURRENT_LIST_DIR}/model/drone_nav_int8.tflite")
if(NOT EXISTS "${model_file}")
    message(FATAL_ERROR
        "Missing ${model_file}. Train/export the model first: "
        "python -m drone_nav.train --data dataset --output artifacts")
endif()

idf_component_register(
    SRCS "app_main.cc" "camera_input.cc" "decision_filter.cc"
    INCLUDE_DIRS "."
    PRIV_REQUIRES esp_timer driver
    EMBED_FILES "model/drone_nav_int8.tflite"
)

target_compile_options(${COMPONENT_LIB} PRIVATE -Wall -Wextra -Werror)

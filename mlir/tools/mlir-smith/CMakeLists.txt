get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

set(LIBS ${dialect_libs} MLIRDialect MLIRIR MLIRSmithLib)

add_mlir_tool(mlir-smith mlir-smith.cpp DEPENDS ${LIBS})

target_link_libraries(mlir-smith PRIVATE ${LIBS})
llvm_update_compile_flags(mlir-smith)

mlir_check_all_link_libraries(mlir-smith)

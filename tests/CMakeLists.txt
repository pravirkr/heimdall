add_executable(testdedisp 
	testdedisp.cpp 
	gasdev.cpp 
	ran1.cpp)

target_include_directories(testdedisp PRIVATE ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(testdedisp ${DEDISP} m)

install(TARGETS testdedisp
	DESTINATION ${PROJECT_SOURCE_DIR}/tests)
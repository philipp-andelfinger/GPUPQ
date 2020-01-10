
# load user-defined options from a separate file that's not under version control,
# so options like libraries and source files can be switched via version control
ifeq ("$(wildcard Makefile.options)","")
$(error Makefile.options does not exist. Please copy Makefile.options.default and fit it to your needs.)
else
include Makefile.options
endif

# ---------- C++ Options ----------

CC       = g++
CFLAGS   = $(OPT) $(DBG) $(WARN) $(ETC) #-I/usr/include/sfml
CLDFLAGS = -L$(CUDAROOT)/lib64 -lcudart -lcudadevrt -pthread #-lsfml-graphics -lsfml-window -lsfml-system

# ---------- CUDA Options ----------

NVCC      = $(CUDAROOT)/bin/nvcc
# expt-extended-lambda needed for mordengpu device-tagged lambdas
CUFLAGS   = $(OPT) $(DBG) $(CUDBG) $(ETC) $(CUARCH) \
						-rdc=true \
						-I$(ROOT_DIR)/lib/cub-1.4.1 \
						-I$(ROOT_DIR)/lib/googletest-1.7.0/include \
						-I$(ROOT_DIR)/lib/moderngpu/src \
						--expt-extended-lambda -Xptxas -dlcm=ca
CULDFLAGS = $(CUARCH)

# ---------- Test Options ----------

TEST_LDFLAGS = -L../lib/googletest-1.7.0/build \
							 -lgtest -lgtest_main -lpthread
TEST_CLDFLAGS = -Wl,-R,$(ROOT_DIR)/lib/googletest-1.7.0/lib

# ----------------------------------

SOURCES = main.cc des.cu sort.cu \
					local_array_queue.cu \
					heap_queue.cu \
					global_array_queue.cu \
					local_heap_queue.cu \
					local_splay_queue.cu \
					phold.cu \
					a_star.cu \
					a_star/check_cuda_errors.cpp #\
          #gl_interop.cu



# @ surpresses default command echo from Make
#CMD_PREFIX = @

CSOURCES := $(filter %.cc, $(SOURCES))
CUSOURCES := $(filter %.cu, $(SOURCES))

COBJECTS := ${CSOURCES:%.cc=$(BUILD_PATH)/%.o}
CUOBJECTS := ${CUSOURCES:%.cu=$(BUILD_PATH)/%.o}

DEPS = ${COBJECTS:.o=.d} ${CUOBJECTS:.o=.d} ${BUILD_PATH}/test_hq.d ${BUILD_PATH}/test_laq.d ${BUILD_PATH}/test_gaq.d

# ----------------------------------

TEST1_SOURCES = test_hq.cu \
								phold.cu \
								des.cu \
								heap_queue.cu

TEST1_CSOURCES := $(filter %.cc, $(TEST1_SOURCES))
TEST1_CUSOURCES := $(filter %.cu, $(TEST1_SOURCES))

TEST1_COBJECTS := ${TEST1_CSOURCES:%.cc=$(BUILD_PATH)/%.o}
TEST1_CUOBJECTS := ${TEST1_CUSOURCES:%.cu=$(BUILD_PATH)/%.o}

# ----------------------------------

TEST2_SOURCES = test_laq.cu \
		  				 phold.cu \
		  				 des.cu \
							 local_array_queue.cu \
							 sort.cu

TEST2_CSOURCES := $(filter %.cc, $(TEST2_SOURCES))
TEST2_CUSOURCES := $(filter %.cu, $(TEST2_SOURCES))

TEST2_COBJECTS := ${TEST2_CSOURCES:%.cc=$(BUILD_PATH)/%.o}
TEST2_CUOBJECTS := ${TEST2_CUSOURCES:%.cu=$(BUILD_PATH)/%.o}

# ----------------------------------

TEST3_SOURCES = test_gaq.cu \
		  				 phold.cu \
		  				 des.cu \
							 global_array_queue.cu \
							 sort.cu

TEST3_CSOURCES := $(filter %.cc, $(TEST3_SOURCES))
TEST3_CUSOURCES := $(filter %.cu, $(TEST3_SOURCES))

TEST3_COBJECTS := ${TEST3_CSOURCES:%.cc=$(BUILD_PATH)/%.o}
TEST3_CUOBJECTS := ${TEST3_CUSOURCES:%.cu=$(BUILD_PATH)/%.o}

# ----------------------------------

CCCOLOR   = "\033[34m"
NVCCCOLOR = "\033[36m"
LINKCOLOR = "\033[34;1m"
SRCCOLOR  = "\033[33m"
BINCOLOR  = "\033[37;1m"
ENDCOLOR  = "\033[0m"

PRINT_CC   = @printf '  %b %b\n' $(CCCOLOR)CC$(ENDCOLOR) $(SRCCOLOR)$@$(ENDCOLOR) 1>&2;
PRINT_NVCC = @printf '  %b %b\n' $(NVCCCOLOR)NVCC$(ENDCOLOR) $(SRCCOLOR)$@$(ENDCOLOR) 1>&2;
PRINT_LINK = @printf '  %b %b\n' $(LINKCOLOR)LINK$(ENDCOLOR) $(BINCOLOR)$@$(ENDCOLOR) 1>&2;

# ----------------------------------

.SECONDARY: $(COBJECTS) $(CUOBJECTS)

all: $(BIN_PATH)/$(TARGET)

test-hq: $(BIN_PATH)/test-hq

test-laq: $(BIN_PATH)/test-laq

test-gaq: $(BIN_PATH)/test-gaq

$(BIN_PATH)/%: dirs $(COBJECTS) $(CUOBJECTS)
	$(PRINT_LINK)
	$(CMD_PREFIX)$(NVCC) -dlink -o $(BUILD_PATH)/cuda_link.o $(CUOBJECTS) $(CULDFLAGS)
	$(CMD_PREFIX)$(CC) $(CUOBJECTS) $(BUILD_PATH)/cuda_link.o $(COBJECTS) -MP -MMD -o $@ -o $@ $(CLDFLAGS)

$(BUILD_PATH)/%.o: %.cc
	$(PRINT_CC)
	$(CMD_PREFIX)$(CC) $(CFLAGS) -MP -MMD -c $< -o $@

$(BUILD_PATH)/%.o: %.cu
	$(PRINT_NVCC)
	$(CMD_PREFIX)$(NVCC) $(CUFLAGS) -M $< -o ${@:.o=.d} -odir $(@D)
	$(CMD_PREFIX)$(NVCC) $(CUFLAGS) -c $< -o $@

$(BIN_PATH)/test-hq: dirs $(TEST1_COBJECTS) $(TEST1_CUOBJECTS)
	$(PRINT_LINK)
	$(CMD_PREFIX)$(NVCC) -dlink -o $(BUILD_PATH)/cuda_link_test.o $(TEST1_CUOBJECTS) $(CULDFLAGS) $(TEST_LDFLAGS)
	$(CMD_PREFIX)$(CC) $(TEST1_CUOBJECTS) $(BUILD_PATH)/cuda_link_test.o $(TEST1_COBJECTS) -MP -MMD -o $@ -o $@ $(CLDFLAGS) $(TEST_LDFLAGS) $(TEST_CLDFLAGS)

$(BIN_PATH)/test-laq: dirs $(TEST2_COBJECTS) $(TEST2_CUOBJECTS)
	$(PRINT_LINK)
	$(CMD_PREFIX)$(NVCC) -dlink -o $(BUILD_PATH)/cuda_link_test.o $(TEST2_CUOBJECTS) $(CULDFLAGS) $(TEST_LDFLAGS)
	$(CMD_PREFIX)$(CC) $(TEST2_CUOBJECTS) $(BUILD_PATH)/cuda_link_test.o $(TEST2_COBJECTS) -MP -MMD -o $@ -o $@ $(CLDFLAGS) $(TEST_LDFLAGS) $(TEST_CLDFLAGS)

$(BIN_PATH)/test-gaq: dirs $(TEST3_COBJECTS) $(TEST3_CUOBJECTS)
	$(PRINT_LINK)
	$(CMD_PREFIX)$(NVCC) -dlink -o $(BUILD_PATH)/cuda_link_test.o $(TEST3_CUOBJECTS) $(CULDFLAGS) $(TEST_LDFLAGS)
	$(CMD_PREFIX)$(CC) $(TEST3_CUOBJECTS) $(BUILD_PATH)/cuda_link_test.o $(TEST3_COBJECTS) -MP -MMD -o $@ -o $@ $(CLDFLAGS) $(TEST_LDFLAGS) $(TEST_CLDFLAGS)

.PHONY: dirs
dirs:
	@echo "Creating directories"
	$(CMD_PREFIX)mkdir -p $(dir $(COBJECTS))
	$(CMD_PREFIX)mkdir -p $(dir $(CUOBJECTS))
	$(CMD_PREFIX)mkdir -p $(BUILD_PATH)
	$(CMD_PREFIX)mkdir -p $(BIN_PATH)

.PHONY: clean
clean:
	@echo "Deleting binaries"
	$(CMD_PREFIX)$(RM) $(TARGET)
	@echo "Deleting directories"
	$(CMD_PREFIX)$(RM) -r $(BUILD_PATH)

-include $(DEPS)

# Makefile -- eoq_parallel_decode + eoq_rans_v2 build targets
#
# Targets:
#   make all                Build static libraries
#   make test               Build and run all test programs
#   make test-parallel      Build and run parallel decode tests only
#   make test-rans-v2       Build and run rANS v2 decoder tests only
#   make bench-rans-v2      Build and run rANS v2 with -O3 -march=native
#   make clean              Remove build artifacts
#   make help               Show this help
#
# Variables:
#   CC                   C compiler (default: cc)
#   CFLAGS               Extra compiler flags
#   NUM_THREADS          Override thread count for tests (default: auto)

CC      ?= cc
CFLAGS  ?= -O2
WARN     = -Wall -Wextra -Wpedantic -Wno-unused-parameter
STD      = -std=c99

# POSIX threads
LDFLAGS  = -lpthread -lm

# Path to v1 encoder (needed by rANS v2 tests for round-trip verification)
V1_DIR   = ../llamacpp

# --- Parallel decode library ---

LIB_SRC  = eoq_parallel_decode.c
LIB_OBJ  = eoq_parallel_decode.o
LIB_HDR  = eoq_parallel_decode.h eoq_rans_v2.h
LIB_A    = libeoq_parallel_decode.a

TEST_SRC = test_parallel_decode.c
TEST_BIN = test_parallel_decode

# --- rANS v2 decoder ---

V2_SRC       = eoq_rans_v2.c
V2_HDR       = eoq_rans_v2.h
V1_SRC       = $(V1_DIR)/eoq_rans.c
V1_HDR       = $(V1_DIR)/eoq_ggml.h
V2_TEST_SRC  = test_rans_v2.c
V2_TEST_BIN  = test_rans_v2
V2_BENCH_BIN = test_rans_v2_bench

# Include paths
V2_INCLUDES  = -I. -I$(V1_DIR)

.PHONY: all test test-parallel test-rans-v2 bench-rans-v2 clean help

all: $(LIB_A)

# --- Static library (the .c only, mock-free) ---

$(LIB_OBJ): $(LIB_SRC) $(LIB_HDR)
	$(CC) $(STD) $(WARN) $(CFLAGS) -c $(LIB_SRC) -o $(LIB_OBJ)

$(LIB_A): $(LIB_OBJ)
	ar rcs $(LIB_A) $(LIB_OBJ)

# --- Parallel decode test program ---
#
# The test provides its own mock implementations of all eoq_rans_v2 functions,
# so we compile test_parallel_decode.c together with eoq_parallel_decode.c
# WITHOUT linking against a real rANS library.

$(TEST_BIN): $(TEST_SRC) $(LIB_SRC) $(LIB_HDR)
	$(CC) $(STD) $(WARN) $(CFLAGS) \
		-D_POSIX_C_SOURCE=200112L \
		$(TEST_SRC) $(LIB_SRC) \
		$(LDFLAGS) \
		-o $(TEST_BIN)

test-parallel: $(TEST_BIN)
	@echo ""
	@echo "--- Running parallel decode tests ---"
	@echo ""
	./$(TEST_BIN)

# --- rANS v2 decoder test program ---
#
# Links against BOTH v2 (decoder) and v1 (encoder) to verify wire-format
# compatibility via encode(v1) -> decode(v2) round-trips.

# Note: -DEOQ_NO_V1_COMPAT suppresses the v2 eoq_decode_tensor wrapper
# since the v1 encoder object already provides that symbol.
$(V2_TEST_BIN): $(V2_TEST_SRC) $(V2_SRC) $(V1_SRC) $(V2_HDR) $(V1_HDR)
	$(CC) $(STD) $(WARN) $(CFLAGS) $(V2_INCLUDES) -DEOQ_NO_V1_COMPAT \
		-o $(V2_TEST_BIN) \
		$(V2_TEST_SRC) $(V2_SRC) $(V1_SRC) \
		$(LDFLAGS)

test-rans-v2: $(V2_TEST_BIN)
	@echo ""
	@echo "--- Running rANS v2 decoder tests ---"
	@echo ""
	./$(V2_TEST_BIN)

# --- rANS v2 benchmark build (aggressive optimization) ---

bench-rans-v2: $(V2_TEST_SRC) $(V2_SRC) $(V1_SRC) $(V2_HDR) $(V1_HDR)
	$(CC) $(STD) $(WARN) -O3 -DNDEBUG -march=native $(V2_INCLUDES) -DEOQ_NO_V1_COMPAT \
		-o $(V2_BENCH_BIN) \
		$(V2_TEST_SRC) $(V2_SRC) $(V1_SRC) \
		$(LDFLAGS)
	@echo ""
	@echo "--- Running rANS v2 benchmark ---"
	@echo ""
	./$(V2_BENCH_BIN)

# --- Run all tests ---

test: test-parallel test-rans-v2

clean:
	rm -f $(LIB_OBJ) $(LIB_A) $(TEST_BIN) $(V2_TEST_BIN) $(V2_BENCH_BIN)

help:
	@echo "Targets:"
	@echo "  all             - Build static library (libeoq_parallel_decode.a)"
	@echo "  test            - Build and run all tests"
	@echo "  test-parallel   - Build and run parallel decode tests"
	@echo "  test-rans-v2    - Build and run rANS v2 decoder tests"
	@echo "  bench-rans-v2   - Build + run rANS v2 with -O3 -march=native"
	@echo "  clean           - Remove build artifacts"
	@echo ""
	@echo "Example:"
	@echo "  make test CC=gcc CFLAGS='-O2 -g -fsanitize=thread'"

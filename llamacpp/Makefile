CC = cc
CFLAGS = -O2 -Wall -Wextra -std=c99

test_rans: test_rans.c eoq_rans.c eoq_ggml.h
	$(CC) $(CFLAGS) -o test_rans test_rans.c eoq_rans.c -lm

clean:
	rm -f test_rans

.PHONY: clean

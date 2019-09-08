# Perform out of bounds on indices (0=False, 1=True)
CHECK_INDICES=0

all: masked_grad

masked_grad:
	$(CC) -DMI_CHECK_INDEX=$(CHECK_INDICES) -fPIC -std=c99 -O3 \
		-shared -o masked_gradient.so masked_gradient.c

clean:
	rm -f *.so

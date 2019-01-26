# Perform out of bounds on indices (0=False, 1=True)
CHECK_INDICES=0

masked_grad:
	gcc -DMI_CHECK_INDEX=$(CHECK_INDICES) -fPIC -std=c99 -O3 \
		-shared -o _masked_gradient.so _masked_gradient.c

clean:
	rm -f *.so

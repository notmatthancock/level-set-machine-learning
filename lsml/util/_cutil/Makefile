# Perform out of bounds on indices (0=False, 1=True)
CHECK_INDICES=0

all: masked_grad com_ray

masked_grad:
	$(CC) -DMI_CHECK_INDEX=$(CHECK_INDICES) -fPIC -std=c99 -O3 \
		-shared -o masked_gradient.so masked_gradient.c

com_ray:
	$(CC) -DMI_CHECK_INDEX=$(CHECK_INDICES) -fPIC -std=c99 -O3 \
		-shared -o com_ray_sample.so com_ray_sample.c

clean:
	rm -f *.so

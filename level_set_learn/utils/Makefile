# Uncommenting will print a warning whenever an array is copied.
OPTS=#-DF2PY_REPORT_ON_ARRAY_COPY=1

# Uncommenting DMI_CHECK_INDEX will check all indices in C routines.
CFLAGS="-std=c99"# -DMI_CHECK_INDEX=1"

all: trilinear masked_grad

masked_grad:
	CFLAGS=$(CFLAGS) f2py $(OPTS) _masked_grad.pyf \
	-c _masked_grad.c -m _masked_grad

trilinear:
	CFLAGS=$(CFLAGS) f2py $(OPTS) trilinear.pyf \
	-c trilinear.c -m trilinear

clean:
	rm -f *.mod *.so

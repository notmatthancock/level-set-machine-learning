# Uncommenting will print a warning whenever an array is copied.
OPTS=#-DF2PY_REPORT_ON_ARRAY_COPY=1

# Uncommenting DMI_CHECK_INDEX will check all indices in C routines.
CFLAGS="-std=c99"# -DMI_CHECK_INDEX=1"

all: to_com_samples normal_samples radii_from_mask trilinear masked_grad

to_com_samples:
	CFLAGS=$(CFLAGS) f2py $(OPTS) to_com_samples.pyf \
	-c to_com_samples.c -m to_com_samples

normal_samples:
	CFLAGS=$(CFLAGS) f2py $(OPTS) normal_samples.pyf \
	-c normal_samples.c -m normal_samples

radii_from_mask:
	CFLAGS=$(CFLAGS) f2py $(OPTS) radii_from_mask.pyf \
	-c radii_from_mask.c -m radii_from_mask

masked_grad:
	CFLAGS=$(CFLAGS) f2py $(OPTS) _masked_grad.pyf \
	-c _masked_grad.c -m _masked_grad

trilinear:
	CFLAGS=$(CFLAGS) f2py $(OPTS) trilinear.pyf \
	-c trilinear.c -m trilinear

clean:
	rm -f *.mod *.so

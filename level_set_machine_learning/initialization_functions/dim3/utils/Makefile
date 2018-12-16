# Uncommenting will print a warning whenever an array is copied.
OPTS=#-DF2PY_REPORT_ON_ARRAY_COPY=1

# Uncommenting DMI_CHECK_INDEX will check all indices in C routines.
CFLAGS="-std=c99"# -DMI_CHECK_INDEX=1"

radii_from_mask:
	CFLAGS=$(CFLAGS) f2py $(OPTS) radii_from_mask.pyf \
	-c radii_from_mask.c -m radii_from_mask

clean:
	rm -f *.mod *.so

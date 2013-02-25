COMPILER=mpicxx
GCC=-O3 -ftree-vectorize
ICC=-O3 -fp-model fast -xHost -ipo -no-prec-div

gcc:
	$(COMPILER)  -g main.cpp $(GCC) -o  ../bin/pgbrt
icc:
	$(COMPILER) main.cpp $(ICC) -o -g ../bin/pgbrt

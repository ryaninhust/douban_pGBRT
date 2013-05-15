COMPILER=mpicxx
GCC=-O3 -ftree-vectorize
ICC=-O3 -fp-model fast -xHost -ipo -no-prec-div

gcc:
	$(COMPILER)  -g main.cpp $(GCC) -o  ../bin/ents
icc:
	$(COMPILER) main.cpp $(ICC) -o -g ../bin/pgbrt

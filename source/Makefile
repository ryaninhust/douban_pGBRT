COMPILER=mpicxx
GCC=-O3 -ftree-vectorize -pthread -lmongoclient -lboost_thread-mt -lboost_filesystem -lboost_program_options -lboost_system
ICC=-O3 -fp-model fast -xHost -ipo -no-prec-div

gcc:
	$(COMPILER)  -g main.cpp $(GCC) -o  ../bin/ents
icc:
	$(COMPILER) main.cpp $(ICC) -o -g ../bin/pgbrt

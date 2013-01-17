mpirun -np 4 ../bin/pgbrt  ../../learning_to_challange_dataset/set1.train.txt 473134 700 4 10 0.1 -V ../../learning_to_challange_dataset/set1.valid.txt -v 30 -m > out.log
cat out.log | python ../scripts/crossval.py 5 -r | python ../scripts/compiletest.py test
cat test.dat | ./test > test.pred
python ../scripts/evaluate.py ../../learning_to_challange_dataset/set1.test.txt test.pred

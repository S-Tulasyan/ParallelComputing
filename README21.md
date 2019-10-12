## Modulus based matrix multiplication implementation using Serial and Parallel programming methods
### Compilation and execution instructions

* mm_serial.cpp
	* g++ -o mm_serial -std=c++11 mm_serial.cpp -lm
	* ./mm_serial

* mgs_serial.cpp
	* g++ -o mgs_serial -std=c++11 mgs_serial.cpp -lm
	* ./mgs_serial

* msor_serial.cpp
	* g++ -o msor_serial -std=c++11 msor_serial.cpp -lm
	* ./msor_serial

* mm_parallel.cpp
	* We have tested the code for 4,8,16 and 32 threads.
	* export OMP_NUM_THREADS=(specify no. of threads) 
	* g++ -o mm_p -fopenmp -std=c++11 mm_parallel.cpp -lm
	* ./mm_p

* mgs_parallel.cpp
	* We have tested the code for 4,8,16 and 32 threads.
	* export OMP_NUM_THREADS=(specify no. of threads) 
	* g++ -o mgs_p -fopenmp -std=c++11 mgs_parallel.cpp -lm
	* ./mgs_p

* msor_parallel.cpp
	* We have tested the code for 4,8,16 and 32 threads.
	* export OMP_NUM_THREADS=(specify no. of threads) 
	* g++ -o msor_p -fopenmp -std=c++11 msor_parallel.cpp -lm
	* ./msor_p
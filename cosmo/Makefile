copy:
	cp cpp/portable_math.cpp ./
	cp cpp/portable_math.h ./
	cp f90/*.f90 ./
	cp driver/main.f90 ./

clean:
	rm -f *.h *.cpp *.cu *.cub *.ptx *.f90 *.mod *.o *.out *.a

acc: clean copy
	CC -D__GPUACC__ -h acc -hfp0 -hflex_mp=intolerant -c portable_math.cpp
	ftn -h acc -hfp0 -hflex_mp=intolerant -c hyperbolic.f90
	ftn -h acc -hfp0 -hflex_mp=intolerant -c inversetrigo.f90
	ftn -h acc -hfp0 -hflex_mp=intolerant -c mchpow.f90
	ftn -h acc -hfp0 -hflex_mp=intolerant -c transcendental.f90
	ftn -h acc -hfp0 -hflex_mp=intolerant -c trigo.f90
	ftn -h acc -hfp0 -hflex_mp=intolerant -c main.f90	
	ftn -h acc -hfp0 -hflex_mp=intolerant portable_math.o hyperbolic.o inversetrigo.o mchpow.o transcendental.o trigo.o main.o -o acc.out

noacc: clean copy
	CC  -h noacc -hfp0 -hflex_mp=intolerant -c portable_math.cpp
	ftn -h noacc -hfp0 -hflex_mp=intolerant -c hyperbolic.f90
	ftn -h noacc -hfp0 -hflex_mp=intolerant -c inversetrigo.f90
	ftn -h noacc -hfp0 -hflex_mp=intolerant -c mchpow.f90
	ftn -h noacc -hfp0 -hflex_mp=intolerant -c transcendental.f90
	ftn -h noacc -hfp0 -hflex_mp=intolerant -c trigo.f90
	ftn -h noacc -hfp0 -hflex_mp=intolerant -c main.f90	
	ftn -h noacc -hfp0 -hflex_mp=intolerant portable_math.o hyperbolic.o inversetrigo.o mchpow.o transcendental.o trigo.o main.o -o noacc.out
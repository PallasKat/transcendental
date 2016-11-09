program shadow
  use trigo
  use inversetrigo
  implicit none
  
  integer, parameter :: N = 3, fid=1204, bid=1980, verbose=1
  double precision, parameter :: a = -3.14, b = 3.14
  
  integer :: i, j
  double precision :: x, imgSum, delta
  
  double precision, dimension(N) :: preimg, img
  
  !double precision, dimension(:), allocatable :: preimg, img
  !allocate(img(N))
  !allocate(preimg(N))
  
  if (N-1 .le. 0) then
    delta = abs(b - a)
  else
    delta = abs(b - a)/(N-1)
  end if

  x = a
  do i=1,N
    preimg(i) = x
    x = x + delta
  end do

  if (verbose .ge. 1) then
    print *, "a", a
    print *, "x", x
    print *, "N", N
    print *, "delta", delta
    print *, "range", abs(b - a)
  end if

  if (verbose .ge. 2) then
    print *, "x", x
    print *, "preimg", preimg
    print *, "img", img
  end if
  
  !$acc data copy(preimg, img)
  !$acc parallel private(x)
  !$acc loop
  do i=1,N
    x = preimg(i)
    img(i) = ATAN(x)
  end do
  !$acc end parallel
  !$acc end data

  if (verbose .ge. 1) then
    print *, "preimg", preimg
    print *, "img", img
  end if
  
  imgSum = 0.0
  do i=1,N
    imgSum = imgSum + img(i)
  end do
  
  print *, "sum", imgSum
  
  ! write results to a text file
  open (unit=fid, file="results.txt", action="write", status="replace")
  do i=1,N
    write (fid,*) preimg(i), " ", img(i)
  end do
  write (fid,*) N, " ", imgSum
  close(fid)
  
  open (unit=bid, file="results.bin", form="unformatted")
  write (bid) img
  close(bid)
  
  !deallocate(preimg)
  !deallocate(img)
  
end program shadow

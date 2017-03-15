module transcendental
  use iso_c_binding
  use iso_fortran_env

  ! -------------------------------------------------------------------------
  ! INTERFACE TO USE THE C/C++ IMPLEMENTATIONS:
  ! - EXP
  ! - LOG
  ! -------------------------------------------------------------------------
  interface LOG
    module procedure log_scalar, log_vect
  end interface LOG
  
  interface EXP
    module procedure exp_scalar, exp_vect, exp_matrix
  end interface EXP

  interface
    pure real(c_double) function C_EXP(x) result(y) bind(C, name="br_exp")
      !$acc routine seq
      use iso_c_binding
      implicit none
      real(c_double), value :: x
    end function C_EXP

    pure real(c_double) function C_LOG(x) result(y) bind(C, name="br_log")
      !$acc routine seq
      use iso_c_binding
      implicit none
      real(c_double), value :: x
    end function C_LOG
  end interface

  interface LOG10
    module procedure MCH_LOG10
  end interface LOG10

contains
  ! ===========================================================
  ! LOGARITHM IN BASE 10: COMPUTED AS LOG X DIVIDED BY LN
  ! OF 10
  ! ===========================================================
  pure real(kind=real64) function MCH_LOG10(x) result(y)
    !$acc routine seq
    real(kind=real64), intent(in) :: x
    real(kind=real64), parameter :: LN_10 = 2.30258509299404568402
    y = C_LOG(x)/LN_10
  end function MCH_LOG10

  ! ===========================================================
  ! SCALAR FORM OF THE LOG FUNCTION
  ! ===========================================================  
  pure real(kind=real64) function log_scalar(x) result(y)
    !$acc routine seq
    real(kind=real64), intent(in) :: x
    y = C_LOG(x)
  end function log_scalar

  ! ===========================================================
  ! VECTOR FORM OF THE LOG FUNCTION
  ! ===========================================================
  pure function log_vect(x) result(y)
    !$acc routine seq
    real(kind=real64), dimension(:), intent(in) :: x

    integer :: i
    real(kind=real64) :: y(size(x))

    do i = 1, size(x)
      y(i) = C_LOG(x(i))
    end do
  end function log_vect
  
  ! ===========================================================
  ! SCALAR FORM OF THE EXP FUNCTION
  ! ===========================================================
  pure real(kind=real64) function exp_scalar(x) result(y)
    !$acc routine seq
    real(kind=real64), intent(in) :: x
    y = C_EXP(x)
  end function exp_scalar

  ! ===========================================================
  ! VECTOR FORM OF THE EXP FUNCTION
  ! ===========================================================
  pure function exp_vect(x) result(y)
    !$acc routine seq
    real(kind=real64), dimension(:), intent(in) :: x

    integer :: i
    real(kind=real64) :: y(size(x))

    do i = 1, size(x)
      y(i) = C_EXP(x(i))
    end do
  end function exp_vect

  ! ===========================================================
  ! 2D MATRIX FORM OF THE EXP FUNCTION
  ! ===========================================================
  pure function exp_matrix(x) result(y)
    !$acc routine seq
    real(kind=real64), dimension(:,:), intent(in) :: x

    integer :: i, j
    real(kind=real64) :: y(size(x, 1), size(x, 2))

    do j = 1, size(x, 2)
      do i = 1, size(x, 1)
        y(i, j) = C_EXP(x(i, j))
      end do
    end do
  end function exp_matrix

end module transcendental

module mchpow
  use iso_fortran_env
  implicit none
  
  interface POW
    module procedure pow_scalar_ff, pow_scalar_fi, pow_scalar_if, pow_scalar_ii !, &
                     !pow_vect_ff, pow_matrix_ff, &
                     !pow_3dmatrix_ff
  end interface POW
  
  interface
    elemental real(c_double) function C_POW(x, y) result(z) bind(C, name="br_pow")
      !$acc routine seq
      use iso_c_binding
      implicit none
      real(c_double), value :: x, y
    end function C_POW
  end interface

contains
  ! ============================================================================
  ! SCALAR FORM OF THE FUNCTION
  ! ============================================================================
  elemental real(kind=real64) function pow_scalar_ff(x, y) result(z)
    !$acc routine seq
    real(kind=real64), intent(in) :: x, y
    z = C_POW(x, y)
  end function pow_scalar_ff
  
  elemental real(kind=real64) function pow_scalar_fi(x, y) result(z)
    !$acc routine seq
    real(kind=real64), intent(in) :: x
    integer(kind=int32), intent(in) :: y
    !z = x**y ! if y < 0 => 1.0/(x^y)
    z = C_POW(r, y)
  end function pow_scalar_fi
  
  elemental real(kind=real64) function pow_scalar_if(x, y) result(z)
    !$acc routine seq
    integer(kind=int32), intent(in) :: x
    real(kind=real64), intent(in) :: y
    real(kind=real64) :: r
    r = real(x)
    z = C_POW(r, y)
  end function pow_scalar_if
  
  elemental integer(kind=int32) function pow_scalar_ii(x, y) result(z)
    !$acc routine seq
    integer(kind=int32), intent(in) :: x
    integer(kind=int32), intent(in) :: y
    z = x**y
  end function pow_scalar_ii
  
  ! ============================================================================
  ! VECTOR FORM OF THE FUNCTION
  ! ============================================================================
  pure function pow_vect_ff(x, y) result(z)
    !$acc routine seq
    real(kind=real64), dimension(:), intent(in) :: x
    real(kind=real64), intent(in) :: y    
    integer(kind=int32) :: i
    real(kind=real64) :: z(size(x))

    do i = 1, size(x)
      z(i) = x(i)**y
    end do
  end function pow_vect_ff

  ! ============================================================================
  ! 2D MATRIX FORM OF THE EXP FUNCTION
  ! ============================================================================
  pure function pow_matrix_ff(x, y) result(z)
    !$acc routine seq
    real(kind=real64), dimension(:,:), intent(in) :: x
    real(kind=real64), intent(in) :: y
    integer(kind=int32) :: i, j
    real(kind=real64) :: z(size(x, 1), size(x, 2))

    do j = 1, size(x, 2)
      do i = 1, size(x, 1)
        z(i,j) = C_POW(x(i,j), y)
      end do
    end do
  end function pow_matrix_ff

  ! ============================================================================
  ! 2D MATRIX FORM OF THE EXP FUNCTION
  ! ============================================================================
  pure function pow_3dmatrix_ff(x, y) result(z)
    !$acc routine seq
    real(kind=real64), dimension(:,:,:), intent(in) :: x
    real(kind=real64), intent(in) :: y
    integer(kind=int32) :: i, j, k
    real(kind=real64) :: z(size(x,1), size(x,2), size(x,3))

    do k = 1, size(x,3)
      do j = 1, size(x,2)
        do i = 1, size(x,1)
          z(i,j,k) = C_POW(x(i,j,k), y)
        end do
      end do
    end do
  end function pow_3dmatrix_ff

end module mchpow

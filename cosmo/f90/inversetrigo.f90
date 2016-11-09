module inversetrigo
  use iso_fortran_env  
  implicit none

  interface ACOS
    module procedure acos_scalar_f
  end interface ACOS

  interface ASIN
    module procedure asin_scalar_f
  end interface ASIN

  interface ATAN
    module procedure atan_scalar_f
  end interface ATAN

  interface
    pure real(c_double) function C_ACOS(x) result(y) bind(C, name="br_acos")
      !$acc routine seq
      use iso_c_binding
      implicit none
      real(c_double), value :: x
    end function C_ACOS

    pure real(c_double) function C_ASIN(x) result(y) bind(C, name="br_asin")
      !$acc routine seq
      use iso_c_binding
      implicit none
      real(c_double), value :: x
    end function C_ASIN

    pure real(c_double) function C_ATAN(x) result(y) bind(C, name="br_atan")
      !$acc routine seq
      use iso_c_binding
      implicit none
      real(c_double), value :: x
    end function C_ATAN
  end interface

contains
  ! ===========================================================
  ! INVERT COSINE
  ! ===========================================================
  pure real(kind=real64) function acos_scalar_f(x) result(y)
    !$acc routine seq
    real(kind=real64), intent(in) :: x
    y = C_ACOS(x)
  end function acos_scalar_f

  ! ===========================================================
  ! INVERT SINE
  ! ===========================================================
  pure real(kind=real64) function asin_scalar_f(x) result(y)
    !$acc routine seq
    real(kind=real64), intent(in) :: x
    y = C_ASIN(x)
  end function asin_scalar_f

  ! ===========================================================
  ! INVERT TANGENT
  ! ===========================================================
  pure real(kind=real64) function atan_scalar_f(x) result(y)
    !$acc routine seq
    real(kind=real64), intent(in) :: x
    y = C_ATAN(x)
  end function atan_scalar_f

end module inversetrigo

module trigo
  use iso_fortran_env  
  implicit none

  interface COS
    module procedure cos_scalar_f
  end interface COS

  interface SIN
    module procedure sin_scalar_f
  end interface SIN

  interface TAN
    module procedure tan_scalar_f
  end interface TAN

  interface
    pure real(c_double) function C_COS(x) result(y) bind(C, name="br_cos")
      !$acc routine seq
      use iso_c_binding
      implicit none
      real(c_double), value :: x
    end function C_COS

    pure real(c_double) function C_SIN(x) result(y) bind(C, name="br_sin")
      !$acc routine seq
      use iso_c_binding
      implicit none
      real(c_double), value :: x
    end function C_SIN

    pure real(c_double) function C_TAN(x) result(y) bind(C, name="br_tan")
      !$acc routine seq
      use iso_c_binding
      implicit none
      real(c_double), value :: x
    end function C_TAN
  end interface

contains
  ! ------------------------------------------------------------------------
  ! COMPUTE THE SINE
  ! ------------------------------------------------------------------------
  pure real(kind=real64) function sin_scalar_f(x) result(y)
    !$acc routine seq
    real(kind=real64), intent(in) :: x
    y = C_SIN(x)
  end function sin_scalar_f

  ! ------------------------------------------------------------------------
  ! COMPUTE THE COSINE
  ! ------------------------------------------------------------------------
  pure real(kind=real64) function cos_scalar_f(x) result(y)
    !$acc routine seq
    real(kind=real64), intent(in) :: x
    y = C_COS(x)
  end function cos_scalar_f

  ! ------------------------------------------------------------------------
  ! COMPUTE THE TANGENT
  ! ------------------------------------------------------------------------
  pure real(kind=real64) function tan_scalar_f(x) result(y)
    !$acc routine seq
    real(kind=real64), intent(in) :: x
    y = C_TAN(x)
  end function tan_scalar_f

end module

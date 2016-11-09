module hyperbolic
  use iso_fortran_env
  implicit none

  interface SINH
    module procedure sinh_scalar_f
  end interface SINH

  interface TANH
    module procedure tanh_scalar_f
  end interface TANH

  interface
    pure real(c_double) function C_SINH(x) result(y) bind(C, name="br_sinh")
      !$acc routine seq
      use iso_c_binding
      implicit none
      real(c_double), value :: x
    end function C_SINH

    pure real(c_double) function C_TANH(x) result(y) bind(C, name="br_tanh")
      !$acc routine seq
      use iso_c_binding
      implicit none
      real(c_double), value :: x
    end function C_TANH
  end interface

contains
  ! ===========================================================
  ! HYPERBOLIC COSINE
  ! ===========================================================
  pure real(kind=real64) function sinh_scalar_f(x) result(y)
    !$acc routine seq
    real(kind=real64), intent(in) :: x
    y = C_SINH(x)
  end function sinh_scalar_f

  ! ===========================================================
  ! HYPERBOLIC TANGENT
  ! ===========================================================
  pure real(kind=real64) function tanh_scalar_f(x) result(y)
    !$acc routine seq
    real(kind=real64), intent(in) :: x
    y = C_TANH(x)
  end function tanh_scalar_f
end module hyperbolic

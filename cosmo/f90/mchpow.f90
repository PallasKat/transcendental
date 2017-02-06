module mchpow
  use iso_fortran_env
  implicit none
  
  interface POW
    module procedure pow_scalar_ff, pow_scalar_fi, pow_scalar_if, pow_scalar_ii
  end interface POW

contains

  elemental function pow_scalar_ff(x, y) result(z)
    use, intrinsic :: iso_c_binding
    implicit none

    real(kind=real64), intent(in) :: x
    real(kind=real64), intent(in) :: y
    real(kind=real64) :: z

    interface
      pure function C_POW(x, y) bind(C, name="br_pow")
        import
        real(c_double), value, intent(in) :: x
        real(c_double), value, intent(in) :: y
        real(c_double) :: C_POW
      end function
    end interface

    z = C_POW(x, y)
  end function pow_scalar_ff

  elemental function pow_scalar_fi(x, i) result(z)
    use, intrinsic :: iso_c_binding
    implicit none

    real(kind=real64), intent(in) :: x
    integer(kind=int32), intent(in) :: i
    real(kind=real64) :: z
    real(c_double) :: r

    interface
      pure function C_POW(x, y) bind(C, name="br_pow")
        import
        real(c_double), value, intent(in) :: x
        real(c_double), value, intent(in) :: y
        real(c_double) :: C_POW
      end function
    end interface
    r = real(i)
    z = C_POW(x, r)
  end function pow_scalar_fi

  elemental function pow_scalar_if(i, y) result(z)
    use, intrinsic :: iso_c_binding
    implicit none

    integer(kind=int32), intent(in) :: i
    real(kind=real64), intent(in) :: y
    real(kind=real64) :: z
    real(c_double) :: r

    interface
      pure function C_POW(x, y) bind(C, name="br_pow")
        import
        real(c_double), value, intent(in) :: x
        real(c_double), value, intent(in) :: y
        real(c_double) :: C_POW
      end function
    end interface
    r = real(i)
    z = C_POW(r, y)
  end function pow_scalar_if

  elemental function pow_scalar_ii(x, y) result(z)
    implicit none

    integer(kind=int32), intent(in) :: x
    integer(kind=int32), intent(in) :: y
    integer(kind=int32) :: z

    z = x**y
  end function pow_scalar_ii

end module mchpow

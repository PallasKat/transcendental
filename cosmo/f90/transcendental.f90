module transcendental
  use iso_c_binding
  use iso_fortran_env

  ! -------------------------------------------------------------------------
  ! INTERFACE TO USE THE C/C++ IMPLEMENTATIONS:
  ! - EXP
  ! - LOG
  ! -------------------------------------------------------------------------
  interface LOG
    module procedure log_scalar !, log_vect
  end interface LOG
  
  interface EXP
    module procedure exp_scalar !, exp_vect, exp_matrix
  end interface EXP

  !interface
  !  elemental real(c_double) function C_EXP(x) result(y) bind(C, name="br_exp")
  !    !$acc routine seq
  !    use iso_c_binding
  !    implicit none
  !    real(c_double), value :: x
  !  end function C_EXP

  !  elemental real(c_double) function C_LOG(x) result(y) bind(C, name="br_log")
  !    !$acc routine seq
  !    use iso_c_binding
  !    implicit none
  !    real(c_double), value :: x
  !  end function C_LOG
  !end interface

  interface LOG10
    module procedure log10_scalar ! MCH_LOG10
  end interface LOG10

contains

  elemental function log_scalar(x) result(y)
    use, intrinsic :: iso_c_binding
    implicit none

    real(c_double), intent(in) :: x    
    real(c_double) :: br_log
    real(kind=real64) :: y

    interface
      pure function C_LOG(x) bind(C, name = 'br_log')
        import
        real(c_double), value, intent(in) :: x
        real(c_double) :: C_LOG
      end function
    end interface

    y = C_LOG(x)
  end function

  elemental function exp_scalar(x) result(y)
    use, intrinsic :: iso_c_binding
    implicit none

    real(c_double), intent(in) :: x
    real(c_double) :: br_exp
    real(kind=real64) :: y

    interface
      pure function C_EXP(x) bind(C, name = 'br_exp')
        import
        real(c_double), value, intent(in) :: x
        real(c_double) :: C_EXP
      end function
    end interface
    
    y = C_EXP(x)
  end function

  elemental function log10_scalar(x) result(y)
    use, intrinsic :: iso_c_binding
    implicit none

    real(kind=real64), parameter :: INV_LN_10 = 0.434294481903251827651129
    real(c_double), intent(in) :: x
    real(c_double) :: br_exp
    real(kind=real64) :: y

    interface
      pure function C_EXP(x) bind(C, name = 'br_exp')
        import
        real(c_double), value, intent(in) :: x
        real(c_double) :: C_EXP
      end function
    end interface
    
    y = C_EXP(x)*INV_LN_10
  end function

end module transcendental


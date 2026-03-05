# mypy: ignore-errors
"""
This function prints the source code for spherical_harmonics_ylms.py to console

spherical_harmonics pre-computes the analytical solutions to each real spherical harmonic with sympy
the script contains different functions for different degrees l and orders m

Marc Russwurm
"""

import sys
from datetime import datetime

from sympy import Abs, Symbol, assoc_legendre, cos, factorial, pi, sin, sqrt

theta = Symbol("theta")
phi = Symbol("phi")


def calc_ylm(degree, order):
    """
    see last equation of https://en.wikipedia.org/wiki/Spherical_harmonics#Real_form
    """
    if order < 0:
        Plm = assoc_legendre(degree, Abs(order), cos(theta))
        Plm_bar = (
            sqrt(
                ((2 * degree + 1) / (4 * pi))
                * (
                    factorial(degree - Abs(order))
                    / factorial(degree + Abs(order))
                )
            )
            * Plm
        )

        Ylm = (-1) ** order * sqrt(2) * Plm_bar * sin(Abs(order) * phi)
    elif order == 0:
        Ylm = sqrt((2 * degree + 1) / (4 * pi)) * assoc_legendre(
            degree, order, cos(theta)
        )
    else:  # order > 0
        Plm = assoc_legendre(degree, order, cos(theta))
        Plm_bar = (
            sqrt(
                ((2 * degree + 1) / (4 * pi))
                * (factorial(degree - order) / factorial(degree + order))
            )
            * Plm
        )

        Ylm = (-1) ** order * sqrt(2) * Plm_bar * cos(order * phi)
    return Ylm


def print_function(degree, order):
    fname = f"Yl{degree}_m{order}".replace("-", "_minus_")
    print()
    print("@torch.jit.script")
    print(f"def {fname}(theta, phi):")
    print("    return " + str(calc_ylm(degree, order).evalf()))


# max number of Legendre Polynomials
L = 101

head = (
    """\"\"\"
analytic expressions of spherical harmonics generated with sympy file
Marc Russwurm generated """
    + str(datetime.date(datetime.now()))
    + """

run
python """
    + sys.argv[0]
    + """ > spherical_harmonics_ylm.py

to generate the source code
\"\"\"

import torch
from torch import cos, sin

def get_SH(m,l):
  fname = f"Yl{l}_m{m}".replace("-","_minus_")
  return globals()[fname]

def SH(m, l, phi, theta):
  Ylm = get_SH(m,l)
  return Ylm(theta, phi)
"""
)
print(head)
print()

for degree in range(L):
    for order in range(-degree, degree + 1):
        print_function(degree, order)

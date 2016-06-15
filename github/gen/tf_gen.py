from collections import namedtuple
import csv

# To check if we need to use the map from Pyhton 2.x or 3.x
try:
    from itertools import imap
except ImportError:
    imap = map

# A simple container of string constants as there is no way to declare const
# values
class ConstString:
  def __init__(self):
    self._s2 = 2*" "
    self._l1 = "\n" + self._s2
    self._l2 = self._l1 + self._s2
    self._l3 = self._l2 + self._s2

  def l1(self):
    return self._l1

  def l2(self):    
    return self._l2

  def l3(self):
    return self._l3

FuncData = namedtuple('Functors', 'name, cpu, gpu, lib, narg')

def readFunctors(fname):
  functors = []  
  with open(fname, "rb") as infile:
    for func in imap(FuncData._make, csv.reader(infile)):
      functors.append(func)
  return functors

def genCppCode(cName, fName, nArg=1, dev=False):
  cs = ConstString()
  argDecla = "(double x)"
  argCall = "(x)"
  if nArg == 2:
    argDecla = "(double x, double y)"
    argCall = "(x, y)"

  cDecla = "class " + cName + " {" + cs.l1() + "public:" + cs.l2()
  cOp = "double operator() " + argDecla + " const {" + cs.l3()
  cBody = "return " + fName + argCall + ";" + cs.l2()
  cEnd = "}\n};"

  return cDecla + cOp + cBody + cEnd

def cfunctors(func):
  name = func.name.capitalize()
  n = int(func.narg)
  fCpu = genCppCode("Cpu" + name, func.cpu, n)
  fGpu = genCppCode("Gpu" + name, func.gpu, n)
  fLib = genCppCode("Lib" + name, func.lib, n)
  return fCpu, fGpu, fLib

fname = "fdata.csv"
fs = readFunctors(fname)
for f in fs:
  c, g, l = cfunctors(f)

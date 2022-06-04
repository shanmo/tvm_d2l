import numpy as np

np.random.seed(0)
n = 100
# Note that NumPy in default uses 64-bit floating-points or 64-bit integers,
# which is different from 32-bit floating point typically used in deep learning,
# so we explicitly cast the data type
a = np.random.normal(size=n).astype(np.float32)
b = np.random.normal(size=n).astype(np.float32)
c = a + b

def vector_add(a, b, c):
    for i in range(n):
        c[i] = a[i] + b[i]

d = np.empty(shape=n, dtype=np.float32)
vector_add(a, b, d)
np.testing.assert_array_equal(c, d)

def get_abc(shape, constructor=None):
    np.random.seed(0)
    # Note that NumPy in default uses 64-bit floating-points or 64-bit integers,
    # which is different from 32-bit floating point typically used in deep learning,
    # so we explicitly cast the data type
    a = np.random.normal(size=n).astype(np.float32)
    b = np.random.normal(size=n).astype(np.float32)
    c = np.empty_like(a)
    if constructor:
        a, b, c = [constructor(x) for x in (a, b, c)]
    return a, b, c

import tvm
from tvm import te # te stands for tensor expression

def vector_add(n):
    A = te.placeholder((n,), name='a')
    B = te.placeholder((n,), name='b')
    C = te.compute(A.shape, lambda i: A[i] + B[i], name='c')
    return A, B, C

A, B, C = vector_add(n)
print(f"type(A) {type(A)}")
print(f"type(C) {type(C)}")

s = te.create_schedule(C.op)
tvm.lower(s, [A, B, C], simple_mode=True)

mod = tvm.build(s, [A, B, C])
print(f"type(mod) {type(mod)}")

x = np.ones(2)
y = tvm.nd.array(x)
print(f"type(y) {type(y)}")
print(f"y.asnumpy() {type(y.asnumpy())}")

a, b, c = get_abc(100, tvm.nd.array)
mod(a, b, c)
np.testing.assert_array_equal(a.asnumpy() + b.asnumpy(), c.asnumpy())

'''
type(A) <class 'tvm.te.tensor.Tensor'>
type(C) <class 'tvm.te.tensor.Tensor'>
type(mod) <class 'tvm.driver.build_module.OperatorModule'>
type(y) <class 'tvm.runtime.ndarray.NDArray'>
y.asnumpy() <class 'numpy.ndarray'>
'''

try:
    print("wrong size")
    a, b, c = get_abc(200, tvm.nd.array)
    mod(a, b, c)
except tvm.TVMError as e:
    print(f"error {e}")

try:
    print("wrong datatype")
    a, b, c = get_abc(100, tvm.nd.array)
    a = tvm.nd.array(a.asnumpy().astype('float64'))
    mod(a, b, c)
except tvm.TVMError as e:
    print(f"error {e}")

'''
wrong size
Traceback (most recent call last):
  1: TVMFuncCall
  0: tvm::runtime::PackedFuncObj::Extractor<tvm::runtime::PackedFuncSubObj<tvm::runtime::WrapPackedFunc(int (*)(TVMValue*, int*, int, TVMValue*, int*, void*), tvm::runtime::ObjectPtr<tvm::runtime::Object> const&)::{lambda(tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)#1}> >::Call(tvm::runtime::PackedFuncObj const*, tvm::runtime::TVMArgs, tvm::runtime::TVMRetValue*)
  File "/home/sean/mylibs/tvm/src/runtime/library_module.cc", line 80
TVMError: 
---------------------------------------------------------------
An error occurred during the execution of TVM.
For more information, please see: https://tvm.apache.org/docs/errors.html
---------------------------------------------------------------

  Check failed: ret == 0 (-1 vs. 0) : Assert fail: (((tir.tvm_struct_get(arg.a, 0, 5) == (uint8)2) && (tir.tvm_struct_get(arg.a, 0, 6) == (uint8)32)) && (tir.tvm_struct_get(arg.a, 0, 7) == (uint16)1)), arg.a.dtype is expected to be float32
'''

mod_fname = 'vector-add.tar'
mod.export_library(mod_fname)

loaded_mod = tvm.runtime.load_module(mod_fname)
a, b, c = get_abc(100, tvm.nd.array)
loaded_mod(a, b, c)
np.testing.assert_array_equal(a.asnumpy() + b.asnumpy(), c.asnumpy())
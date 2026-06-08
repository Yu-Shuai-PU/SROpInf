"""Common type aliases for array annotations."""
import numpy as np
import numpy.typing as npt
from typing import Annotated, Tuple, Union, Callable

ROMTensor0D = np.float64
ROMTensor1D = Annotated[npt.NDArray[np.float64], lambda arr: arr.ndim == 1]
ROMTensor2D = Annotated[npt.NDArray[np.float64], lambda arr: arr.ndim == 2]
ROMTensorHD = Annotated[npt.NDArray[np.float64], lambda arr: arr.ndim >= 3]
ROMTensorTuple = Tuple[Union[ROMTensor0D, ROMTensor1D, ROMTensor2D, ROMTensorHD], ...] # for tuple of ROM tensors (for now, only real-valued quadratic ROMs)

Vector = Annotated[npt.NDArray[np.float64], lambda arr: arr.ndim == 1]
Matrix = Annotated[npt.NDArray[np.float64], lambda arr: arr.ndim == 2]
ComplexVector = Annotated[npt.NDArray[np.complex128], lambda arr: arr.ndim == 1]
ComplexMatrix = Annotated[npt.NDArray[np.complex128], lambda arr: arr.ndim == 2]
ScalarFunc = Callable[[float], float]
VectorFunc = Callable[[Vector], Vector]
MatrixFunc = Callable[[Matrix], Matrix]
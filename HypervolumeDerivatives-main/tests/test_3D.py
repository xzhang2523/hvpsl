import sys

import numpy as np

sys.path.insert(0, "./")
from hvd import HypervolumeDerivatives


def test_3D_case1():
    ref = np.array([9, 10, 12])
    hvh = HypervolumeDerivatives(3, 3, ref, minimization=True)
    out = hvh.compute_hessian(X=np.array([[5.0, 1.0, 7.0], [2.0, 3.0, 10.0]]))
    assert np.all(
        out["HVdY2"]
        == np.array(
            [
                [0, 5, 9, 0, -2, -7],
                [5, 0, 4, 0, 0, -0],
                [9, 4, 0, 0, -0, 0],
                [0, 0, 0, 0, 2, 7],
                [-2, 0, 0, 2, 0, 3],
                [-7, 0, 0, 7, 3, 0],
            ]
        )
    )


def test_3D_case2():
    ref = np.array([9, 10, 12])
    hvh = HypervolumeDerivatives(3, 3, ref, minimization=True)
    out = hvh.compute_hessian(X=np.array([[-1, 5, 7], [2, 3, 10]]))
    assert np.all(
        out["HVdY2"]
        == np.array(
            [
                [0, 5, 5, 0, 0, -0],
                [5, 0, 10, -2, 0, -7],
                [5, 10, 0, -0, 0, 0],
                [0, -2, 0, 0, 2, 2],
                [0, 0, 0, 2, 0, 7],
                [0, -7, 0, 2, 7, 0],
            ]
        )
    )


def test_3D_case3():
    ref = np.array([9, 10, 12])
    hvh = HypervolumeDerivatives(3, 3, ref, minimization=True)
    out = hvh.compute_hessian(X=np.array([[5, 3, 7], [2, 1, 10]]))
    assert np.all(
        out["HVdY2"]
        == np.array(
            [
                [0, 3, 7, 0, 0, -7],
                [3, 0, 4, 0, 0, -4],
                [7, 4, 0, 0, 0, 0],
                [0, -0, 0, 0, 2, 9],
                [-0, 0, 0, 2, 0, 7],
                [-7, -4, 0, 9, 7, 0],
            ]
        )
    )


def test_3D_case4():
    ref = np.array([10, 13, 23])
    hvh = HypervolumeDerivatives(3, 3, ref, minimization=True)
    out = hvh.compute_hessian(X=np.array([(8, 7, 10), (4, 11, 17), (2, 9, 21)]))
    assert np.all(out["HVdY"] == np.array([-62, -26, -12, -8, -16, -8, -8, -12, -16]))
    assert np.all(
        out["HVdY2"]
        == np.array(
            [
                [0, 13, 6, 0, -4, -2, 0, -2, -2],
                [13, 0, 2, 0, 0, 0, 0, 0, 0],
                [6, 2, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 4, 2, 0, 0, -2],
                [-4, 0, 0, 4, 0, 4, 0, 0, -4],
                [-2, 0, 0, 2, 4, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 2, 4],
                [-2, 0, 0, 0, 0, 0, 2, 0, 6],
                [-2, 0, 0, -2, -4, 0, 4, 6, 0],
            ]
        )
    )


def test_3D_case5():
    ref = np.array([17, 35, 7])
    hvh = HypervolumeDerivatives(3, 3, ref, minimization=True)
    out = hvh.compute_hessian(
        X=np.array([(16, 23, 1), (14, 32, 2), (12, 27, 3), (10, 21, 4), (8, 33, 5), (6.5, 31, 6)])
    )
    assert np.all(
        out["HVdY"]
        == np.array([-25, -3, -12, -3, -2, -6, -8, -4, -26, -36, -21, -54, -2, -2, -4, -4, -3.5, -10])
    )
    assert np.all(
        out["HVdY2"]
        == np.array(
            [
                [
                    0.0,
                    3.0,
                    12.0,
                    0.0,
                    -1.0,
                    -3.0,
                    0.0,
                    -1.0,
                    -5.0,
                    0.0,
                    0.0,
                    -4.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                ],
                [3.0, 0.0, 1.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    12.0,
                    1.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, -3.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, -2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-3.0, 0.0, 0.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    1.0,
                    8.0,
                    0.0,
                    0.0,
                    -8.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                ],
                [-1.0, 0.0, 0.0, -0.0, 0.0, 0.0, 1.0, 0.0, 4.0, 0.0, 0.0, -4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    -5.0,
                    0.0,
                    0.0,
                    -3.0,
                    -2.0,
                    0.0,
                    8.0,
                    4.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                ],
                [
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    3.0,
                    14.0,
                    0.0,
                    -1.0,
                    -2.0,
                    0.0,
                    -1.0,
                    -2.0,
                ],
                [
                    -0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    3.0,
                    0.0,
                    7.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    -0.0,
                ],
                [
                    -4.0,
                    -1.0,
                    0.0,
                    -0.0,
                    -0.0,
                    0.0,
                    -8.0,
                    -4.0,
                    0.0,
                    14.0,
                    7.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, -2.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, -2.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -2.0, 0.0, 0.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 1.0, 4.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, -0.0, 0.0, 0.0, 1.0, 0.0, 3.5],
                [
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -2.0,
                    0.0,
                    0.0,
                    -2.0,
                    -2.0,
                    0.0,
                    4.0,
                    3.5,
                    0.0,
                ],
            ]
        )
    )


def test_3D_case6():
    ref = np.array([3, 3, 3])
    hvh = HypervolumeDerivatives(3, 3, ref, minimization=True)
    out = hvh.compute_hessian(X=np.array([(1, 2, -1)]))
    assert np.all(
        out["HVdY2"]
        == np.array(
            [
                [0, 4, 1],
                [4, 0, 2],
                [1, 2, 0],
            ]
        )
    )


def test_with_dominated_points():
    ref = np.array([9, 10, 12])
    hvh = HypervolumeDerivatives(3, 3, ref, minimization=True)
    out = hvh.compute_hessian(X=np.array([[-1, -2, 7], [2, 1, 10]]))
    assert np.all(
        out["HVdY2"]
        == np.array(
            [
                [0, 5, 12, 0, 0, 0],
                [5, 0, 10, 0, 0, 0],
                [12, 10, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
    )
import pandas as pd
import numpy as np
import math
import time
import logging
from tqdm import tqdm
from module.p00_config import get_configuration
import io
import os

## P06 MatrixSolver
class MatrixSolver:
    def __init__(self, grid_ids, pred_elem, steps, x_file):
        self.grid_ids = grid_ids
        self.pred_elem = pred_elem
        self.steps = steps
        self.x_file = x_file
    
    
     # 행렬 파일 읽기
    def read_matrix_file(self, filename):
        if not os.path.exists(filename):
            logging.error(f"P06: File not found: {filename}")
            return None
        
        with open(filename, 'r') as file:
            file.readline()
            file.readline()
            line = file.readline().strip()
            data = line.split()
            if len(data) == 2:
                nRow, nCol = int(data[0]), int(data[1])
            file.readline()
            matrix = np.zeros((nRow, nCol))
            for i in range(nRow):
                line = file.readline().strip()
                data = line.split()
                for j in range(len(data)):
                    matrix[i][j] = float(data[j])
        return matrix

    def calculate_and_write_sol_matrix(self, P, invS, filename):
        SOL = P @ invS
        with open(filename, 'w') as file:
            nRow, nCol = SOL.shape
            file.write("###SOL MATRIX\n")
            file.write("#### nRow nCol\n")
            file.write(f"{nRow} {nCol}\n")
            file.write("####PRED/SENS\n")
            self.Matrix_write(SOL, file, nRow, nCol)
        logging.info(f"P06: SOL Matrix written to {filename}")

    def calculate_pseudo_inverse(self, S):
        invS = np.linalg.pinv(S)
        with open("S+_MATRIX.OUT", 'w') as file:
            nRow, nCol = invS.shape
            file.write("###S+ MATRIX\n")
            file.write("#### nRow nCol\n")
            file.write(f"{nRow} {nCol}\n")
            file.write("####MODE/SENS\n")
            self.Matrix_write(invS, file, nRow, nCol)

            E = invS @ S
            # 행렬 E의 내용은 매우 길기 때문에 DEBUG 레벨로 로깅
            logging.debug(f"P06: invS @ S (Identity Matrix Check):\n{E}")

            nRow, nCol = E.shape
            file.write("###invS x S MATRIX\n")
            file.write("#### nRow nCol\n")
            file.write(f"{nRow} {nCol}\n")
            file.write("####MODE/MODE\n")
            self.Matrix_write(E, file, nRow, nCol)
        return invS

    def Matrix_write(self, A, file, nRow, nCol):
        for i in range(nRow):
            for j in range(nCol):
                file.write(f"{A[i][j]:.6E}\t")
            file.write("\n")

    def calculate_y_matrix(self, sol_xx, sol_yy, sol_xy, sol_t1, sol_t2, sol_t3):
        with open(self.x_file, 'r') as fpx:            
            fpx.readline()
            for step in range(self.steps):
                buffer = fpx.readline()
                x_data = [float(x) for x in buffer.strip().split()]
                # x_data는 반복문 안에서 계속 출력되므로 DEBUG 레벨로 로깅
                logging.debug(f"P06: Step {step} X_DATA: {x_data}")

                time_str = x_data[0]
                output_filename = f'Y_MATRIX_{step:02d}.f06'
                with open(output_filename, 'w') as fpy:
                    fpy.write("1                         ")
                    fpy.write(f"\n     TIME: {time_str}seconds ")
                    fpy.write("\n0                                                                                                            SUBCASE 1   \n")
                    fpy.write("\n                                             D I S P L A C E M E N T   V E C T O R")
                    fpy.write("\n\n      POINT ID.   TYPE          T1             T2             T3             R1             R2             R3")

                    for idx, gid in enumerate(self.grid_ids):
                        t1 = np.dot(sol_t1[idx, :], x_data[1:])
                        t2 = np.dot(sol_t2[idx, :], x_data[1:])
                        t3 = np.dot(sol_t3[idx, :], x_data[1:])
                        r1, r2, r3 = 0, 0, 0
                        fpy.write("\n{:>14d}      G     {:>13e}  {:>13e}  {:>13e}  {:>13e}  {:>13e}  {:>13e}".format(
                            gid, t1, t2, t3, r1, r2, r3))

                    # CTRIA3 응력
                    fpy.write(f"\n     TIME: {time_str}seconds ")
                    fpy.write("\n0                                                                                                            SUBCASE 1   \n")
                    fpy.write("\n                           S T R E S S E S   I N   T R I A N G U L A R   E L E M E N T S   ( T R I A 3 )")
                    fpy.write("\n\n  ELEMENT      FIBER               STRESSES IN ELEMENT COORD SYSTEM             PRINCIPAL STRESSES (ZERO SHEAR)                 ")
                    fpy.write("\n    ID.       DISTANCE           NORMAL-X       NORMAL-Y   1   SHEAR-XY       ANGLE         MAJOR           MINOR        VON MISES")                    
                    for idx, (elem_id, elem_type) in enumerate(self.pred_elem):
                        if elem_type == 'CTRIA3':
                            s_xx = np.dot(sol_xx[idx, :], x_data[1:])
                            s_yy = np.dot(sol_yy[idx, :], x_data[1:])
                            s_xy = np.dot(sol_xy[idx, :], x_data[1:])
                            von_mises = (s_xx ** 2 - s_xx * s_yy + s_yy ** 2 + 3 * s_xy ** 2) ** 0.5
                            fpy.write("\n0{:8d}   {:>13e}     {:>13e}  {:>13e}  {:>13e}   {:>8.4f}   {:>13e}   {:>13e}  {:>13e}".format(
                                elem_id, +1, s_xx, s_yy, s_xy, 0, 0, 0, von_mises))
                            fpy.write("\n            {:>13e}     {:>13e}  {:>13e}  {:>13e}   {:>8.4f}   {:>13e}   {:>13e}  {:>13e}".format(
                                -1, s_xx, s_yy, s_xy, 0, 0, 0, von_mises))

                    # CQUAD4 응력
                    fpy.write(f"\n     TIME: {time_str}seconds ")
                    fpy.write("\n0                                                                                                            SUBCASE 1   \n")
                    fpy.write("\n                           S T R E S S E S   I N   Q U A D R I L A T E R A L   E L E M E N T S   ( Q U A D 4 )")
                    fpy.write("\n  ELEMENT      FIBER               STRESSES IN ELEMENT COORD SYSTEM             PRINCIPAL STRESSES (ZERO SHEAR)                 ")
                    fpy.write("\n    ID.       DISTANCE           NORMAL-X       NORMAL-Y      SHEAR-XY       ANGLE         MAJOR           MINOR        VON MISES")
                    for idx, (elem_id, elem_type) in enumerate(self.pred_elem):
                        if elem_type == 'CQUAD4':
                            s_xx = np.dot(sol_xx[idx, :], x_data[1:])
                            s_yy = np.dot(sol_yy[idx, :], x_data[1:])
                            s_xy = np.dot(sol_xy[idx, :], x_data[1:])
                            von_mises = (s_xx ** 2 - s_xx * s_yy + s_yy ** 2 + 3 * s_xy ** 2) ** 0.5
                            fpy.write("\n0{:8d}   {:>13e}     {:>13e}  {:>13e}  {:>13e}   {:>8.4f}   {:>13e}   {:>13e}  {:>13e}".format(
                                elem_id, +1, s_xx, s_yy, s_xy, 0, 0, 0, von_mises))
                            fpy.write("\n            {:>13e}     {:>13e}  {:>13e}  {:>13e}   {:>8.4f}   {:>13e}   {:>13e}  {:>13e}".format(
                                -1, s_xx, s_yy, s_xy, 0, 0, 0, von_mises))
    
                logging.info(f">>> P06: {output_filename} created successfully.")
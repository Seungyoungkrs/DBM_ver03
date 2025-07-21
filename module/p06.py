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
            print(f'{filename} 파일을 찾을 수 없습니다.')
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
        print(f"SOL Matrix written to {filename}")

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
            print(E)
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
                print(x_data)
                time_str = x_data[0]
                with open(f'Y_MATRIX_{step:02d}.f06', 'w') as fpy:
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
    
                print(f">>> Y_MATRIX_{step:02d}.f06 파일 생성 완료")
                    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#     def generate_matrix(self, sen_results, ref_results):        
#         self.GlStsAll_sen_df = self.process_initial_data(sen_results)
#         self.GlStsAll_ref_df = self.process_initial_data(ref_results)
#         self.BM_plus, self.M, self.M_pinv, self.B = self.compute_matrices()        
#         self.verify_identity_matrix()         
        
#     def cal_BM_matrix(self):
#         print("Calculate BM Matrix.") 
#         self.cal_BM()       
        
#     def process_initial_data(self, results):
#         full_results = [val for pair in zip(results['full']['Real'], results['full']['Imag']) for val in pair]
#         min_results = [val for pair in zip(results['min']['Real'], results['min']['Imag']) for val in pair]
#         df = pd.DataFrame(full_results + min_results)
#         df.columns = ["ElemNo", "LC_name", "LC_no", "LC", "HA", "Freq", "sig_xx", "sig_yy", "sig_zz", "sig_xy", "sig_yz", "sig_zx"]
#         return df

#     def compute_matrices(self):
#         """
#         Computes the necessary matrices for analysis.
        
#         Returns:
#         - BM_plus: B * pseudo-inverse of M
#         - M: Transpose of sensor mode matrix
#         - M_pinv: Pseudo-inverse of M
#         - B: Transpose of reference mode matrix
#         """        
#         try:
#             M = self.GlStsMode_senMat_df.T
#             B = self.GlStsMode_refMat_df.T
#             M_pinv = np.linalg.pinv(M)
#             BM_plus = B @ M_pinv
#             return BM_plus, M, M_pinv, B
#         except np.linalg.LinAlgError as e:
#             print(f"Error in matrix computation: {e}")
#             return None, None, None, None

#     def verify_identity_matrix(self):         # 행렬 계산이 제대로되 는지 확인하는 코드   
#         chk_M = self.M @ self.M_pinv          # M * pinv(M)은 반드시 단위행렬이 되어야함
#         identity = np.eye(chk_M.shape[0])
#         if np.allclose(chk_M, identity, atol=1e-5):
#             print("M @ M_pinv is close to the identity matrix.")
#         else:
#             print("M @ M_pinv is not close to the identity matrix.")    # 해당 메시지가 뜨면 안됨

#     def process_element(self, filtered_r, filtered_i, column_index, i):        
#         sts_r = filtered_r.iloc[len(self.freq) * i : len(self.freq) * (i + 1), column_index]
#         sts_i = filtered_i.iloc[len(self.freq) * i : len(self.freq) * (i + 1), column_index]
#         return sts_r.reset_index(drop=True), sts_i.reset_index(drop=True)

#     def make_Y_all_matrix(self, filtered_r, filtered_i, column_index, num_elements):
#         data_r = []
#         data_i = []        
#         for i in range(num_elements):
#             sts_r, sts_i = self.process_element(filtered_r, filtered_i, column_index, i)
#             sts_df_r = pd.DataFrame(sts_r).reset_index(drop=True)
#             sts_df_i = pd.DataFrame(sts_i).reset_index(drop=True)
#             sts_df_r.columns = [f"Sensor{i+1}"]
#             sts_df_i.columns = [f"Sensor{i+1}"]
#             data_r.append(sts_df_r)
#             data_i.append(sts_df_i)
#         result_df_r = pd.concat(data_r, axis=1)
#         result_df_i = pd.concat(data_i, axis=1)
#         result_df_r.columns = [f"Sensor{i+1}" for i in range(num_elements)]
#         result_df_i.columns = [f"Sensor{i+1}" for i in range(num_elements)]

#         result_df_r.index = self.freq
#         result_df_i.index = self.freq            
#         result = (result_df_r**2 + result_df_i**2)**0.5
        
#         return result, result_df_r, result_df_i
    
#     def optimize_dataframe(self, df):
#         df['LC_name'] = df['LC_name'].astype('category')
#         df['LC'] = df['LC'].astype('category')
#         return df

#     def filter_data(self, df, ha, lc, stress_type):
#         ha_mask = df['HA'] == ha
#         lc_mask = df['LC_name'] == lc
#         desc_mask = df['LC'].str.contains(stress_type, case=False, na=False)
#         return df[ha_mask & lc_mask & desc_mask].copy()
    
#     def cal_BM(self):
#         Ysen_all_list = []
#         Ysen_all_r_list = []
#         Ysen_all_i_list = []
#         Yref_all_list = []        

#         sen_elemno_order = self.sensor_ids
#         ref_elemlist = self.ref_sensor_ids.tolist()        
#         ref_elemno_order = [int(x) for x in ref_elemlist]
        
#         self.GlStsAll_sen_df = self.optimize_dataframe(self.GlStsAll_sen_df)
#         self.GlStsAll_ref_df = self.optimize_dataframe(self.GlStsAll_ref_df)

#         pd_version = tuple(map(int, pd.__version__.split('.')))                # Pandas version check (버전에 따른 정렬 문제 발생하여 분기 설정)
        
#         for rLC in self.LC:
#             for rHA in self.ha:                
#                 df_real_sen = self.filter_data(self.GlStsAll_sen_df, rHA, rLC, 'Real')
#                 df_imag_sen = self.filter_data(self.GlStsAll_sen_df, rHA, rLC, 'Imag')
#                 df_real_ref = self.filter_data(self.GlStsAll_ref_df, rHA, rLC, 'Real')
#                 df_imag_ref = self.filter_data(self.GlStsAll_ref_df, rHA, rLC, 'Imag')       
                
#                 if pd_version >= (2, 0, 0):
#                     # Pandas 2.x logic
#                     # Ensure ElemNo is int
#                     df_real_sen["ElemNo"] = df_real_sen["ElemNo"].astype(int)
#                     df_imag_sen["ElemNo"] = df_imag_sen["ElemNo"].astype(int)
#                     df_real_ref["ElemNo"] = df_real_ref["ElemNo"].astype(int)
#                     df_imag_ref["ElemNo"] = df_imag_ref["ElemNo"].astype(int)

#                     # Apply pd.Categorical
#                     df_real_sen["ElemNo"] = pd.Categorical(df_real_sen["ElemNo"], categories=sen_elemno_order, ordered=True)
#                     df_imag_sen["ElemNo"] = pd.Categorical(df_imag_sen["ElemNo"], categories=sen_elemno_order, ordered=True)
#                     df_real_ref["ElemNo"] = pd.Categorical(df_real_ref["ElemNo"], categories=ref_elemno_order, ordered=True)
#                     df_imag_ref["ElemNo"] = pd.Categorical(df_imag_ref["ElemNo"], categories=ref_elemno_order, ordered=True)

#                     # Sort ElemNo and Freq separately
#                     df_real_sen = df_real_sen.sort_values(["ElemNo", "Freq"]).reset_index(drop=True)
#                     df_imag_sen = df_imag_sen.sort_values(["ElemNo", "Freq"]).reset_index(drop=True)
#                     df_real_ref = df_real_ref.sort_values(["ElemNo", "Freq"]).reset_index(drop=True)
#                     df_imag_ref = df_imag_ref.sort_values(["ElemNo", "Freq"]).reset_index(drop=True)
#                 else:
#                     # Pandas 1.x logic
#                     df_real_sen.loc[:, "ElemNo"] = pd.Categorical(df_real_sen["ElemNo"], categories=sen_elemno_order, ordered=True)
#                     df_imag_sen.loc[:, "ElemNo"] = pd.Categorical(df_imag_sen["ElemNo"], categories=sen_elemno_order, ordered=True)
#                     df_real_ref.loc[:, "ElemNo"] = pd.Categorical(df_real_ref["ElemNo"], categories=ref_elemno_order, ordered=True)
#                     df_imag_ref.loc[:, "ElemNo"] = pd.Categorical(df_imag_ref["ElemNo"], categories=ref_elemno_order, ordered=True)

#                     df_real_sen = df_real_sen.sort_values(["ElemNo", "Freq"]).reset_index(drop=True)
#                     df_imag_sen = df_imag_sen.sort_values(["ElemNo", "Freq"]).reset_index(drop=True)
#                     df_real_ref = df_real_ref.sort_values(["ElemNo", "Freq"]).reset_index(drop=True)
#                     df_imag_ref = df_imag_ref.sort_values(["ElemNo", "Freq"]).reset_index(drop=True)
                
#                 Ysen, Ysen_r, Ysen_i = self.make_Y_all_matrix(df_real_sen, df_imag_sen, 6, self.num_elem)
#                 Yref, Yref_r, Yref_i = self.make_Y_all_matrix(df_real_ref, df_imag_ref, 6, self.ref_num_elem)                
                
#                 Ysen_all_list.append(Ysen)
#                 Ysen_all_r_list.append(Ysen_r)
#                 Ysen_all_i_list.append(Ysen_i)
#                 Yref_all_list.append(Yref)
                
#         # 반복문이 끝난 후 한 번에 concat 수행
#         Ysen_all = pd.concat(Ysen_all_list, axis=0)
#         Ysen_all_r = pd.concat(Ysen_all_r_list, axis=0)
#         Ysen_all_i = pd.concat(Ysen_all_i_list, axis=0)
#         Yref_all = pd.concat(Yref_all_list, axis=0)
        
#         Ysen.to_csv('Ysen.csv', index=False) 
#         Ysen_r.to_csv('Ysen_r.csv', index=False) 
#         Ysen_i.to_csv('Ysen_i.csv', index=False) 
        
#         Y_r = np.dot(self.BM_plus, Ysen_all_r.T) 
#         Y_i = np.dot(self.BM_plus, Ysen_all_i.T) 

#         Y_np = np.sqrt(Y_r**2 + Y_i**2)
#         column_names = [f"Sensor{i+1}" for i in range(self.ref_num_elem)]  

#         self.Y = pd.DataFrame(Y_np.T, index=Ysen_all_r.index, columns=column_names)
#         self.Yref_all = Yref_all
import pandas as pd
import numpy as np
import math
import logging
from tqdm import tqdm
from module.p00_config import get_configuration
from pathlib import Path

## P05 Stress Matrix 제네레이터
class MatrixGenerator:
    def __init__(self, mode_list_all, pred_data, grid_ids, sens_data, mode_stress_real_full, mode_stress_imag_full, 
                 mode_stress_static_full, mode_stress_real_min, mode_stress_imag_min, mode_stress_static_min, config):
        self.mode_list_all = mode_list_all
        self.pred_data = pred_data
        self.grid_ids = grid_ids
        self.sens_data = sens_data
        self.mode_stress_real_full = mode_stress_real_full
        self.mode_stress_imag_full = mode_stress_imag_full
        self.mode_stress_static_full = mode_stress_static_full
        self.mode_stress_real_min = mode_stress_real_min
        self.mode_stress_imag_min = mode_stress_imag_min
        self.mode_stress_static_min = mode_stress_static_min
        self.config = config

        self.mode_list_full = self.mode_list_all.full
        self.mode_list_min = self.mode_list_all.min

    def print_matrix(self, filename, mode, pred, subcase_stress_r, subcase_stress_i, stress_component):
        try:
            with open(filename, 'w') as fp:
                fp.write("##Stress matrix\n##nRow nCol\n")
                n_row = len(pred.elem_id)
                n_col = len(mode.freq)
                fp.write(f"{n_row} {n_col}\n##Elem\\Mode\n")

                buffer = []
                for eid, etype in pred:
                    row_stress = []
                    for j in range(n_col):
                        ei = subcase_stress_i[j]['map_elem'][eid]
                        real, imag = 0.0, 0.0
                        if stress_component == 'XX':
                            real = subcase_stress_r[j]['elem_all'][ei]['sxx']
                            imag = subcase_stress_i[j]['elem_all'][ei]['sxx']
                        elif stress_component == 'YY':
                            real = subcase_stress_r[j]['elem_all'][ei]['syy']
                            imag = subcase_stress_i[j]['elem_all'][ei]['syy']
                        elif stress_component == 'XY':
                            real = subcase_stress_r[j]['elem_all'][ei]['sxy']
                            imag = subcase_stress_i[j]['elem_all'][ei]['sxy']

                        wt = 2. * math.pi * mode.phase[j]
                        stress = real * math.cos(wt) + imag * math.sin(wt)
                        row_stress.append(stress)

                    buffer.append("\t".join(f"{val:.6e}" for val in row_stress))
                fp.write('\n'.join(buffer) + '\n')
#             print(f'{filename} 파일 쓰기 끝')
        except IOError:
            print(f'{filename} 열기 실패')

    def print_matrix_static(self, filename, mode, pred, subcase_stress, stress_component):
        try:
            with open(filename, 'w') as fp:
                fp.write("##Stress matrix\n##nRow nCol\n")
                n_row = len(pred.elem_id)
                n_col = 1
                fp.write(f"{n_row} {n_col}\n##Elem\\Mode\n")
                # Static 값은 스케일 다운 적용 (스태틱이 작을수록 성능이 더 좋음을 확인함)
                dS = 500
                buffer = []
                for eid, etype in pred:
                    row_stress = []
                    for j in range(n_col):
                        ei = subcase_stress[j]['map_elem'][eid]
                        stress = 0.0
                        if stress_component == 'XX':
                            stress = subcase_stress[j]['elem_all'][ei]['sxx'] / dS
                        elif stress_component == 'YY':
                            stress = subcase_stress[j]['elem_all'][ei]['syy'] / dS
                        elif stress_component == 'XY':
                            stress = subcase_stress[j]['elem_all'][ei]['sxy'] / dS
                        row_stress.append(stress)

                    buffer.append("\t".join(f"{val:.6e}" for val in row_stress))
                fp.write('\n'.join(buffer) + '\n')
#             print(f'{filename} 파일 쓰기 끝')
        except IOError:
            print(f'{filename} 열기 실패')

    def print_matrix_displacement(self, filename, mode, grid_ids, subcase_r, subcase_i, component):
        try:
            with open(filename, 'w') as fp:
                fp.write("##Displacement matrix\n##nRow nCol\n")
                n_row = len(grid_ids)
                n_col = len(mode.freq)
                fp.write(f"{n_row} {n_col}\n##Elem\\Mode\n")

                buffer = []
                for id in grid_ids:
                    row_displacement = []
                    for j in range(n_col):
                        idx = subcase_i[j]['map_grid'][id]
                        real, imag = 0.0, 0.0
                        if component == 'T1':
                            real = subcase_r[j]['grid_all'][idx]['t1']
                            imag = subcase_i[j]['grid_all'][idx]['t1']
                        elif component == 'T2':
                            real = subcase_r[j]['grid_all'][idx]['t2']
                            imag = subcase_i[j]['grid_all'][idx]['t2']
                        elif component == 'T3':
                            real = subcase_r[j]['grid_all'][idx]['t3']
                            imag = subcase_i[j]['grid_all'][idx]['t3']

                        wt = 2. * math.pi * mode.phase[j]
                        displacement = real * math.cos(wt) + imag * math.sin(wt)
                        row_displacement.append(displacement)

                    buffer.append('\t'.join(f'{val:.6e}' for val in row_displacement))
                fp.write('\n'.join(buffer) + '\n')
#             print(f'{filename} 파일 쓰기 끝')
        except IOError:
            print(f'{filename} 열기 실패')

    def print_matrix_displacement_static(self, filename, mode, grid_ids, subcase_dis, component):
        try:
            with open(filename, 'w') as fp:
                fp.write("##Displacement matrix\n##nRow nCol\n")
                n_row = len(grid_ids)
                n_col = 1
                fp.write(f"{n_row} {n_col}\n##Elem\\Mode\n")
                # Static 값은 스케일 다운 적용 (스태틱이 작을수록 성능이 더 좋음을 확인함)
                dS = 5
                
                buffer = []
                for id in grid_ids:
                    row_displacement = []
                    for j in range(n_col):
                        idx = subcase_dis[j]['map_grid'][id]
                        displacement = 0.0
                        if component == 'T1':
                            displacement = subcase_dis[j]['grid_all'][idx]['t1'] / dS
                        elif component == 'T2':
                            displacement = subcase_dis[j]['grid_all'][idx]['t2'] / dS
                        elif component == 'T3':
                            displacement = subcase_dis[j]['grid_all'][idx]['t3'] / dS
                        row_displacement.append(displacement)

                    buffer.append("\t".join(f"{val:.6e}" for val in row_displacement))
                fp.write('\n'.join(buffer) + '\n')
#             print(f'{filename} 파일 쓰기 끝')
        except IOError:
            print(f'{filename} 열기 실패')

    def concatenate_matrices(self, output_filename, input_filenames):
        try:
            total_ncol = 0
            nrow = 0

            for filename in input_filenames:
                with open(filename, 'r') as infile:
                    infile.readline()
                    infile.readline()
                    nrow_ncol = infile.readline().strip()
                    nrow_current, ncol_current = map(int, nrow_ncol.split())
                    total_ncol += ncol_current
                    if nrow == 0:
                        nrow = nrow_current
                    elif nrow != nrow_current:
                        raise ValueError(f"{filename}의 nRow 값이 일치하지 않습니다.")

            with open(output_filename, 'w') as outfile:
                outfile.write("##Stress matrix\n")
                outfile.write("##nRow nCol\n")
                outfile.write(f"{nrow} {total_ncol}\n")
                outfile.write("##Elem\\Mode\n")

                file_handles = [open(filename, 'r') for filename in input_filenames]
                for f in file_handles:
                    for _ in range(4):
                        f.readline()

                for _ in range(nrow):
                    combined_row = []
                    for f in file_handles:
                        line = f.readline().strip()
                        if line:
                            combined_row.extend(line.split())
                    outfile.write("\t".join(combined_row) + "\n")

                for f in file_handles:
                    f.close()

            print(f'{output_filename} 파일 쓰기 끝')
        except IOError as e:
            print(f'파일 처리 중 오류 발생: {e}')
        except ValueError as e:
            print(e)

    def generate_matrices(self):
        # 동적 행렬 생성 (Stress)
        self.print_matrix('P_MATRIX_XX_FULL_DYN.1', self.mode_list_full, self.pred_data, self.mode_stress_real_full, self.mode_stress_imag_full, 'XX')
        self.print_matrix('P_MATRIX_YY_FULL_DYN.1', self.mode_list_full, self.pred_data, self.mode_stress_real_full, self.mode_stress_imag_full, 'YY')
        self.print_matrix('P_MATRIX_XY_FULL_DYN.1', self.mode_list_full, self.pred_data, self.mode_stress_real_full, self.mode_stress_imag_full, 'XY')

        self.print_matrix('P_MATRIX_XX_MIN_DYN.1', self.mode_list_min, self.pred_data, self.mode_stress_real_min, self.mode_stress_imag_min, 'XX')
        self.print_matrix('P_MATRIX_YY_MIN_DYN.1', self.mode_list_min, self.pred_data, self.mode_stress_real_min, self.mode_stress_imag_min, 'YY')
        self.print_matrix('P_MATRIX_XY_MIN_DYN.1', self.mode_list_min, self.pred_data, self.mode_stress_real_min, self.mode_stress_imag_min, 'XY')

        # 정적 행렬 생성 (Stress)
        self.print_matrix_static('P_MATRIX_XX_FULL_STATIC.1', 1, self.pred_data, self.mode_stress_static_full, 'XX')
        self.print_matrix_static('P_MATRIX_YY_FULL_STATIC.1', 1, self.pred_data, self.mode_stress_static_full, 'YY')
        self.print_matrix_static('P_MATRIX_XY_FULL_STATIC.1', 1, self.pred_data, self.mode_stress_static_full, 'XY')

        self.print_matrix_static('P_MATRIX_XX_MIN_STATIC.1', 1, self.pred_data, self.mode_stress_static_min, 'XX')
        self.print_matrix_static('P_MATRIX_YY_MIN_STATIC.1', 1, self.pred_data, self.mode_stress_static_min, 'YY')
        self.print_matrix_static('P_MATRIX_XY_MIN_STATIC.1', 1, self.pred_data, self.mode_stress_static_min, 'XY')

        # 동적 행렬 생성 (Displacement)
        self.print_matrix_displacement('P_MATRIX_T1_FULL_DYN.1', self.mode_list_full, self.grid_ids, self.mode_stress_real_full, self.mode_stress_imag_full, 'T1')
        self.print_matrix_displacement('P_MATRIX_T2_FULL_DYN.1', self.mode_list_full, self.grid_ids, self.mode_stress_real_full, self.mode_stress_imag_full, 'T2')
        self.print_matrix_displacement('P_MATRIX_T3_FULL_DYN.1', self.mode_list_full, self.grid_ids, self.mode_stress_real_full, self.mode_stress_imag_full, 'T3')

        self.print_matrix_displacement('P_MATRIX_T1_MIN_DYN.1', self.mode_list_min, self.grid_ids, self.mode_stress_real_min, self.mode_stress_imag_min, 'T1')
        self.print_matrix_displacement('P_MATRIX_T2_MIN_DYN.1', self.mode_list_min, self.grid_ids, self.mode_stress_real_min, self.mode_stress_imag_min, 'T2')
        self.print_matrix_displacement('P_MATRIX_T3_MIN_DYN.1', self.mode_list_min, self.grid_ids, self.mode_stress_real_min, self.mode_stress_imag_min, 'T3')

        # 정적 행렬 생성 (Displacement)
        self.print_matrix_displacement_static('P_MATRIX_T1_FULL_STATIC.1', 1, self.grid_ids, self.mode_stress_static_full, 'T1')
        self.print_matrix_displacement_static('P_MATRIX_T2_FULL_STATIC.1', 1, self.grid_ids, self.mode_stress_static_full, 'T2')
        self.print_matrix_displacement_static('P_MATRIX_T3_FULL_STATIC.1', 1, self.grid_ids, self.mode_stress_static_full, 'T3')

        self.print_matrix_displacement_static('P_MATRIX_T1_MIN_STATIC.1', 1, self.grid_ids, self.mode_stress_static_min, 'T1')
        self.print_matrix_displacement_static('P_MATRIX_T2_MIN_STATIC.1', 1, self.grid_ids, self.mode_stress_static_min, 'T2')
        self.print_matrix_displacement_static('P_MATRIX_T3_MIN_STATIC.1', 1, self.grid_ids, self.mode_stress_static_min, 'T3')

        # 파일 합치기 (Stress)
        ALL_XX_list = ['P_MATRIX_XX_FULL_DYN.1', 'P_MATRIX_XX_MIN_DYN.1', 'P_MATRIX_XX_FULL_STATIC.1', 'P_MATRIX_XX_MIN_STATIC.1']
        ALL_YY_list = ['P_MATRIX_YY_FULL_DYN.1', 'P_MATRIX_YY_MIN_DYN.1', 'P_MATRIX_YY_FULL_STATIC.1', 'P_MATRIX_YY_MIN_STATIC.1']
        ALL_XY_list = ['P_MATRIX_XY_FULL_DYN.1', 'P_MATRIX_XY_MIN_DYN.1', 'P_MATRIX_XY_FULL_STATIC.1', 'P_MATRIX_XY_MIN_STATIC.1']

        self.concatenate_matrices('P_MATRIX_XX_COMBINED.1', ALL_XX_list)
        self.concatenate_matrices('P_MATRIX_YY_COMBINED.1', ALL_YY_list)
        self.concatenate_matrices('P_MATRIX_XY_COMBINED.1', ALL_XY_list)

        # 파일 합치기 (Displacement)
        ALL_T1_list = ['P_MATRIX_T1_FULL_DYN.1', 'P_MATRIX_T1_MIN_DYN.1', 'P_MATRIX_T1_FULL_STATIC.1', 'P_MATRIX_T1_MIN_STATIC.1']
        ALL_T2_list = ['P_MATRIX_T2_FULL_DYN.1', 'P_MATRIX_T2_MIN_DYN.1', 'P_MATRIX_T2_FULL_STATIC.1', 'P_MATRIX_T2_MIN_STATIC.1']
        ALL_T3_list = ['P_MATRIX_T3_FULL_DYN.1', 'P_MATRIX_T3_MIN_DYN.1', 'P_MATRIX_T3_FULL_STATIC.1', 'P_MATRIX_T3_MIN_STATIC.1']

        self.concatenate_matrices('P_MATRIX_T1_COMBINED.1', ALL_T1_list)
        self.concatenate_matrices('P_MATRIX_T2_COMBINED.1', ALL_T2_list)
        self.concatenate_matrices('P_MATRIX_T3_COMBINED.1', ALL_T3_list)

        self.print_matrix('S_MATRIX_FULL.1', self.mode_list_full, self.sens_data, self.mode_stress_real_full, self.mode_stress_imag_full, 'XX')
        self.print_matrix('S_MATRIX_MIN.1', self.mode_list_min, self.sens_data, self.mode_stress_real_min, self.mode_stress_imag_min, 'XX')
        self.print_matrix_static('S_MATRIX_FULL_STATIC.1', 1, self.sens_data, self.mode_stress_static_full, 'XX')
        self.print_matrix_static('S_MATRIX_MIN_STATIC.1', 1, self.sens_data, self.mode_stress_static_min, 'XX')

        # 파일 합치기 (S_MATRIX)
        ALL_S_list = ['S_MATRIX_FULL.1','S_MATRIX_MIN.1','S_MATRIX_FULL_STATIC.1','S_MATRIX_MIN_STATIC.1']
        self.concatenate_matrices('S_MATRIX_COMBINED.1', ALL_S_list)


        # 합치기 전 파일 삭제
        ALL_del_list = [
            "P_MATRIX_XX_FULL_DYN.1", "P_MATRIX_YY_FULL_DYN.1", "P_MATRIX_XY_FULL_DYN.1",
            "P_MATRIX_XX_MIN_DYN.1", "P_MATRIX_YY_MIN_DYN.1", "P_MATRIX_XY_MIN_DYN.1",
            "P_MATRIX_XX_FULL_STATIC.1", "P_MATRIX_YY_FULL_STATIC.1", "P_MATRIX_XY_FULL_STATIC.1",
            "P_MATRIX_XX_MIN_STATIC.1", "P_MATRIX_YY_MIN_STATIC.1", "P_MATRIX_XY_MIN_STATIC.1",
            "P_MATRIX_T1_FULL_DYN.1", "P_MATRIX_T2_FULL_DYN.1", "P_MATRIX_T3_FULL_DYN.1",
            "P_MATRIX_T1_MIN_DYN.1", "P_MATRIX_T2_MIN_DYN.1", "P_MATRIX_T3_MIN_DYN.1",
            "P_MATRIX_T1_FULL_STATIC.1", "P_MATRIX_T2_FULL_STATIC.1", "P_MATRIX_T3_FULL_STATIC.1",
            "P_MATRIX_T1_MIN_STATIC.1", "P_MATRIX_T2_MIN_STATIC.1", "P_MATRIX_T3_MIN_STATIC.1",
            "S_MATRIX_FULL.1", "S_MATRIX_MIN.1", "S_MATRIX_FULL_STATIC.1", "S_MATRIX_MIN_STATIC.1"
        ]

        for file in ALL_del_list:
            path = Path(file)
            path.unlink()  # 파일 삭제 

# 사용 예시
# mode_list_all, pred_data, mode_stress_real_full, mode_stress_imag_full, mode_stress_static_full,
# mode_stress_real_min, mode_stress_imag_min, mode_stress_static_min, config는 외부에서 정의되어야 함
# generator = MatrixGenerator(mode_list_all, pred_data, mode_stress_real_full, mode_stress_imag_full, mode_stress_static_full,
#                            mode_stress_real_min, mode_stress_imag_min, mode_stress_static_min, config)
# generator.generate_matrices()

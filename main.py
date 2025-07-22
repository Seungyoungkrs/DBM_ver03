"""
DBM (Deformation-Based Monitoring) 계산기 스크립트

이 스크립트는 다음과 같은 순서로 작업을 수행합니다:
1.  (P01) Hydrodynamic Analysis 결과 데이터를 읽고 처리
2.  (P02) 최적화된 모드를 구축하기 위해 별도의 알고리즘으로 처리
3.  (P03) FE 모델(요소, 노드) 정보를 읽음
4.  (P04) OP2 파일로부터 모드별 응력 및 변위 데이터를 읽음
5.  (P05) 응력 데이터를 기반으로 B, M 행렬(Matrix)을 생성
6.  (P06) 생성된 행렬을 사용하여 변환 행렬(SOL Matrix)을 계산
7.  (P07) 최종적으로 센서 데이터(X_MATRIX)로부터 전체 구조물의 응답(Y_MATRIX)을 예측하고 F06 형식 파일로 저장
"""

# --- 1. 표준 라이브러리 및 모듈 임포트 ---
import os
import time
import logging

# --- 로깅 설정 (가장 먼저 실행되도록 위로 이동) ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# 서드파티 라이브러리
import numpy as np

# 자체 모듈
from module.p00_config import get_configuration
from module.fun02_file_io import save_data, load_data
from module.p01 import HydrodynamicDataProcessor
from module.p02 import ModeCalculator
from module.p03 import FEModelLoader
from module.p04 import ModeData, ElementInfo, StressResultLoader
from module.p05 import MatrixGenerator
from module.p06 import MatrixSolver

# --- 2. 헬퍼 함수 ---
def log_execution_summary(config):
    """스크립트 실행 전 주요 설정을 요약하여 로깅합니다."""
    logging.info("=" * 60)
    logging.info(f" DBM Prediction Start: {config['title']} ".center(60, "="))
    logging.info("=" * 60)
    
    # 주요 실행 모드 요약 (0: NEW, 1: LOAD/ORG)
    mode_flags = {
        'Mode Data': 'new_mode_data',
        'Result Data': 'new_result_data',
        'OP2 Load': 'new_load_op2',
        'Ref Data': 'new_ref_data',
        'Sensor Data': 'new_sen_data'
    }
    
    logging.info("Execution Mode Summary:")
    for name, key in mode_flags.items():
        status = 'LOAD (1)' if config.get(key, 0) else 'NEW (0)'
        logging.info(f"  - {name:<15}: {status}")
        
    logging.info("-" * 60)

if __name__ == "__main__":
    start_total_time = time.time()
    config = get_configuration()

    # 설정 요약 로깅 함수 호출
    log_execution_summary(config)

    if config['new_mode_data'] == 0:
        start_time = time.time()
        p01 = HydrodynamicDataProcessor(config)
        p01.process_files()
        freq, vbm_df, hbm_df, tm_df = p01.get_results()
        elapsed_time = time.time() - start_time
        logging.info(f"P01: Hydrodynamic data processing complete. (took {elapsed_time:.2f} sec)")
        
        start_time = time.time()
        p02 = ModeCalculator(vbm_df, hbm_df, tm_df, config['nMode'])
        initial_modes = p02.load_initial_modes() 
        selected_vbm, selected_hbm, selected_tm, optimized_modes = p02.calculate_opt_modes(initial_modes)   
        variables_to_save = {
            'selected_vbm': selected_vbm,
            'selected_hbm': selected_hbm,
            'selected_tm': selected_tm,
            'optimized_modes': optimized_modes
        }
        save_data(variables_to_save, './Output/pkl/optimization_variables.pkl')         
        elapsed_time = time.time() - start_time    
        logging.info(f"P02: Mode calculation complete. Mode information is saved. (took {elapsed_time:.2f} sec)")
        
    elif config['new_mode_data'] == 1: 
        logging.info("P01: Using existing data, skipping Hydrodynamic Data loading.")      
        start_time = time.time()
        loaded_variables = load_data('./Output/pkl/optimization_variables.pkl')
        selected_vbm = loaded_variables['selected_vbm']
        selected_hbm = loaded_variables['selected_hbm']
        selected_tm = loaded_variables['selected_tm']
        optimized_modes = loaded_variables['optimized_modes']        
        elapsed_time = time.time() - start_time    
        logging.info(f"P02: Existing mode data loaded. (took {elapsed_time:.2f} sec)")
    
    start_time = time.time()    
    p03 = FEModelLoader(config)
    ElemNode_ref_df, ElemNode_sen_df, Elem_no, Elem_input_ref, Elem_input_sensor, Elem_ref_df, Node_input_ref = p03.load_fe_model_data()     
    elapsed_time = time.time() - start_time        
    logging.info(f"P03: FE Model loading complete. (took {elapsed_time:.2f} sec)")
     
    start_time = time.time()     
    p04= StressResultLoader(config)
    # Mode Data 생성
    mode_data = ModeData.create_mode_data_from_list(optimized_modes, config)
    
    # Pred Data 생성
    pred_data = ElementInfo()
    grid_ids = []
    PRED_file = config['filepaths']['ref_file']
    if p04.read_elem_file_pred(PRED_file, pred_data, grid_ids):
        logging.info("P04: PRED data loaded successfully.")
    else:
        logging.error("P04: PRED file not found. Terminating process.")
        exit(1) # 오류 종료 시에는 0이 아닌 값을 사용
        
    # Sensor Data 생성
    sens_data = ElementInfo()
    for i in Elem_input_sensor:
        sens_data.elem_type.append('CQUAD4') 
        sens_data.elem_id.append(int(i))
    read_elem = pred_data + sens_data    
    
    # Result Load from OP2 Data
    mode_stress_real_full = p04.read_op2_select(p04.filepaths['op2_real_template'].format('full'), 'full', mode_data, read_elem, grid_ids)
    mode_stress_imag_full = p04.read_op2_select(p04.filepaths['op2_imag_template'].format('full'), 'full', mode_data, read_elem, grid_ids)
    mode_stress_static_full = p04.read_op2_select(p04.filepaths['op2_static_template'].format('full'), 'full', 1, read_elem, grid_ids)
    mode_stress_real_min = p04.read_op2_select(p04.filepaths['op2_real_template'].format('min'), 'min', mode_data, read_elem, grid_ids)
    mode_stress_imag_min = p04.read_op2_select(p04.filepaths['op2_imag_template'].format('min'), 'min', mode_data, read_elem, grid_ids)
    mode_stress_static_min = p04.read_op2_select(p04.filepaths['op2_static_template'].format('min'), 'min', 1, read_elem, grid_ids)
        
    elapsed_time = time.time() - start_time    
    logging.info(f"P04: Stress result loading complete. (took {elapsed_time:.2f} sec)")
    
    start_time = time.time()    
    mode_list_all = ModeData.create_mode_data_from_list(optimized_modes, config)
    p05 = MatrixGenerator(mode_list_all, pred_data, grid_ids, sens_data, mode_stress_real_full, mode_stress_imag_full, mode_stress_static_full,
                               mode_stress_real_min, mode_stress_imag_min, mode_stress_static_min, config)
    p05.generate_matrices()        
    elapsed_time = time.time() - start_time    
    logging.info(f"P05: Prediction Matrix generation complete. (took {elapsed_time:.2f} sec)")
    
    # pred_elem과 x_file, steps 준비
    pred_elem = list(zip(pred_data.elem_id, pred_data.elem_type))
    x_file = 'X_MATRIX.DAT'
    with open(x_file, 'r') as file:
        lines = file.readlines()
    steps = len(lines) - 1

    # MatrixSolver 객체 생성
    p06 = MatrixSolver(grid_ids, pred_elem, steps, x_file)

    # S, P 행렬 읽기
    S = p06.read_matrix_file("S_MATRIX_COMBINED.1")
    P_xx = p06.read_matrix_file("P_MATRIX_XX_COMBINED.1")
    P_yy = p06.read_matrix_file("P_MATRIX_YY_COMBINED.1")
    P_xy = p06.read_matrix_file("P_MATRIX_XY_COMBINED.1")
    P_t1 = p06.read_matrix_file("P_MATRIX_T1_COMBINED.1")
    P_t2 = p06.read_matrix_file("P_MATRIX_T2_COMBINED.1")
    P_t3 = p06.read_matrix_file("P_MATRIX_T3_COMBINED.1")

    # S+ 계산
    invS = p06.calculate_pseudo_inverse(S)

    # SOL 행렬 계산 및 기록
    p06.calculate_and_write_sol_matrix(P_xx, invS, "SOL_MATRIX_XX.2")
    p06.calculate_and_write_sol_matrix(P_yy, invS, "SOL_MATRIX_YY.2")
    p06.calculate_and_write_sol_matrix(P_xy, invS, "SOL_MATRIX_XY.2")
    p06.calculate_and_write_sol_matrix(P_t1, invS, "SOL_MATRIX_T1.2")
    p06.calculate_and_write_sol_matrix(P_t2, invS, "SOL_MATRIX_T2.2")
    p06.calculate_and_write_sol_matrix(P_t3, invS, "SOL_MATRIX_T3.2")

    # SOL 행렬 다시 읽기
    sol_xx = p06.read_matrix_file("SOL_MATRIX_XX.2")
    sol_yy = p06.read_matrix_file("SOL_MATRIX_YY.2")
    sol_xy = p06.read_matrix_file("SOL_MATRIX_XY.2")
    sol_t1 = p06.read_matrix_file("SOL_MATRIX_T1.2")
    sol_t2 = p06.read_matrix_file("SOL_MATRIX_T2.2")
    sol_t3 = p06.read_matrix_file("SOL_MATRIX_T3.2")

    logging.info(f"P06: Matrix summary - Sensor elements: {sol_xx.shape[1]}, Prediction elements: {sol_xx.shape[0]}, Prediction nodes: {sol_t1.shape[0]}")

    # X_MATRIX.DAT 파일에서 스텝 수 계산
    x_file_path = 'X_MATRIX.DAT'
    with open(x_file_path, 'r') as f:
        steps = len(f.readlines()) - 1
    logging.info(f"P06: Found {steps} time steps in '{x_file_path}'.")

    # 최종 F06 결과 파일 생성
    pred_elem = list(zip(pred_data.elem_id, pred_data.elem_type))
    p06.calculate_y_matrix(sol_xx, sol_yy, sol_xy, sol_t1, sol_t2, sol_t3)
    
    logging.info(f"P06: Matrix solving and F06 generation complete. (took {time.time() - start_time:.2f} sec)")
    logging.info(f"All processes finished. Total elapsed time: {time.time() - start_total_time:.2f} sec")

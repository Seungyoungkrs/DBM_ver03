# config.py
import os
import re
import numpy as np
from pathlib import Path
import logging

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class Config:
    """
    프로젝트의 모든 설정을 관리하는 클래스.
    분석에 필요한 파일 경로와 파라미터를 설정합니다.
    """
    def __init__(self):
        # 1. 센서 번호 설정 (파일 경로 생성에 필요)
        self.sensor_number_str = "006"  # 센서 개수 6개에 해당하는 번호
        
        # 2. 파일 경로 설정
        self.filepaths = self._setup_filepaths()
        
        # 3. 센서 ID 로드 (파일이 없으면 기본값 사용)
        self.sensors = self._load_sensor_ids()
        
        # 4. 분석 파라미터 및 실행 플래그 설정
        self.params = self._get_parameters()

    def _setup_filepaths(self):
        """
        주요 입력 파일들의 전체 경로를 설정합니다.
        """
        # 기본 입력 폴더 경로 (현재 위치에서 한 단계 상위 폴더의 AOE2_input)
        base_path = Path.cwd().parent / "AOE2_input"
        sensor_path = base_path / "sensor_input_elem"
        sensor_file = sensor_path / f"sensor_input_{self.sensor_number_str}.txt"
        
        return {
            'hydrodynamic_full': str(base_path / "HYDRODYNAMIC_full.OUT"),
            'hydrodynamic_min': str(base_path / "HYDRODYNAMIC_min.OUT"),
            'data_file': str(base_path / "AOE_II_model.bdf"),
            'ref_file': str(base_path / "ref_with_shell_o1deck.bdf"),
            'op2_real_template': str(base_path / "AOE2_DBM_{}_REAL.op2"),
            'op2_imag_template': str(base_path / "AOE2_DBM_{}_IMAG.op2"),
            'op2_static_template': str(base_path / "Analysis_static_{}.op2"),
            'sensor_file': str(sensor_file),
            'sensor_number': self.sensor_number_str
        }

    def _load_sensor_ids(self):
        """
        sensor.txt 파일에서 센서 ID 목록을 읽어옵니다. 파일이 없으면 기본 목록을 반환합니다.
        """
        sensor_file_path = self.filepaths['sensor_file']
        sensors = []
        if os.path.exists(sensor_file_path):
            logging.info(f"센서 파일 '{sensor_file_path}'에서 센서 ID를 로드합니다.")
            with open(sensor_file_path, 'r') as file:
                for line in file:
                    if line.strip() and not line.startswith('!'):
                        sensors.append(int(line.split()[0]))
        else:
            logging.warning(f"센서 파일을 찾을 수 없어 기본 센서 목록을 사용합니다: {sensor_file_path}")
            # 센서 6개에 대한 기본값 예시
            sensors = [59267, 58470, 138488, 138487, 169675, 169308]
        return sensors

    def _get_parameters(self):
        """
        분석에 필요한 주요 파라미터와 실행 플래그를 반환합니다.
        """
        title = "sen" + self.sensor_number_str
        
        return {
            # --- 분석 조건 ---
            'ha': list(range(0, 181, 15)),  # Heading Angle
            'freq': [round(num, 2) for num in np.arange(0.12, 2.08 + 0.04, 0.04)], # Wave Frequency
            'nPhase': 16,                   # Wave Phase 개수
            'nMode': 400,                   # 변형 모드 개수 (400으로 고정)
            'stations': [11, 21, 31],       # 스테이션 정보 (1/4L, 1/2L, 3/4L)
            'load_case': ["full", "min"],   # LC 정보
            
            # --- 데이터 및 경로 정보 ---
            'filepaths': self.filepaths,
            'title': title,
            'sensors': self.sensors,
            'output_pkl_dir': "./Output/pkl",
            
            # --- 실행 제어 플래그 ---
            # True: 기존 pkl 데이터 로드, False: 원본 데이터부터 새로 계산/생성
            'new_mode_data': True,      # 모드 데이터 (P01, P02)
            'new_result_data': True,    # 해석 결과 데이터 (P04)
            
            # --- (현재 사용 안 함) 레거시 또는 향후 사용될 플래그 ---
            'new_load_op2': True,       # OP2 파일 로드 방식 제어
            'new_ref_data': True,       # Reference 데이터 로드 방식 제어
            'new_sen_data': True,       # 센서 데이터 로드 방식 제어
            'export_ppt': False,        # PPT 결과물 생성 여부
        }

def get_configuration():
    """
    설정 클래스 인스턴스를 생성하고, 설정 딕셔너리를 반환하는 메인 함수.
    다른 모듈에서는 이 함수를 호출하여 설정을 가져옵니다.
    """
    config_manager = Config()
    return config_manager.params

# 스크립트를 직접 실행할 때 설정 내용을 확인하기 위한 테스트 코드
if __name__ == '__main__':
    try:
        config = get_configuration()
        import json
        # 순환 참조를 피하기 위해 filepaths를 잠시 제외하고 출력
        printable_config = config.copy()
        del printable_config['filepaths']
        print(json.dumps(printable_config, indent=4))
        print("\nFile Paths:")
        print(json.dumps(config['filepaths'], indent=4, default=str)) # Path 객체를 문자열로 변환하여 출력
    except ValueError as e:
        logging.error(e)

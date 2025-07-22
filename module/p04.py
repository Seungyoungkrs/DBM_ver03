import pandas as pd
import numpy as np
import os
import pickle
import time
import logging
from datetime import datetime
from pyNastran.op2.op2 import read_op2


class ModeGroup:
    """
    LC_name별로 그룹화된 데이터를 저장하는 클래스.
    예: full, min 등의 그룹.
    """
    def __init__(self):
        self.head = []
        self.freq = []
        self.phase = []
        self.head_index = []
        self.freq_index = []
        self.raw_data = []

    def add_record(self, head, freq, phase, head_index, freq_index, raw_data):
        """그룹에 데이터를 추가합니다."""
        self.head.append(head)
        self.freq.append(freq)
        self.phase.append(phase)
        self.head_index.append(head_index)
        self.freq_index.append(freq_index)
        self.raw_data.append(raw_data)
        
class ModeData:
    def __init__(self):
        self.no_freq = 0          # 주파수 개수
        self.no_head = 0          # 헤드 개수
        self.freq = []            # 주파수 리스트
        self.head = []            # 헤드 리스트
        self.phase = []           # 위상 리스트
        self.LC_name = []         # LC 리스트
        self.list = []            # 모드 전체 리스트
        self.head_index = []      # 헤드 인덱스 리스트
        self.freq_index = []      # 주파수 인덱스 리스트
        self.grouped_data = {}    # LC_name별로 그룹화된 데이터 (ModeGroup 객체 저장)

    def __getattr__(self, name):
        """
        동적 속성 접근을 위한 메서드.
        예: mode1.full -> mode1.grouped_data['full'] 반환
        """
        if name in self.grouped_data:
            return self.grouped_data[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    @staticmethod
    def convert_to_index(value, config_list):
        """
        주어진 값(value)이 config_list에서 몇 번째 인덱스에 해당하는지 반환합니다.
        """
        try:
            return config_list.index(value)
        except ValueError:
            logging.warning(f"P04: Value {value} not found in config list: {config_list}")
            return -1  # 값이 없을 경우 -1 반환
        
    @staticmethod
    def create_mode_data_from_list(data_list, config):
        mode_data = ModeData()
        
        # 데이터 파싱 및 저장
        for line in data_list:
            parts = line.split('/')  # '/'로 구분하여 데이터 분리
            if len(parts) < 4:  # 최소 4개 값이 있는지 확인 (head/freq/phase/LC_name)
                logging.warning(f"P04: Invalid data format, skipping line: {line}")
                continue
            try:
                head = int(parts[0])  
                freq = float(parts[1])  
                phase_value = int(parts[2]) / config['nPhase']
                LC_name = parts[3]  

                mode_data.head.append(head)
                mode_data.freq.append(freq)
                mode_data.phase.append(phase_value)
                mode_data.LC_name.append(LC_name)
    
                # head_index와 freq_index 계산
                head_index = mode_data.convert_to_index(head, config['ha'])
                freq_index = mode_data.convert_to_index(freq, config['freq'])
                mode_data.head_index.append(head_index)
                mode_data.freq_index.append(freq_index)

            except ValueError as e:
                logging.error(f"P04: Data parsing error for line '{line}': {e}")
                continue
            
            # LC_name별로 데이터 그룹화 (ModeGroup 객체 사용)
            if LC_name not in mode_data.grouped_data:
                mode_data.grouped_data[LC_name] = ModeGroup()
            mode_data.grouped_data[LC_name].add_record(
                head=head,
                freq=freq,
                phase=phase_value,
                head_index=head_index,
                freq_index=freq_index,
                raw_data=line
            )
            
        # Config에 정의된 고유값 개수 사용
        mode_data.no_freq = len(config['freq'])  # 주파수의 개수
        mode_data.no_head = len(config['ha'])    # 헤드의 개수
        mode_data.list = data_list
        return mode_data       
        
# 요소정보 클래스
class ElementInfo:
    def __init__(self):
        self.elem_id = []
        self.elem_type = []
    def __add__(self,other):
        if not isinstance(other,ElementInfo):
            raise TypeError("Both operands must be of type ElementInfo")
        combined = ElementInfo()
        combined.elem_id = self.elem_id + other.elem_id
        combined.elem_type = self.elem_type + other.elem_type
        return combined
    def __iter__(self):
        return iter(zip(self.elem_id, self.elem_type))

## P04 응력결과(op2) 로더
class StressResultLoader:
    def __init__(self, config):
        self.config = config
        self.filepaths = config['filepaths']
        self.ha = config['ha']
        self.sensor_ids = config['sensors']
        self.freq = config['freq']
        self.output_dir = config['output_pkl_dir']
        self.op2_static = None
        self.op2_real = None
        self.op2_imag = None
        self.full_data = []
        self.min_data = []
        
    # bdf 읽어서 저장
    def parse_fixed_fields(self, line, width=8):
        result=[]
        for i in range(0, len(line), width):  # 0부터 문자열 길이까지 width 간격으로 반복
            field = line[i: i+width].strip() # 현재 위치에서 width만큼 잘라낸 문자열, 양쪽 공백 제거
            result.append(field)  # 결과 리스트에 추가

        return result

    # bdf 읽어서 저장
    def read_elem_file_pred(self, filename, elem_info, grid_ids):
        #grid_ids = []
        try:
            with open(filename, 'r') as fp:
                lines = fp.readlines()
        except FileNotFoundError:
            logging.error(f"P04: File not found: {filename}")
            return False

        for line in lines:
            if line.startswith('CTRIA3') or line.startswith('CQUAD4'):
                data = self.parse_fixed_fields(line)
                elem_info.elem_type.append(data[0])
                elem_info.elem_id.append(int(data[1]))
            elif line.startswith('GRID'):
                data = self.parse_fixed_fields(line)
                grid_ids.append(int(data[1]))

        with open('PRED_ids.1', 'w') as f:
            for elem_id, elem_type in elem_info:
                f.write(f"{elem_type} {elem_id}\n")
            for id in grid_ids:
                f.write(f'GRID {id}\n')    

        logging.info(f"P04: PRED elements: {len(elem_info.elem_id)}, PRED nodes: {len(grid_ids)} loaded.")
        return True  
        
    # bdf 읽어서 저장
    def read_elem_file_xsn(self, filename, elem_info):
        #grid_ids = []
        try:
            with open(filename, 'r') as fp:
                lines = fp.readlines()
        except FileNotFoundError:
            logging.error(f"P04: File not found: {filename}")
            return False

        for line in lines:
            if line.startswith('CTRIA3') or line.startswith('CQUAD4'):
                data = self.parse_fixed_fields(line)
                elem_info.elem_type.append(data[0])
                elem_info.elem_id.append(int(data[1]))
            
        with open('XSN_ids.1', 'w') as f:
            for elem_id, elem_type in elem_info:
                f.write(f"{elem_type} {elem_id}\n")            

        logging.info(f"P04: PRED elements: {len(elem_info.elem_id)}, PRED nodes: {len(grid_ids)} loaded.")      
        return True  
        
        
    def read_op2_select(self, filename, lc_name, mode_data, read_elem, read_grid):
        filename_op1 = filename[:-3] + 'op1'

        # op1파일 존재
        if os.path.isfile(filename_op1):

            try:
                with open(filename_op1, 'rb') as f:
                    data_to_load = pickle.load(f)
                    nSubcases = data_to_load['nSubcases']
                    ncquad4 = data_to_load['ncquad4']
                    nctria3 = data_to_load['nctria3']
                    ngrid = data_to_load['ngrid']
                    cquad4_element_map = data_to_load['cquad4_element_map']
                    ctria3_element_map = data_to_load['ctria3_element_map']
                    grid_map = data_to_load['grid_map']

                    # 데이터 읽기
                    data = f.read()
                    #데이터 배열로 변환
                    data_array = np.frombuffer(data, dtype = np.float32)

                    cquad4_oxx_mean = np.empty((nSubcases, ncquad4), dtype=np.float32)
                    cquad4_oyy_mean = np.empty((nSubcases, ncquad4), dtype=np.float32)
                    cquad4_txy_mean = np.empty((nSubcases, ncquad4), dtype=np.float32)

                    ctria3_oxx_mean = np.empty((nSubcases, nctria3), dtype=np.float32)
                    ctria3_oyy_mean = np.empty((nSubcases, nctria3), dtype=np.float32)
                    ctria3_txy_mean = np.empty((nSubcases, nctria3), dtype=np.float32)

                    grid_x = np.empty((nSubcases, ngrid), dtype=np.float32)
                    grid_y = np.empty((nSubcases, ngrid), dtype=np.float32)
                    grid_z = np.empty((nSubcases, ngrid), dtype=np.float32)

                    offset = 0
                    for i in range(nSubcases):
                        cquad4_oxx_mean[i] = data_array[offset: offset + ncquad4].reshape(ncquad4)
                        offset += ncquad4
                        cquad4_oyy_mean[i] = data_array[offset: offset + ncquad4].reshape(ncquad4)
                        offset += ncquad4
                        cquad4_txy_mean[i] = data_array[offset: offset + ncquad4].reshape(ncquad4)
                        offset += ncquad4

                        ctria3_oxx_mean[i] = data_array[offset: offset + nctria3].reshape(nctria3)
                        offset += nctria3
                        ctria3_oyy_mean[i] = data_array[offset: offset + nctria3].reshape(nctria3)
                        offset += nctria3
                        ctria3_txy_mean[i] = data_array[offset: offset + nctria3].reshape(nctria3)
                        offset += nctria3

                        grid_x[i] = data_array[offset: offset + ngrid].reshape(ngrid)
                        offset += ngrid
                        grid_y[i] = data_array[offset: offset + ngrid].reshape(ngrid)
                        offset += ngrid
                        grid_z[i] = data_array[offset: offset + ngrid].reshape(ngrid)
                        offset += ngrid

                    logging.info(f"P04: Loading data from cache: {filename_op1}")
            except :
                logging.error(f"P04: Failed to read cache file {filename_op1}: {e}")
                return None

        # op1파일 없음. op2 최초 읽기
        else:
            try:
                start_time = time.time()
                logging.info(f"P04: Cache not found. Loading OP2 file for the first time: {filename}")
                op2 = read_op2(filename, build_dataframe=True, debug = False, log=None)
                logging.info(f"P04: OP2 file loaded successfully in {time.time() - start_time:.2f} seconds.")

            except FileNotFoundError:
                logging.error(f"P04: OP2 file not found: {filename}")
                return None

            # subcase 및 요소수 
            if mode_data == 1:
                nSubcases = 1
            else:
                nSubcases = len(op2.op2_results.stress.cquad4_stress)
            first_subcase_id = next(iter(op2.op2_results.stress.cquad4_stress))
            cquad4_element_map = {}
            ctria3_element_map = {}

            eids = op2.op2_results.stress.cquad4_stress[first_subcase_id].element_node[:,0]
            for idx, eid in enumerate(eids[1::2]):
                cquad4_element_map[eid] = idx
            eids = op2.op2_results.stress.ctria3_stress[first_subcase_id].element_node[:,0]
            for idx, eid in enumerate(eids[1::2]):
                ctria3_element_map[eid] = idx

            grid_ids = op2.displacements[first_subcase_id].node_gridtype[:,0]
            grid_map ={}
            for idx, gid in enumerate(grid_ids):
                grid_map[gid] = idx

            ngrid = len(grid_map)
            ncquad4 = len(cquad4_element_map)
            nctria3 = len(ctria3_element_map)

            data_to_save = {
                'ncquad4': ncquad4,
                'nctria3': nctria3,
                'nSubcases': nSubcases,
                'ngrid' : ngrid, 
                'grid_map': grid_map,
                'cquad4_element_map': cquad4_element_map,
                'ctria3_element_map': ctria3_element_map
            }

            # 응력 데이터 저장하기 
            logging.info(f"P04: Creating new cache file: {filename_op1}")

            cquad4_oxx_mean = np.empty((nSubcases,ncquad4), dtype=np.float32)
            cquad4_oyy_mean = np.empty((nSubcases,ncquad4), dtype=np.float32)
            cquad4_txy_mean = np.empty((nSubcases,ncquad4), dtype=np.float32)

            ctria3_oxx_mean = np.empty((nSubcases,nctria3), dtype=np.float32)
            ctria3_oyy_mean = np.empty((nSubcases,nctria3), dtype=np.float32)
            ctria3_txy_mean = np.empty((nSubcases,nctria3), dtype=np.float32)

            grid_x = np.empty((nSubcases, ngrid), dtype=np.float32)
            grid_y = np.empty((nSubcases, ngrid), dtype=np.float32)
            grid_z = np.empty((nSubcases, ngrid), dtype=np.float32)

            with open(filename_op1, 'wb') as f:
                # pickle 데이터 저장
                pickle.dump(data_to_save,f)

                for subcase_i in range(nSubcases):
                    subcase_id = subcase_i + 1

                    stress_data = op2.op2_results.stress.cquad4_stress[subcase_id].data_frame.to_numpy().astype(np.float32) #index(0),thick(1),xx(2),yy(3),xy(4)
                    cquad4_oxx_mean[subcase_i] = 0.5*(stress_data[0::2,2] + stress_data[1::2,2])
                    cquad4_oyy_mean[subcase_i] = 0.5*(stress_data[0::2,3] + stress_data[1::2,3])
                    cquad4_txy_mean[subcase_i] = 0.5*(stress_data[0::2,4] + stress_data[1::2,4])

                    f.write(cquad4_oxx_mean[subcase_i].tobytes())
                    f.write(cquad4_oyy_mean[subcase_i].tobytes())
                    f.write(cquad4_txy_mean[subcase_i].tobytes())

                    stress_data = op2.op2_results.stress.ctria3_stress[subcase_id].data_frame.to_numpy().astype(np.float32) #index(0),thick(1),xx(2),yy(3),xy(4)
                    ctria3_oxx_mean[subcase_i] = 0.5*(stress_data[0::2,2] + stress_data[1::2,2])
                    ctria3_oyy_mean[subcase_i] = 0.5*(stress_data[0::2,3] + stress_data[1::2,3])
                    ctria3_txy_mean[subcase_i] = 0.5*(stress_data[0::2,4] + stress_data[1::2,4])

                    f.write(ctria3_oxx_mean[subcase_i].tobytes())
                    f.write(ctria3_oyy_mean[subcase_i].tobytes())
                    f.write(ctria3_txy_mean[subcase_i].tobytes())

                    displacement_data = op2.displacements[subcase_id].data[0]  #data[0]에 [ngrid, 6]크기로 정의됨.
                    grid_x[subcase_i] = displacement_data[:,0].astype(np.float32)
                    grid_y[subcase_i] = displacement_data[:,1].astype(np.float32)
                    grid_z[subcase_i] = displacement_data[:,2].astype(np.float32)

                    f.write(grid_x[subcase_i].tobytes())
                    f.write(grid_y[subcase_i].tobytes())
                    f.write(grid_z[subcase_i].tobytes())

        # 변환된 결과        
        if mode_data == 1:        
            result_id = [1]
        elif lc_name =='full':
            self.full_data = [item for item in mode_data.list if item.split('/')[3] == 'full']
            full_mode_ha_index = [self.convert_to_index(int(value.split('/')[0]), self.config['ha']) for value in self.full_data]
            full_mode_freq_index = [self.convert_to_index(float(value.split('/')[1]), self.config['freq']) for value in self.full_data]
            result_id = [(full_mode_ha_index[i]) * mode_data.no_freq + full_mode_freq_index[i] for i in range(len(full_mode_ha_index))]           
        elif lc_name =='min':
            self.min_data = [item for item in mode_data.list if item.split('/')[3] == 'min']
            min_mode_ha_index = [self.convert_to_index(int(value.split('/')[0]), self.config['ha']) for value in self.min_data]
            min_mode_freq_index = [self.convert_to_index(float(value.split('/')[1]), self.config['freq']) for value in self.min_data]
            result_id = [(min_mode_ha_index[i]) * mode_data.no_freq + min_mode_freq_index[i] for i in range(len(min_mode_ha_index))]
        else:
            logging.warning(f"P04: Unknown lc_name '{lc_name}' found. Skipping this case.")

        subcase_stress_all = []   

        for subcase_id in result_id:
            subcase_i = subcase_id -1
            ei = 0
            #subcase_stress = {'elem_all': [], 'map_elem':{}}
            subcase_stress = {
                'elem_all': [], 
                'map_elem':{},
                'grid_all': [],
                'map_grid':{}
                }
            for elem_id, elem_type in read_elem:
                try:
                    if elem_type == 'CQUAD4':
                        idx = cquad4_element_map[elem_id]
                        oxx = cquad4_oxx_mean[subcase_i][idx]
                        oyy = cquad4_oyy_mean[subcase_i][idx]
                        txy = cquad4_txy_mean[subcase_i][idx]                    
                    elif elem_type == 'CTRIA3':
                        idx = ctria3_element_map[elem_id]
                        oxx = ctria3_oxx_mean[subcase_i][idx]
                        oyy = ctria3_oyy_mean[subcase_i][idx]
                        txy = ctria3_txy_mean[subcase_i][idx]                    
                    else:
                        continue
                except ValueError:
                    continue                    

                subcase_stress['map_elem'][elem_id] = ei
                ei += 1
                elem_stress = {
                    'id' : elem_id,
                    'sxx' : oxx,
                    'syy' : oyy,
                    'sxy' : txy
                }
                subcase_stress["elem_all"].append(elem_stress)

            ni = 0
            #subcase_displacement = {'grid_all': [], 'map_grid': []}
            for id in read_grid:
                idx = grid_map[id]
                t1 = grid_x[subcase_i][idx]
                t2 = grid_y[subcase_i][idx]
                t3 = grid_z[subcase_i][idx]

                subcase_stress['map_grid'][id] = ni
                ni += 1
                grid_displacement={
                    'id': id,
                    't1': t1,
                    't2': t2,
                    't3' : t3
                }
                subcase_stress['grid_all'].append(grid_displacement)


            subcase_stress_all.append(subcase_stress)
        logging.info(f"P04: Processed {len(subcase_stress_all)} subcases successfully.")

        return subcase_stress_all    

     # 변환 함수
    def convert_to_index(self, value, config_list):
        try:
            return config_list.index(value)  # config_list에서 value의 인덱스를 반환
        except ValueError:
            return -1  # value가 config_list에 없는 경우 -1 반환 (예외 처리)
        
#     def read_op2_x_data(self, filename, lc_name, read_elem):
#         filename_xsn = filename[:-3] + 'xsn'

#         # xsn파일 존재
#         if os.path.isfile(filename_xsn):

#             try:
#                 with open(filename_xsn, 'rb') as f:
#                     data_to_load = pickle.load(f)
#                     nSubcases = data_to_load['nSubcases']
#                     ncquad4 = data_to_load['ncquad4']
#                     nctria3 = data_to_load['nctria3']
#                     cquad4_element_map = data_to_load['cquad4_element_map']
#                     ctria3_element_map = data_to_load['ctria3_element_map']

#                     # 데이터 읽기
#                     data = f.read()
#                     #데이터 배열로 변환
#                     data_array = np.frombuffer(data, dtype = np.float32)

#                     cquad4_oxx_mean = np.empty((nSubcases, ncquad4), dtype=np.float32)
# #                     cquad4_oyy_mean = np.empty((nSubcases, ncquad4), dtype=np.float32)
# #                     cquad4_txy_mean = np.empty((nSubcases, ncquad4), dtype=np.float32)

#                     ctria3_oxx_mean = np.empty((nSubcases, nctria3), dtype=np.float32)
# #                     ctria3_oyy_mean = np.empty((nSubcases, nctria3), dtype=np.float32)
# #                     ctria3_txy_mean = np.empty((nSubcases, nctria3), dtype=np.float32)

#                     offset = 0
#                     for i in range(nSubcases):
#                         cquad4_oxx_mean[i] = data_array[offset: offset + ncquad4].reshape(ncquad4)
#                         offset += ncquad4
# #                         cquad4_oyy_mean[i] = data_array[offset: offset + ncquad4].reshape(ncquad4)
# #                         offset += ncquad4
# #                         cquad4_txy_mean[i] = data_array[offset: offset + ncquad4].reshape(ncquad4)
# #                         offset += ncquad4

#                         ctria3_oxx_mean[i] = data_array[offset: offset + nctria3].reshape(nctria3)
#                         offset += nctria3
# #                         ctria3_oyy_mean[i] = data_array[offset: offset + nctria3].reshape(nctria3)
# #                         offset += nctria3
# #                         ctria3_txy_mean[i] = data_array[offset: offset + nctria3].reshape(nctria3)
# #                         offset += nctria3

#                     print(f'{filename_xsn}파일 읽음. ')
#             except :
#                 print(f'{filename} 열기 실패')
#                 return None

#         # xsn파일 없음. op2 최초 읽기
#         else:
#             try:
#                 start_time = time.time()
#                 now = datetime.now()
#                 formatted_time = now.strftime("%H:%M:%S")
#                 print(f'op2 로딩 시작 {formatted_time}')

#                 op2 = read_op2(filename, build_dataframe=True, debug = False, log=None)

#                 elapsed_time = time.time() - start_time    
#                 print(f'op2 파일 로딩: {elapsed_time/60:.1f}분')

#             except FileNotFoundError:
#                 print(f'{filename} 열기 실패')
#                 return None

# #             # subcase 및 요소수 
# #             if mode_data == 1:
# #                 nSubcases = 1
# #             else:
#             nSubcases = len(op2.op2_results.stress.cquad4_stress)
#             print(nSubcases)
#             first_subcase_id = next(iter(op2.op2_results.stress.cquad4_stress))
#             cquad4_element_map = {}
#             ctria3_element_map = {}

#             eids = op2.op2_results.stress.cquad4_stress[first_subcase_id].element_node[:,0]
#             for idx, eid in enumerate(eids[1::2]):
#                 cquad4_element_map[eid] = idx
#             eids = op2.op2_results.stress.ctria3_stress[first_subcase_id].element_node[:,0]
#             for idx, eid in enumerate(eids[1::2]):
#                 ctria3_element_map[eid] = idx

#             ncquad4 = len(cquad4_element_map)
#             nctria3 = len(ctria3_element_map)

#             data_to_save = {
#                 'ncquad4': ncquad4,
#                 'nctria3': nctria3,
#                 'nSubcases': nSubcases,                
#                 'cquad4_element_map': cquad4_element_map,
#                 'ctria3_element_map': ctria3_element_map
#             }

#             # 응력 데이터 저장하기 
#             print(filename_xsn + '  write')

#             cquad4_oxx_mean = np.empty((nSubcases,ncquad4), dtype=np.float32)
# #             cquad4_oyy_mean = np.empty((nSubcases,ncquad4), dtype=np.float32)
# #             cquad4_txy_mean = np.empty((nSubcases,ncquad4), dtype=np.float32)

#             ctria3_oxx_mean = np.empty((nSubcases,nctria3), dtype=np.float32)
# #             ctria3_oyy_mean = np.empty((nSubcases,nctria3), dtype=np.float32)
# #             ctria3_txy_mean = np.empty((nSubcases,nctria3), dtype=np.float32)

#             with open(filename_xsn, 'wb') as f:
#                 # pickle 데이터 저장
#                 pickle.dump(data_to_save,f)

#                 for subcase_i in range(nSubcases):
#                     subcase_id = subcase_i + 1

#                     stress_data = op2.op2_results.stress.cquad4_stress[subcase_id].data_frame.to_numpy().astype(np.float32) #index(0),thick(1),xx(2),yy(3),xy(4)
#                     cquad4_oxx_mean[subcase_i] = 0.5*(stress_data[0::2,2] + stress_data[1::2,2])
# #                     cquad4_oyy_mean[subcase_i] = 0.5*(stress_data[0::2,3] + stress_data[1::2,3])
# #                     cquad4_txy_mean[subcase_i] = 0.5*(stress_data[0::2,4] + stress_data[1::2,4])

#                     f.write(cquad4_oxx_mean[subcase_i].tobytes())
# #                     f.write(cquad4_oyy_mean[subcase_i].tobytes())
# #                     f.write(cquad4_txy_mean[subcase_i].tobytes())

#                     stress_data = op2.op2_results.stress.ctria3_stress[subcase_id].data_frame.to_numpy().astype(np.float32) #index(0),thick(1),xx(2),yy(3),xy(4)
#                     ctria3_oxx_mean[subcase_i] = 0.5*(stress_data[0::2,2] + stress_data[1::2,2])
# #                     ctria3_oyy_mean[subcase_i] = 0.5*(stress_data[0::2,3] + stress_data[1::2,3])
# #                     ctria3_txy_mean[subcase_i] = 0.5*(stress_data[0::2,4] + stress_data[1::2,4])

#                     f.write(ctria3_oxx_mean[subcase_i].tobytes())
# #                     f.write(ctria3_oyy_mean[subcase_i].tobytes())
# #                     f.write(ctria3_txy_mean[subcase_i].tobytes())

# #         # 변환된 결과        
# #         if mode_data == 1:        
# #             result_id = [1]
# #         elif lc_name =='full':
# #             self.full_data = [item for item in mode_data.list if item.split('/')[3] == 'full']
# #             full_mode_ha_index = [self.convert_to_index(int(value.split('/')[0]), self.config['ha']) for value in self.full_data]
# #             full_mode_freq_index = [self.convert_to_index(float(value.split('/')[1]), self.config['freq']) for value in self.full_data]
# #             result_id = [(full_mode_ha_index[i]) * mode_data.no_freq + full_mode_freq_index[i] for i in range(len(full_mode_ha_index))]           
# #         elif lc_name =='min':
# #             self.min_data = [item for item in mode_data.list if item.split('/')[3] == 'min']
# #             min_mode_ha_index = [self.convert_to_index(int(value.split('/')[0]), self.config['ha']) for value in self.min_data]
# #             min_mode_freq_index = [self.convert_to_index(float(value.split('/')[1]), self.config['freq']) for value in self.min_data]
# #             result_id = [(min_mode_ha_index[i]) * mode_data.no_freq + min_mode_freq_index[i] for i in range(len(min_mode_ha_index))]
# #         else:
# #             print('Nothing')

#         subcase_stress_all = []   

#         for subcase_id in range(0,nSubcases):
#             subcase_i = subcase_id -1
#             ei = 0
#             #subcase_stress = {'elem_all': [], 'map_elem':{}}
#             subcase_stress = {
#                 'elem_all': [], 
#                 'map_elem':{}
#                 }
#             for elem_id, elem_type in read_elem:
#                 try:
#                     if elem_type == 'CQUAD4':
#                         idx = cquad4_element_map[elem_id]
#                         oxx = cquad4_oxx_mean[subcase_i][idx]
# #                         oyy = cquad4_oyy_mean[subcase_i][idx]
# #                         txy = cquad4_txy_mean[subcase_i][idx]                    
#                     elif elem_type == 'CTRIA3':
#                         idx = ctria3_element_map[elem_id]
#                         oxx = ctria3_oxx_mean[subcase_i][idx]
# #                         oyy = ctria3_oyy_mean[subcase_i][idx]
# #                         txy = ctria3_txy_mean[subcase_i][idx]                    
#                     else:
#                         continue
#                 except ValueError:
#                     continue                    

#                 subcase_stress['map_elem'][elem_id] = ei
#                 ei += 1
#                 elem_stress = {
#                     'id' : elem_id,
#                     'sxx' : oxx
# #                     'syy' : oyy,
# #                     'sxy' : txy
#                 }
#                 subcase_stress["elem_all"].append(elem_stress)
            
#             subcase_stress_all.append(subcase_stress)
#         print(f'subcase_stress_all count: {len(subcase_stress_all)}')

#         return subcase_stress_all         
    
#     def read_op2_ref_data(self, filename, lc_name, read_elem):
#         filename_ref = filename[:-3] + 'ref'

#         # ref파일 존재
#         if os.path.isfile(filename_ref):

#             try:
#                 with open(filename_ref, 'rb') as f:
#                     data_to_load = pickle.load(f)
#                     nSubcases = data_to_load['nSubcases']
#                     ncquad4 = data_to_load['ncquad4']
#                     nctria3 = data_to_load['nctria3']
#                     cquad4_element_map = data_to_load['cquad4_element_map']
#                     ctria3_element_map = data_to_load['ctria3_element_map']

#                     # 데이터 읽기
#                     data = f.read()
#                     #데이터 배열로 변환
#                     data_array = np.frombuffer(data, dtype = np.float32)

#                     cquad4_oxx_mean = np.empty((nSubcases, ncquad4), dtype=np.float32)
#                     cquad4_oyy_mean = np.empty((nSubcases, ncquad4), dtype=np.float32)
#                     cquad4_txy_mean = np.empty((nSubcases, ncquad4), dtype=np.float32)

#                     ctria3_oxx_mean = np.empty((nSubcases, nctria3), dtype=np.float32)
#                     ctria3_oyy_mean = np.empty((nSubcases, nctria3), dtype=np.float32)
#                     ctria3_txy_mean = np.empty((nSubcases, nctria3), dtype=np.float32)

#                     offset = 0
#                     for i in range(nSubcases):
#                         cquad4_oxx_mean[i] = data_array[offset: offset + ncquad4].reshape(ncquad4)
#                         offset += ncquad4
#                         cquad4_oyy_mean[i] = data_array[offset: offset + ncquad4].reshape(ncquad4)
#                         offset += ncquad4
#                         cquad4_txy_mean[i] = data_array[offset: offset + ncquad4].reshape(ncquad4)
#                         offset += ncquad4

#                         ctria3_oxx_mean[i] = data_array[offset: offset + nctria3].reshape(nctria3)
#                         offset += nctria3
#                         ctria3_oyy_mean[i] = data_array[offset: offset + nctria3].reshape(nctria3)
#                         offset += nctria3
#                         ctria3_txy_mean[i] = data_array[offset: offset + nctria3].reshape(nctria3)
#                         offset += nctria3

#                     print(f'{filename_ref}파일 읽음. ')
#             except :
#                 print(f'{filename} 열기 실패')
#                 return None

#         # ref파일 없음. op2 최초 읽기
#         else:
#             try:
#                 start_time = time.time()
#                 now = datetime.now()
#                 formatted_time = now.strftime("%H:%M:%S")
#                 print(f'op2 로딩 시작 {formatted_time}')

#                 op2 = read_op2(filename, build_dataframe=True, debug = False, log=None)

#                 elapsed_time = time.time() - start_time    
#                 print(f'op2 파일 로딩: {elapsed_time/60:.1f}분')

#             except FileNotFoundError:
#                 print(f'{filename} 열기 실패')
#                 return None

# #             # subcase 및 요소수 
# #             if mode_data == 1:
# #                 nSubcases = 1
# #             else:
#             nSubcases = len(op2.op2_results.stress.cquad4_stress)
#             print(nSubcases)
#             first_subcase_id = next(iter(op2.op2_results.stress.cquad4_stress))
#             cquad4_element_map = {}
#             ctria3_element_map = {}

#             eids = op2.op2_results.stress.cquad4_stress[first_subcase_id].element_node[:,0]
#             for idx, eid in enumerate(eids[1::2]):
#                 cquad4_element_map[eid] = idx
#             eids = op2.op2_results.stress.ctria3_stress[first_subcase_id].element_node[:,0]
#             for idx, eid in enumerate(eids[1::2]):
#                 ctria3_element_map[eid] = idx

#             ncquad4 = len(cquad4_element_map)
#             nctria3 = len(ctria3_element_map)

#             data_to_save = {
#                 'ncquad4': ncquad4,
#                 'nctria3': nctria3,
#                 'nSubcases': nSubcases,                
#                 'cquad4_element_map': cquad4_element_map,
#                 'ctria3_element_map': ctria3_element_map
#             }

#             # 응력 데이터 저장하기 
#             print(filename_ref + '  write')

#             cquad4_oxx_mean = np.empty((nSubcases,ncquad4), dtype=np.float32)
#             cquad4_oyy_mean = np.empty((nSubcases,ncquad4), dtype=np.float32)
#             cquad4_txy_mean = np.empty((nSubcases,ncquad4), dtype=np.float32)

#             ctria3_oxx_mean = np.empty((nSubcases,nctria3), dtype=np.float32)
#             ctria3_oyy_mean = np.empty((nSubcases,nctria3), dtype=np.float32)
#             ctria3_txy_mean = np.empty((nSubcases,nctria3), dtype=np.float32)

#             with open(filename_ref, 'wb') as f:
#                 # pickle 데이터 저장
#                 pickle.dump(data_to_save,f)

#                 for subcase_i in range(nSubcases):
#                     subcase_id = subcase_i + 1

#                     stress_data = op2.op2_results.stress.cquad4_stress[subcase_id].data_frame.to_numpy().astype(np.float32) #index(0),thick(1),xx(2),yy(3),xy(4)
#                     cquad4_oxx_mean[subcase_i] = 0.5*(stress_data[0::2,2] + stress_data[1::2,2])
#                     cquad4_oyy_mean[subcase_i] = 0.5*(stress_data[0::2,3] + stress_data[1::2,3])
#                     cquad4_txy_mean[subcase_i] = 0.5*(stress_data[0::2,4] + stress_data[1::2,4])

#                     f.write(cquad4_oxx_mean[subcase_i].tobytes())
#                     f.write(cquad4_oyy_mean[subcase_i].tobytes())
#                     f.write(cquad4_txy_mean[subcase_i].tobytes())

#                     stress_data = op2.op2_results.stress.ctria3_stress[subcase_id].data_frame.to_numpy().astype(np.float32) #index(0),thick(1),xx(2),yy(3),xy(4)
#                     ctria3_oxx_mean[subcase_i] = 0.5*(stress_data[0::2,2] + stress_data[1::2,2])
#                     ctria3_oyy_mean[subcase_i] = 0.5*(stress_data[0::2,3] + stress_data[1::2,3])
#                     ctria3_txy_mean[subcase_i] = 0.5*(stress_data[0::2,4] + stress_data[1::2,4])

#                     f.write(ctria3_oxx_mean[subcase_i].tobytes())
#                     f.write(ctria3_oyy_mean[subcase_i].tobytes())
#                     f.write(ctria3_txy_mean[subcase_i].tobytes())

# #         # 변환된 결과        
# #         if mode_data == 1:        
# #             result_id = [1]
# #         elif lc_name =='full':
# #             self.full_data = [item for item in mode_data.list if item.split('/')[3] == 'full']
# #             full_mode_ha_index = [self.convert_to_index(int(value.split('/')[0]), self.config['ha']) for value in self.full_data]
# #             full_mode_freq_index = [self.convert_to_index(float(value.split('/')[1]), self.config['freq']) for value in self.full_data]
# #             result_id = [(full_mode_ha_index[i]) * mode_data.no_freq + full_mode_freq_index[i] for i in range(len(full_mode_ha_index))]           
# #         elif lc_name =='min':
# #             self.min_data = [item for item in mode_data.list if item.split('/')[3] == 'min']
# #             min_mode_ha_index = [self.convert_to_index(int(value.split('/')[0]), self.config['ha']) for value in self.min_data]
# #             min_mode_freq_index = [self.convert_to_index(float(value.split('/')[1]), self.config['freq']) for value in self.min_data]
# #             result_id = [(min_mode_ha_index[i]) * mode_data.no_freq + min_mode_freq_index[i] for i in range(len(min_mode_ha_index))]
# #         else:
# #             print('Nothing')

#         subcase_stress_all = []   

#         for subcase_id in range(0,nSubcases):
#             subcase_i = subcase_id -1
#             ei = 0
#             #subcase_stress = {'elem_all': [], 'map_elem':{}}
#             subcase_stress = {
#                 'elem_all': [], 
#                 'map_elem':{}
#                 }
#             for elem_id, elem_type in read_elem:
#                 try:
#                     if elem_type == 'CQUAD4':
#                         idx = cquad4_element_map[elem_id]
#                         oxx = cquad4_oxx_mean[subcase_i][idx]
#                         oyy = cquad4_oyy_mean[subcase_i][idx]
#                         txy = cquad4_txy_mean[subcase_i][idx]                    
#                     elif elem_type == 'CTRIA3':
#                         idx = ctria3_element_map[elem_id]
#                         oxx = ctria3_oxx_mean[subcase_i][idx]
#                         oyy = ctria3_oyy_mean[subcase_i][idx]
#                         txy = ctria3_txy_mean[subcase_i][idx]                    
#                     else:
#                         continue
#                 except ValueError:
#                     continue                    

#                 subcase_stress['map_elem'][elem_id] = ei
#                 ei += 1
#                 elem_stress = {
#                     'id' : elem_id,
#                     'sxx' : oxx,
#                     'syy' : oyy,
#                     'sxy' : txy
#                 }
#                 subcase_stress["elem_all"].append(elem_stress)
            
#             subcase_stress_all.append(subcase_stress)
#         print(f'subcase_stress_all count: {len(subcase_stress_all)}')

#         return subcase_stress_all           
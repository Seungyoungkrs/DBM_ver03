import pandas as pd
import numpy as np
import os
import pickle
from module.p00_config import  get_configuration

## P03 FE 모델 로더
class FEModelLoader:
    def __init__(self, config):
        self.file_paths = config['filepaths']
        self.sensor_ids = config['sensors']

    def load_file(self, filepath):
        with open(filepath, 'r') as file:
            return file.readlines()

    def get_element_data(self, model_data, elem_ids):
        elem_data = []
        id_set = set(map(str, elem_ids))
        for line in model_data:
            parts = line.split()
            if len(parts) > 5 and parts[1] in id_set:
                if parts[0] == "CQUAD4":
                    elem_data.append([parts[1], parts[3], parts[4], parts[5], parts[6]])
                elif parts[0] == "CTRIA3":
                    elem_data.append([parts[1], parts[3], parts[4], parts[5], parts[5]])
        return elem_data

    def parse_node_data(self, model_data):
        node_dict = {}
        for line in model_data:
            if line.startswith("GRID"):
                node_id = line[8:16].strip()  # 노드 ID 추출
                coord_x = self.format_scientific(line[24:32].strip())
                coord_y = self.format_scientific(line[32:40].strip())
                coord_z = self.format_scientific(line[40:48].strip())

                try:
                    # 좌표를 실수로 변환하고 저장
                    coords = [float(coord_x), float(coord_y), float(coord_z)]
                    node_dict[node_id] = coords
                except ValueError as e:
                    print(f"Error converting coordinates for node {node_id}: {e}")

        return node_dict

    def format_scientific(self, coord):
        # 지수 표현이 있을 경우 이를 올바른 형태로 변환
        if '-' in coord[1:] or '+' in coord[1:]:
            # 마지막 - 또는 + 기호의 위치를 찾아 그 앞은 숫자, 뒤는 지수로 처리
            last_index = max(coord.rfind('-'), coord.rfind('+'), 1)
            number = coord[:last_index]
            exponent = coord[last_index:]
            coord = f"{number}e{exponent}"
        return coord

    def get_node_coordinates(self, elem_data, node_dict):
        detailed_node_coords = []
        for elem in elem_data:
            node_ids = elem[1:]
            coords_list = [node_dict.get(node_id) for node_id in node_ids if node_id in node_dict]
            if all(coords_list):
                if coords_list[2] == coords_list[3]:
                    individual_coords = [coord for sublist in coords_list for coord in sublist]
                    avg_coords = [sum(coords) / 3 for coords in zip(*coords_list)]
                    detailed_node_coords.append([elem[0]] + individual_coords + avg_coords)
                else:
                    individual_coords = [coord for sublist in coords_list for coord in sublist]
                    avg_coords = [sum(coords) / 4 for coords in zip(*coords_list)]
                    detailed_node_coords.append([elem[0]] + individual_coords + avg_coords)
        return detailed_node_coords

    def get_node_data(self, ref_data):
        """
        파일 내용(리스트)에서 노드 ID를 추출합니다.

        Parameters:
            ref_data (list): 파일의 각 줄이 포함된 리스트

        Returns:
            list: 추출된 노드 ID 리스트
        """
        node_ids = []  # 노드 ID를 저장할 리스트

        # 각 줄에서 "GRID"로 시작하는 데이터 처리
        for line in ref_data:
            if line.startswith("GRID"):  # "GRID"로 시작하는 줄만 처리
                node_id = line[8:16].strip()  # 노드 ID가 8~16번째 문자에 위치
                node_ids.append(int(node_id))  # 노드 ID를 정수로 변환 후 추가

        return node_ids  # 모든 노드 ID를 반환

    
    def parse_range(self, text):
        result = []
        elements = text.split()

        for elem in elements:
            if elem.isdigit():  # 숫자인 경우에만 처리
                result.append(int(elem))
            elif ':' in elem:  # 범위가 콜론을 포함하는 경우
                if elem.count(':') == 1:  # 일반 범위
                    start, end = map(int, elem.split(':'))
                    result.extend(range(start, end + 1))
                elif elem.count(':') == 2:  # 특수 범위 (간격이 있는 범위)
                    start, end, step = map(int, elem.split(':'))
                    result.extend(range(start, end + 1, step))
        return result

    def load_model(self):
        model_data = self.load_file(self.file_paths['data_file'])
        ref_file_path = self.file_paths['ref_file']
        ref_data = self.load_file(ref_file_path)

        if ref_file_path.endswith('.bdf'):
            ref_df = self.parse_bdf(ref_data)
        else:
            ref_df = self.parse_txt(ref_data)

        ref_ids = list(ref_df['ElemID'].astype(int))
        elem_sen = self.get_element_data(model_data, self.sensor_ids)
        elem_ref = self.get_element_data(model_data, ref_ids)
        node_dict = self.parse_node_data(model_data)
        sen_coordinates = self.get_node_coordinates(elem_sen, node_dict)
        ref_coordinates = self.get_node_coordinates(elem_ref, node_dict)

       # 노드 참조 데이터 추가
        node_input_ref = self.get_node_data(ref_data)
        
        return elem_sen, elem_ref, sen_coordinates, ref_coordinates, ref_ids, ref_df, node_input_ref


    def parse_bdf(self, ref_data):
        # 결과를 저장할 빈 리스트를 생성합니다.
        elem_ids = []
        stations = []

        for line in ref_data:
            parts = line.split()
            if parts and (parts[0] == "CQUAD4" or parts[0] == "CTRIA3"):
                elem_id = parts[1]
                station = 0  # 기본 스테이션 값, 필요시 추가 정보를 사용하여 수정
                elem_ids.append(elem_id)
                stations.append(station)

        # elem_ids와 stations 리스트를 사용하여 DataFrame 생성
        ref_df = pd.DataFrame({
            'ElemID': elem_ids,
            'Station': stations
        })

        return ref_df

    def parse_txt(self, ref_data):
        # 기존 txt 파일 처리 로직
        elem_ids = []
        stations = []

        for line in ref_data[1:]:
            if line.strip():
                if not line.startswith('!'):
                    split_line = line.split()
                    elem_ids.append(split_line[0])
                    stations.append(split_line[1])

        ref_df = pd.DataFrame({
            'ElemID': elem_ids,
            'Station': stations
        })

        return ref_df

    
    def set_node_elem_list(self, elem_sen, elem_ref, sen_coordinates, ref_coordinates):
        elem_sen_df = pd.DataFrame(elem_sen)
        node_sen_df = pd.DataFrame(sen_coordinates).drop(0, axis=1)
        ElemNode_sen_df = pd.concat([elem_sen_df, node_sen_df], axis=1)
        ElemNode_sen_df.columns = [
            "ElemNo", "Node1", "Node2", "Node3", "Node4",
            "N1x", "N1y", "N1z", "N2x", "N2y", "N2z", "N3x", "N3y", "N3z", "N4x", "N4y", "N4z",
            "coX", "coY", "coZ"
        ]

        elem_ref_df = pd.DataFrame(elem_ref)
        node_ref_df = pd.DataFrame(ref_coordinates).drop(0, axis=1)
        ElemNode_ref_df = pd.concat([elem_ref_df, node_ref_df], axis=1)
        ElemNode_ref_df.columns = [
            "ElemNo", "Node1", "Node2", "Node3", "Node4",
            "N1x", "N1y", "N1z", "N2x", "N2y", "N2z", "N3x", "N3y", "N3z", "N4x", "N4y", "N4z",
            "coX", "coY", "coZ"
        ]
        
        Elem_input_sensor = self.sensor_ids
        Elem_input_ref = list(map(int, (ElemNode_ref_df["ElemNo"])))
        Elem_no = [len(Elem_input_ref), len(Elem_input_sensor)]
        
        return ElemNode_ref_df, ElemNode_sen_df, Elem_no, Elem_input_ref, Elem_input_sensor    
    
    def load_fe_model_data(self):
        elem_sen, elem_ref, sen_coordinates, ref_coordinates, ref_ids, ref_df, node_input_ref = self.load_model()
        ElemNode_ref_df, ElemNode_sen_df, Elem_no, Elem_input_ref, Elem_input_sensor = self.set_node_elem_list(
            elem_sen, elem_ref, sen_coordinates, ref_coordinates)
        
        return ElemNode_ref_df, ElemNode_sen_df, Elem_no, ref_ids, Elem_input_sensor, ref_df, node_input_ref
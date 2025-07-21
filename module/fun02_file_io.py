import pickle
import h5py   
import numpy as np  
import logging
from tqdm import tqdm 

def save_data(variables, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(variables, file)

def load_data(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)

def load_hdf5_data(hdf5_file, group_name, sub_group_name):
    """HDF5 파일에서 특정 그룹과 서브 그룹의 데이터를 로드하여 리스트 형태로 반환합니다."""
    with h5py.File(hdf5_file, 'r') as f:
        group = f[group_name][sub_group_name]
        table_data = group['table'][:]
        data = []
        for row in tqdm(table_data, desc=f"Loading {group_name}/{sub_group_name} hdf5 data"):
            data.append([
                row['ID'],
                row['Type'].decode('utf-8'),
                row['Value'],
                row['Description'].decode('utf-8'),
                row['Col5'].decode('utf-8') if isinstance(row['Col5'], bytes) else row['Col5'],
                row['Col6'],
                row['Metric1'],
                row['Metric2'],
                row['Metric3'],
                row['Metric4'],
                row['Metric5'],
                row['Metric6']
            ])
        return data    

def load_ref_data(file_path, pickle_file):    
    # 'full/Static', 'full/Real', 'full/Imag' 그룹의 데이터를 로드
    full_static_data = load_hdf5_data(file_path, 'full', 'Static') 
    full_real_data = load_hdf5_data(file_path, 'full', 'Real')
    full_imag_data = load_hdf5_data(file_path, 'full', 'Imag')
    
    # 'min/Static', 'min/Real', 'min/Imag' 그룹의 데이터를 로드
    min_static_data = load_hdf5_data(file_path, 'min', 'Static')
    min_real_data = load_hdf5_data(file_path, 'min', 'Real')
    min_imag_data = load_hdf5_data(file_path, 'min', 'Imag')

    # 초기 results 딕셔너리 생성
    ref_results = {
        'full': {
            'Static': full_static_data,
            'Real': full_real_data,
            'Imag': full_imag_data
        },
        'min': {
            'Static': min_static_data,
            'Real': min_real_data,
            'Imag': min_imag_data
        }
    }
    GlStsMode_refMat_df = load_data('./Output/pkl/GlStsMode_refMat_df.pkl')
       
        
    return ref_results, GlStsMode_refMat_df
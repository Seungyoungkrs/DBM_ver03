#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import math
import gc
import os
import pickle
from module.p00_config import  get_configuration
from module.fun01_data_processing import arr_load, st_load, riTrans

## P01 유체동역학(ISTAS V2) 계산 데이터 처리
class HydrodynamicDataProcessor:
    def __init__(self, config):
        self.filepaths = config['filepaths']
        self.stations = config['stations']
        self.nPhase = config['nPhase']
        self.data_dfs = {'full': {'vbm': [], 'hbm': [], 'tm': []},
                         'min': {'vbm': [], 'hbm': [], 'tm': []}}
        self.freq = None
        self.final_vbm_df = pd.DataFrame()
        self.final_hbm_df = pd.DataFrame()
        self.final_tm_df = pd.DataFrame()        

    def read_hydrodynamic_data(self, filepath):
        """HYDRODYNAMIC.OUT 파일에서 데이터를 읽습니다."""
        try:
            with open(filepath, 'r') as file:
                data_lines = file.readlines()
            return data_lines
        except FileNotFoundError:
            print(f"Error: 파일 {filepath}이 존재하지 않습니다.")
            return []
        except Exception as e:
            print(f"오류 발생: {e}")
            return []    
    
    def extract_dimension_data(self, data_lines):
        """읽어온 데이터 리스트에서 차원 데이터를 추출"""
        rho = float(data_lines[1][33:46])
        lbp = float(data_lines[3][43:56])
        breadth = float(data_lines[4][10:23])
        v_s = float(data_lines[4][30:43]) * 0.51444444  # 노트를 m/s로 변환
        cogx = float(data_lines[9][11:24])
        cogy = float(data_lines[9][23:37])
        cogz = float(data_lines[9][36:50])
        d_ap = float(data_lines[3][7:20])
        d_fp = float(data_lines[3][25:38])
        no_st = int(data_lines[2][49:51])
        d_lcg_wl = d_ap + (d_fp - d_ap) * (cogx - lbp) / (0 - lbp)

        dim_data = {
            "rho": rho,
            "lbp": lbp,
            "breadth": breadth,
            "v_s": v_s,
            "cogx": cogx,
            "cogy": cogy,
            "cogz": cogz,
            "d_ap": d_ap,
            "d_fp": d_fp,
            "no_st": no_st,
            "d_lcg_wl": d_lcg_wl
        }
        return dim_data
    
    def extract_load_data(self, raw_data_list, st1, st2, st3, dim_data):
        """읽어온 데이터 리스트에서 하중 데이터 추출"""
        tm1, tm2, tm3 =[],[],[]
        vbm, hbm  =[],[]
        vbm_all_st,hbm_all_st,tm_all_st =[],[],[]
        no_st = dim_data.get('no_st', 0)
        i=1
        for one_line in raw_data_list:
            # Freq 읽기
            if one_line[41:52] == "INCIDENCE= ":
                r1 = raw_data_list[i+2:i+8]
                ha = float(one_line[53:65])
                period = float(one_line[28:41])
                freq = round(2*math.pi/period,5)

            # TM
            if (one_line[0:12] == "TORSIONAL MT"):
                t1, t2, t3 = raw_data_list[i+st1], raw_data_list[i+st2], raw_data_list[i+st3]              
                tm1.append(arr_load(t1, freq, ha))
                tm2.append(arr_load(t2, freq, ha))
                tm3.append(arr_load(t3, freq, ha))

                r1 = raw_data_list[i+1:i+no_st+1]  # 모든 station
                for data in r1:
                    tm_all_st.append(st_load(data, freq, ha))     

            #VBM 
            if (one_line[0:9] == " PITCH MT"):
                vm = arr_load(raw_data_list[i+st2], freq, ha)
                vbm.append(vm)

                r1 = raw_data_list[i+1:i+no_st+1]  # 모든 station
                for data in r1:
                    vbm_all_st.append(st_load(data, freq, ha))     
            #HBM                
            if (one_line[0:7] == " YAW MT"):
                hm = arr_load(raw_data_list[i+st2], freq, ha)
                hbm.append(hm)

                r1 = raw_data_list[i+1:i+no_st+1]  # 모든 station
                for data in r1:
                    hbm_all_st.append(st_load(data, freq, ha))                
            i += 1

        # 처리된 데이터 반환
        return tm1, tm2, tm3, vbm, hbm, vbm_all_st, hbm_all_st, tm_all_st        
 
    def process_load_data(self, tm1, tm2, tm3, vbm, hbm, vbm_all_st, hbm_all_st, tm_all_st, dim_data, LC_name):
        config = get_configuration()
        ha = config['ha']   # 0도에서 180도까지 15도 간격
        nPhase = config['nPhase']  # 단계 수

        # 주어진 데이터와 차원 데이터에서 필요한 값 추출    
        V_s = dim_data.get('v_s', 0)
        rho = dim_data.get('rho', 0)
        Lbp = dim_data.get('lbp', 0)
        no_st = dim_data.get('no_st', 0)

        # 특정스테이션 Load 
        tm1raw_df = pd.DataFrame(tm1)
        tm1raw_df.columns = ['Heading Angle', 'Frequency', 'Real', 'Imag', 'Magnitude', 'Phase']
        tm2raw_df = pd.DataFrame(tm2)
        tm2raw_df.columns = ['Heading Angle', 'Frequency', 'Real', 'Imag', 'Magnitude', 'Phase']
        tm3raw_df = pd.DataFrame(tm3)
        tm3raw_df.columns = ['Heading Angle', 'Frequency', 'Real', 'Imag', 'Magnitude', 'Phase']
        vbmraw_df = pd.DataFrame(vbm)
        vbmraw_df.columns = ['Heading Angle', 'Frequency', 'Real', 'Imag', 'Magnitude', 'Phase']
        hbmraw_df = pd.DataFrame(hbm)
        hbmraw_df.columns = ['Heading Angle', 'Frequency', 'Real', 'Imag', 'Magnitude', 'Phase']

        # 길이방향 Load
        vbmall_df = pd.DataFrame(vbm_all_st)
        vbmall_df.columns = ['Heading Angle', 'Frequency', 'Station','Real', 'Imag', 'Magnitude', 'Phase']
        vbmall_df.fillna(0, inplace=True)          # 1번 staticn에 결측치가 간혹 생기므로 이 부분을 0으로 채움
        hbmall_df = pd.DataFrame(hbm_all_st)
        hbmall_df.columns = ['Heading Angle', 'Frequency', 'Station','Real', 'Imag', 'Magnitude', 'Phase']
        hbmall_df.fillna(0, inplace=True)
        tmall_df = pd.DataFrame(tm_all_st)
        tmall_df.columns = ['Heading Angle', 'Frequency', 'Station','Real', 'Imag', 'Magnitude', 'Phase']
        tmall_df.fillna(0, inplace=True)

        # Frequency 정의
        HA0 = vbmraw_df.loc[(vbmraw_df['Heading Angle'] == 0)]

        temp_freq = pd.DataFrame(HA0.Frequency)
        temp_freq = temp_freq.reset_index()    
        temp_freq = temp_freq.drop(columns=['index'])
        EFreq_all = temp_freq
        tm1r_all,tm1i_all = temp_freq,temp_freq
        tm2r_all,tm2i_all = temp_freq,temp_freq
        tm3r_all,tm3i_all = temp_freq,temp_freq
        vbmr_all,vbmi_all = temp_freq,temp_freq
        hbmr_all,hbmi_all = temp_freq,temp_freq

        for i in ha:
            tm1_raw = tm1raw_df.loc[(tm1raw_df['Heading Angle'] == i)]
            tm2_raw = tm2raw_df.loc[(tm2raw_df['Heading Angle'] == i)]
            tm3_raw = tm3raw_df.loc[(tm3raw_df['Heading Angle'] == i)]
            vbm_raw = vbmraw_df.loc[(vbmraw_df['Heading Angle'] == i)]
            hbm_raw = hbmraw_df.loc[(hbmraw_df['Heading Angle'] == i)]    
            Freq = vbm_raw['Frequency']    
            tm1r_all,tm1i_all = riTrans(tm1_raw, tm1r_all, tm1i_all, i)
            tm2r_all,tm2i_all = riTrans(tm2_raw, tm2r_all, tm2i_all, i)
            tm3r_all,tm3i_all = riTrans(tm3_raw, tm3r_all, tm3i_all, i)
            vbmr_all,vbmi_all = riTrans(vbm_raw, vbmr_all, vbmi_all, i)
            hbmr_all,hbmi_all = riTrans(hbm_raw, hbmr_all, hbmi_all, i)

            #조우주파수
            EFreq = Freq - Freq ** 2 / 9.81 * V_s * math.cos(i*math.pi/180)        
            EFreq = EFreq.reset_index()    
            EFreq = EFreq.drop(columns=['index'])
            EFreq.columns = ['{} deg'.format(i)]
            EFreq_all = pd.concat([EFreq_all, EFreq],axis=1)    

        Freq_temp = EFreq_all.Frequency
        Freq_dummy = EFreq_all.copy()
        for i in range(0,Freq_dummy.shape[1]):
            Freq_dummy.iloc[:,i] = Freq_temp

        # 무차원 계수
        tm1_all =  (tm1r_all ** 2 + tm1i_all ** 2) ** 0.5 * rho * EFreq_all ** 2 * Lbp ** 4
        tm2_all =  (tm2r_all ** 2 + tm2i_all ** 2) ** 0.5 * rho * EFreq_all ** 2 * Lbp ** 4
        tm3_all =  (tm3r_all ** 2 + tm3i_all ** 2) ** 0.5 * rho * EFreq_all ** 2 * Lbp ** 4
        vbm_all =  (vbmr_all ** 2 + vbmi_all ** 2) ** 0.5 * rho * EFreq_all ** 2 * Lbp ** 4
        hbm_all =  (hbmr_all ** 2 + hbmi_all ** 2) ** 0.5 * rho * EFreq_all ** 2 * Lbp ** 4

        # Frequency행 다시 정의
        tm1_all.Frequency =  EFreq_all.Frequency
        tm2_all.Frequency =  EFreq_all.Frequency
        tm3_all.Frequency =  EFreq_all.Frequency
        vbm_all.Frequency =  EFreq_all.Frequency
        hbm_all.Frequency =  EFreq_all.Frequency    

        # 길이방향 Load 무차원 계수
        for i in range(0,vbmall_df.shape[0]):
            EFreq_a = vbmall_df.iloc[i,1] - vbmall_df.iloc[i,1] ** 2 / 9.81 * V_s * math.cos(vbmall_df.iloc[i,0]*math.pi/180) 
            vbmall_df.iloc[i,3] = (vbmall_df.iloc[i,3]) * rho * EFreq_a ** 2 * Lbp ** 4
            hbmall_df.iloc[i,3] = (hbmall_df.iloc[i,3]) * rho * EFreq_a ** 2 * Lbp ** 4
            tmall_df.iloc[i,3] = (tmall_df.iloc[i,3]) * rho * EFreq_a ** 2 * Lbp ** 4
            vbmall_df.iloc[i,4] = (vbmall_df.iloc[i,4]) * rho * EFreq_a ** 2 * Lbp ** 4
            hbmall_df.iloc[i,4] = (hbmall_df.iloc[i,4]) * rho * EFreq_a ** 2 * Lbp ** 4
            tmall_df.iloc[i,4] = (tmall_df.iloc[i,4]) * rho * EFreq_a ** 2 * Lbp ** 4    
            vbmall_df.iloc[i,5] = (vbmall_df.iloc[i,5]) * rho * EFreq_a ** 2 * Lbp ** 4
            hbmall_df.iloc[i,5] = (hbmall_df.iloc[i,5]) * rho * EFreq_a ** 2 * Lbp ** 4
            tmall_df.iloc[i,5] = (tmall_df.iloc[i,5]) * rho * EFreq_a ** 2 * Lbp ** 4        

        nphase2 = int(nPhase / 2)

        # 각 결과를 저장할 리스트 초기화
        vbm_list, hbm_list, tm_list = [], [], []
        data_list, data_nor_list = [], []

        # 결과 저장을 위한 NumPy 배열 초기화
        vbm_results = np.zeros((no_st, len(ha) * len(Freq_temp) * nphase2))
        hbm_results = np.zeros_like(vbm_results)
        tm_results = np.zeros_like(vbm_results)
        data_results = np.zeros((no_st * 3, len(ha) * len(Freq_temp) * nphase2))
        data_nor_results = np.zeros_like(data_results)

        for i in range(1, len(ha) + 1):
            for j in range(1, len(Freq_temp) + 1):        
                a1 = no_st * (i - 1) + (j - 1) * len(ha) * no_st
                a2 = no_st * i + (j - 1) * len(ha) * no_st
                a, b, c = vbmall_df.iloc[a1:a2, :], hbmall_df.iloc[a1:a2, :], tmall_df.iloc[a1:a2, :]

                # 각 데이터 프레임에서 필요한 값을 NumPy 배열로 추출
                y1, z1 = a['Magnitude'].values, np.radians(a['Phase'].values)
                y2, z2 = b['Magnitude'].values, np.radians(b['Phase'].values)
                y3, z3 = c['Magnitude'].values, np.radians(c['Phase'].values)

                for ii in range(nphase2):
                    # 각 조건에 대한 계산 수행
                    sin_vals = np.sin(2 * np.pi * ii / nPhase + z1)
                    vbm_vals = y1 * sin_vals
                    hbm_vals = y2 * sin_vals
                    tm_vals = y3 * sin_vals

                    # 계산된 결과를 리스트에 저장
                    vbm_list.append(vbm_vals)
                    hbm_list.append(hbm_vals)
                    tm_list.append(tm_vals)                    

        # 계산이 완료된 후, 리스트의 내용을 DataFrame으로 변환
        vbm_df = pd.DataFrame(np.vstack(vbm_list).T)
        hbm_df = pd.DataFrame(np.vstack(hbm_list).T)
        tm_df = pd.DataFrame(np.vstack(tm_list).T)
        column_names = []
        
        for i in ha:
            for j in Freq_temp:
                for ii in range(1, nphase2 + 1):
                    column_name = '{}/{}/{}/{}'.format(i, j, ii, LC_name)
                    column_names.append(column_name)

        # 생성된 컬럼명을 DataFrame에 할당
        vbm_df.columns = column_names
        hbm_df.columns = column_names
        tm_df.columns = column_names
        freq =  list(Freq.iloc[:])

        return ha, freq, nPhase, vbm_df, hbm_df, tm_df
        
    def process_files(self):
        self.freq = None  # Initialize to ensure it captures the latest freq from processed files
        for file_type, filepath in self.filepaths.items():
            if 'hydrodynamic' in file_type:
                LC_name = file_type.split('_')[-1]
                if LC_name not in self.data_dfs:
                    self.data_dfs[LC_name] = {'vbm': [], 'hbm': [], 'tm': []}
                raw_data = self.read_hydrodynamic_data(filepath)
                if not raw_data:
                    print(f"No data in {filepath}")
                    continue
                dim_data = self.extract_dimension_data(raw_data)
                if not dim_data:
                    print(f"No dimension data in {filepath}")
                    continue
                load_params = self.extract_load_data(raw_data, *self.stations, dim_data)
                ha, file_freq, nPhase, vbm_df, hbm_df, tm_df = self.process_load_data(*load_params, dim_data, LC_name)
                self.freq = file_freq  # Set freq to the latest valid freq data

                if not vbm_df.empty:
                    self.data_dfs[LC_name]['vbm'].append(vbm_df)
                if not hbm_df.empty:
                    self.data_dfs[LC_name]['hbm'].append(hbm_df)
                if not tm_df.empty:
                    self.data_dfs[LC_name]['tm'].append(tm_df)
                print(f"Data processed for {LC_name} in {filepath}")
        self.concatenate_and_save()

    def concatenate_and_save(self):
        try:
            # VBM 데이터 프레임 결합
            final_vbm_df = pd.concat([self.data_dfs['full']['vbm'][0], self.data_dfs['min']['vbm'][0]], axis=1)
            final_vbm_df.to_csv('combined_full_min_vbm.csv')
            print("Combined VBM data saved.")

            # HBM 데이터 프레임 결합
            final_hbm_df = pd.concat([self.data_dfs['full']['hbm'][0], self.data_dfs['min']['hbm'][0]], axis=1)
            final_hbm_df.to_csv('combined_full_min_hbm.csv')
            print("Combined HBM data saved.")

            # TM 데이터 프레임 결합
            final_tm_df = pd.concat([self.data_dfs['full']['tm'][0], self.data_dfs['min']['tm'][0]], axis=1)
            final_tm_df.to_csv('combined_full_min_tm.csv')
            print("Combined TM data saved.")

            # 클래스 변수에 저장
            self.final_vbm_df = final_vbm_df
            self.final_hbm_df = final_hbm_df
            self.final_tm_df = final_tm_df

        except KeyError as e:
            print(f"Key error: {e} - Check the 'data_dfs' structure and ensure it contains the correct keys.")
        except Exception as e:
            print(f"An error occurred: {e}")

    def get_results(self):
        if self.freq is None or self.final_vbm_df.empty or self.final_hbm_df.empty or self.final_tm_df.empty:
            print("No data available to return. Check data processing steps.")
            return None
        return self.freq, self.final_vbm_df, self.final_hbm_df, self.final_tm_df


import pandas as pd
import numpy as np
import math
import gc
import os
import pickle
from module.p00_config import  get_configuration
from module.fun01_data_processing import arr_load, st_load, riTrans, loadTrans
from sklearn.cluster import AgglomerativeClustering

## P02 모드계산 및 선정
class ModeCalculator:
    def __init__(self, vbm_data_df, hbm_data_df, tm_data_df, n_modes):
        self.vbm_df = vbm_data_df
        self.hbm_df = hbm_data_df
        self.tm_df = tm_data_df
        self.n_modes = n_modes

    def fill_diagonal_with_nan(self, matrix):
        """Set the diagonal of the given square matrix to NaN."""
        np.fill_diagonal(matrix.values, np.nan)

    def calculate_max_correlation_excluding_self(self, correlation_matrix):
        """Calculate the maximum correlation excluding self-correlation."""
        self.fill_diagonal_with_nan(correlation_matrix)
        return np.nanmax(correlation_matrix.values)
    
    def generate_initial_modes(self):        
        # 상관관계 행렬 계산 및 절대값 변환
        vbm_corr = np.abs(self.vbm_df.corr())
        hbm_corr = np.abs(self.hbm_df.corr())
        tm_corr = np.abs(self.tm_df.corr())

        # 평균 상관관계 행렬 계산
        abs_corr_matrix = (vbm_corr + hbm_corr + tm_corr) / 3
        
        # 대각선 값을 매우 큰 값으로 설정하여 (자기 자신과의 상관관계는 제외)
        np.fill_diagonal(abs_corr_matrix.values, 1.0)
        
        # Agglomerative Clustering 수행
        num_clusters = self.n_modes
        agglom = AgglomerativeClustering(n_clusters=num_clusters, metric='euclidean', linkage='average')

        # 거리 행렬로 변환
        distance_matrix = 1 - abs_corr_matrix.values
        clusters = agglom.fit_predict(distance_matrix)
        
        # 상관관계 합이 최소인 데이터 선택
        def select_min_corr_sum(clusters):
            selected_indices = []
            for cluster_id in range(num_clusters):
                cluster_indices = np.where(clusters == cluster_id)[0]
                if len(cluster_indices) > 0:
                    cluster_center = cluster_indices[np.argmin(abs_corr_matrix.iloc[cluster_indices, cluster_indices].sum(axis=1))]
                    selected_indices.append(cluster_center)
            return np.array(selected_indices)        
        
        # # 초기 대표 데이터 선택
        min_corr_indices = select_min_corr_sum(clusters)
        
        return min_corr_indices


    def optimize_modes(self, initial_modes):
        corr_matrices = [self.vbm_df.corr().abs(), self.hbm_df.corr().abs(), self.tm_df.corr().abs()]
        total_corr = sum(corr_matrices) / len(corr_matrices)        
        total_corr.iloc[initial_modes, initial_modes ]
        optimized_modes =total_corr.iloc[initial_modes,initial_modes].index

        for iteration in range(1, 2001):
            trial_count = 0
            improved = False
            selected_corr = total_corr.loc[optimized_modes, optimized_modes]
                   
            self.fill_diagonal_with_nan(selected_corr)       
            max_corr_value = np.nanmax(selected_corr.values) 
            max_mode = selected_corr.columns[(selected_corr == max_corr_value).any()].tolist()[0]
            replace_mode = max_mode
            
            for candidate_mode in total_corr.columns.difference(optimized_modes):
                trial_modes = optimized_modes.copy()            
                trial_modes_list = trial_modes.tolist()
                replace_index = trial_modes_list.index(replace_mode)
                trial_modes_list[replace_index] = candidate_mode
                trial_modes = pd.Index(trial_modes_list)    
                new_corr_matrices = [df[trial_modes].corr().abs() for df in [self.vbm_df, self.hbm_df, self.tm_df]]
                new_total_corr = sum(new_corr_matrices) / len(new_corr_matrices)
                new_value = self.calculate_max_correlation_excluding_self(new_total_corr)

                if new_value < max_corr_value:
                    print(f"Iteration {iteration}: Replaced {replace_mode} with {candidate_mode}. New corr: {new_value:.4f}, Improvement after {trial_count} trials")
                    optimized_modes = trial_modes
                    improved = True
                    break
                else:
                    trial_count += 1

            if not improved:
                print(f"Iteration {iteration}: No improvement after {trial_count} trials.")
                if trial_count >= 2:
                    print("No improvement for 2 consecutive iterations. Optimization ends.")
                    break

        optimized_modes_with_index = ["/".join([mode, str(self.vbm_df.columns.get_loc(mode))]) for mode in optimized_modes]
        selected_vbm = self.vbm_df[optimized_modes]
        selected_hbm = self.hbm_df[optimized_modes]
        selected_tm = self.tm_df[optimized_modes]        
        optimized_modes = optimized_modes_with_index

        return selected_vbm, selected_hbm, selected_tm, optimized_modes


    def load_initial_modes(self):
        initial_modes = self.generate_initial_modes()
        return initial_modes
    
    def calculate_opt_modes(self,initial_modes):
        selected_vbm_max, selected_hbm_max, selected_tm_max, optimized_modes_max  = self.optimize_modes(initial_modes)
        return selected_vbm_max, selected_hbm_max, selected_tm_max, optimized_modes_max  # This could also include optimized_modes_max if needed
    
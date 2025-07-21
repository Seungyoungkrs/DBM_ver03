# DBM_ver03
AOE2 대상으로 하는 DBM 예측 코드의 GIT 버전

## 1. 프로젝트 개요

함정에 부착된 Strain Gauge 데이터를 기반 으로 함정 선체 응답을 예측할 수 있도록 전선구조해석 결과를 기반으로 선체 구조 전체의 응력 및 변위를 예측하는 DBM(또는 WMSR) 예측 방법에 대한 스크립트입니다. 
* WMSR : Wave induced Mode based Strain to Responce

## 2. 필수 폴더 및 파일 구조

스크립트를 정상적으로 실행하기 위해서는 아래와 같은 폴더 및 파일 구조를 반드시 준수해야 합니다. **프로젝트 폴더**와 **입력 데이터 폴더(`AOE2_input`)**는 동일한 위치에 있어야 합니다.
```
project_root/
│
├── 📂 AOE2_input/
│   │
│   ├── 📂 sensor_input_elem/
│   │   └── 📄 sensor_input_006.txt
│   │
│   ├── 📄 HYDRODYNAMIC_full.OUT
│   ├── 📄 HYDRODYNAMIC_min.OUT
│   ├── 📄 AOE_II_model.bdf
│   ├── 📄 ref_with_shell_o1deck.bdf
│   ├── 📄 AOE2_DBM_full_REAL.op2
│   ├── 📄 AOE2_DBM_min_REAL.op2
│   ├── 📄 AOE2_DBM_full_IMAG.op2
│   ├── 📄 AOE2_DBM_min_IMAG.op2
│   ├── 📄 Analysis_static_full.op2
│   └── 📄 Analysis_static_min.op2
│
└── 📂 DBM_Project/
│
├── 📄 main.py
├── 📂 module/
│   ├── 📄 p00_config.py
│   └── ... (p01, p02 등 다른 모듈 파일들)
│
└── 📂 Output/
└── 📂 pkl/
```

### 필수 입력 파일 목록

-   **HYDRODYNAMIC (full/min).OUT**: 대상 함정에 대한 유체동역학 해석 결과 데이터 (모든 LC : 대상함에 경우에는 Full/Min)
-   **AOE_II_model.bdf**: 전체 FE 모델
-   **ref_with_shell_o1deck.bdf**: 응력을 예측할 대상 요소에 대한 bdf 파일 (반드시 전체 FE 모델에서 추출한 모델이어야 함 : ID가 완전히 같아야 함)
-   **AOE2_DBM_* (REAL/IMAG).op2**: 전선구조해석 결과 중 REAL/IMAG 파일 (모든 )
-   **Analysis_static_*.op2**: 정적 해석 결과
-   **sensor_input_elem/sensor_input_XXX.txt**: 사용할 센서의 요소 ID 목록. 현재 `p00_config.py`에 `006`으로 설정되어 있습니다.

## 3. 실행 방법

1.  위의 **필수 폴더 및 파일 구조**에 맞게 모든 입력 파일을 배치합니다.
2.  터미널 또는 커맨드 프롬프트에서 `DBM_Project` 폴더(main.py가 있는 폴더)로 이동합니다.
3.  아래 명령어를 실행합니다.
    ```bash
    python main.py
    ```
4.  실행이 완료되면 `Output/pkl` 폴더와 `Y_MATRIX_XX.f06`과 같은 결과 파일이 생성됩니다.

## 4. 주요 설정 변경

`module/p00_config.py` 파일 내에서 주요 분석 파라미터를 수정할 수 있습니다.

-   **`nMode`**: 해석에 사용할 변형 모드의 개수를 설정합니다. (현재 `400`으로 고정)
-   **`sensor_number_str`**: 사용할 센서 파일 번호를 설정합니다. `sensor_input_XXX.txt` 파일의 `XXX`에 해당하는 부분입니다. (현재 `"006"`으로 고정)
-   **`new_mode_data`**:
    -   `True`: 기존에 생성된 모드 데이터(`optimization_variables.pkl`)를 로드하여 사용합니다. (시간 절약)
    -   `False`: Hydrodynamic 데이터로부터 새로운 모드 데이터를 계산합니다. (최초 실행 시 또는 조건 변경 시)
-   **`sensors`**: `_load_sensor_ids` 메서드 내에서 센서 파일을 찾지 못했을 경우 사용할 기본 센서 ID 목록을 직접 수정할 수 있습니다.

## 5. 결과 파일 관련

결과 파일 용량이 매우 방대한 관계(전체 결과 : 150GB, 참조 요소 결과 : 10GB)로 sylee82@krs.co.kr로 요청하시면 별도로 송부드립니다.


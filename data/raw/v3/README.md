수집한 원본 데이터 저장 디렉토리 입니다.

v3 데이터는 다양한 class들의 데이터를 Ai-hub를 통해 다운받아 roboflow에서 bbox 결함을 수정하여 재가공한 데이터입니다.
stage1_train_images는 Detection 전용 데이터로 다중 객체로 구성 된 데이터 입니다.
stage2_train_images는 Classification 전용 데이터로 단일 객체의 이미지로 class 분포를 맞춰 구성 된 데이터 입니다.

구조는 아래와 같습니다.
stage1_train_annotations/.json (1개의 json에 모든 객체 정보 포함)
stage1_train_images/.png
stage2_train_annotations/.json (1개의 json에 모든 객체 정보 포함)
stage2_train_images/.png


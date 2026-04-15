# LIBERO 데모 웹 UI 설계 문서

## 1. 문서 목적

이 문서는 RLinf 내부에 구현된 LIBERO / LIBERO-Plus 디버그 데모 웹 UI의 현재 설계를 정리한다.

대상 구현은 다음 세 파일을 중심으로 동작한다.

- [toolkits/eval_scripts_openpi/libero_debug_action_chunk.py](/workspace/RLinf/toolkits/eval_scripts_openpi/libero_debug_action_chunk.py)
- [toolkits/eval_scripts_openpi/libero_debug_video_dashboard.py](/workspace/RLinf/toolkits/eval_scripts_openpi/libero_debug_video_dashboard.py)
- [examples/embodiment/run_liberoplus_debug_action_chunk_openpi_pi05_base.sh](/workspace/RLinf/examples/embodiment/run_liberoplus_debug_action_chunk_openpi_pi05_base.sh)

이 문서는 "현재 어떤 의도로 설계되어 있고, 각 버튼이 어떤 의미를 가지는지"를 설명하는 데 집중한다.

## 2. 목표

이 데모 UI의 목표는 다음과 같다.

- RLinf 내부 의존성만으로 LIBERO 계열 policy를 interactive하게 디버깅할 수 있어야 한다.
- 모델을 매번 다시 띄우지 않고 같은 세션 안에서 task를 바꿔가며 확인할 수 있어야 한다.
- policy가 생성한 action chunk를 사람이 직접 수정하고, 그 수정 결과를 영상으로 확인할 수 있어야 한다.
- "실제 환경을 전진시키는 실행"과 "같은 입력 장면에서 시험만 해보는 실행"을 분리해야 한다.
- 결과물과 상태 파일은 `/data/inference_results/...` 아래에 남아야 한다.

## 3. 범위

현재 데모 UI는 다음 범위를 지원한다.

- LIBERO type 선택
  - 설치된 패키지 기준으로 동적으로 노출
  - 예: `plus`, `standard`
  - `pro` 패키지가 설치되지 않은 환경에서는 dropdown에 보이지 않음
- suite 선택
  - `libero_spatial`
  - `libero_object`
  - `libero_goal`
  - `libero_10`
  - `libero_90`
- task 단위 interactive debug
- chunk 단위 action 수정
- live video preview
- stable chunk video 저장
- task metadata 표시
  - suite별 task 개수
  - LIBERO-Plus taxonomy

참고:

- taxonomy 표시는 `plus`일 때만 의미가 있다.
- `standard`와 `plus`는 실제로 smoke test가 수행된 상태다.
- type dropdown은 현재 OpenPI venv에서 import 가능한 패키지만 보여준다.

## 4. 상위 구조

전체 구조는 세 레이어로 나뉜다.

### 4.1 Launcher 레이어

쉘 스크립트 `run_liberoplus_debug_action_chunk_openpi_pi05_base.sh`가 다음을 담당한다.

- OpenPI / LIBERO 계열 가상환경 선택
- RLinf repo를 `PYTHONPATH`에 추가
- LIBERO / LIBERO-Plus asset path 구성
- 모델 경로, suite, task id, type, video 옵션 등 환경변수 수집
- 실제 debug worker Python 프로세스 실행

### 4.2 Debug Worker 레이어

`libero_debug_action_chunk.py`가 다음을 담당한다.

- policy 로딩
- LIBERO 환경 생성
- 현재 task의 초기 상태 세팅
- policy action chunk 생성
- action chunk 수정 / simulate / run / task switch 처리
- live 상태 JSON 기록
- live preview mp4/gif 기록
- stable chunk video 저장

### 4.3 Dashboard 레이어

`libero_debug_video_dashboard.py`가 다음을 담당한다.

- 현재 run directory를 읽어서 웹에 상태 제공
- tmux worker 세션 제어
- task metadata 조회 및 캐시
- 버튼 입력을 API로 변환
- HTML/JS 렌더링

## 5. 세션 설계

tmux 세션은 두 개를 기본으로 사용한다.

- `libero_debug_dashboard`
  - 웹 서버 프로세스
- `libero_debug_worker`
  - 실제 policy / environment를 돌리는 worker

예전 task-specific 이름인 `libero_debug_goal1716_chunk` 대신 generic한 `libero_debug_worker`를 기본으로 사용한다.

## 6. 데이터 저장 구조

run directory는 대략 다음 형태를 가진다.

```text
/data/inference_results/liberoplus_action_debug/<timestamp>-<exp_name>/<exp_name>/
```

대표 산출물은 다음과 같다.

- `debug_summary.json`
- `debug_trace.jsonl`
- `live/status.json`
- `live/current_chunk.mp4`
- `live/current_chunk.gif`
- `live/latest_frame.jpg`
- `live/last_completed_chunk.mp4`
- `live/last_completed_chunk.gif`
- `chunk_videos/*.mp4`
- `chunk_videos/*.gif`
- `videos/success/*.mp4`
- `videos/failure/*.mp4`
- `dashboard_temp/*.json`

`dashboard_temp/`는 웹 UI가 action override를 임시로 넘길 때만 사용하는 temp 저장소다. 사용자가 action을 수정해서 `Simulate Chunk` 또는 `Run Chunk`를 눌렀을 때 여기 JSON이 잠깐 생성된다.

## 7. UI 설계

## 7.1 Run Controls

왼쪽 패널은 현재 다음 정보를 가진다.

- `Libero Type`
- `Suite`
- `Task ID`
- `Trial Index`
- `Set Task`
- `Selected Task Info`
- `Suite Task Counts`
- 현재 세션 정보
- 현재 run directory
- progress bar

### Selected Task Info

현재 입력된 `type / suite / task id` 조합에 대해 다음을 보여준다.

- suite 이름
- suite 내 task 개수
- 선택한 task id
- taxonomy

taxonomy는 `plus`일 때만 의미가 있고, `standard`나 `pro`에서는 `n/a`로 처리된다.

### Suite Task Counts

현재 선택한 type 기준으로 suite별 task 개수를 모두 보여준다.

예:

- `libero_spatial: N tasks`
- `libero_object: N tasks`
- `libero_goal: N tasks`

또한 suite dropdown label에도 개수가 함께 붙는다.

## 7.2 Live Chunk / Status

이 패널은 3개의 영역으로 구성된다.

### 좌측 상단: Live Video + 실행 버튼

- 현재 live preview mp4
- `Simulate Chunk`
- `Run Chunk`
- `Reset Edits`

### 우측 상단: 상태 카드

- phase
- libero type
- suite
- task id
- trial
- chunk
- env step
- taxonomy
- 마지막 action 등

### 하단 전체폭: Action Editor

action editor는 `Live Chunk / Status` 섹션 내부 하단에서 전체 폭을 가로지르도록 설계되어 있다.

의도는 다음과 같다.

- 영상/버튼은 왼쪽에 고정
- 상태 정보는 오른쪽 카드로 요약
- action table은 둘 아래에서 가로 전체를 사용

## 7.3 Completed Chunk Preview

이 영역은 실제로 완료된 chunk의 stable media를 순회하는 곳이다.

- `Prev Saved`
- `Next Saved`
- `Latest Chunk`
- `Play 0 -> Latest`
- `Stop Sequence`

중요:

- 여기는 live preview가 아니라 저장된 stable chunk video를 보는 영역이다.
- 현재 task에 속하는 chunk만 필터링해서 보여준다.

## 8. 버튼 의미

## 8.1 Set Task

의도:

- 새로운 task를 고른다.
- 가능하면 같은 모델 세션을 재사용한다.
- type이 바뀌지 않으면 worker 안에서 task만 switch한다.
- type이 바뀌면 worker를 새로 띄운다.

결과:

- task 시작 scene이 바로 live preview로 보인다.
- 첫 번째 policy chunk도 함께 준비된다.
- phase는 `awaiting_chunk_execution`으로 들어간다.

## 8.2 Simulate Chunk

의도:

- 현재 policy 입력 scene에서 action chunk를 비파괴 방식으로 시험한다.
- 실제 environment state를 전진시키지 않는다.

세부 동작:

- 현재 chunk의 action 값을 수정해도 됨
- 수정값은 `dashboard_temp/*.json`에 임시 저장
- 별도의 preview env를 새로 만들고
- 현재 planning 시점 scene까지 복원한 뒤
- 수정된 action chunk를 실행
- 결과 영상을 `live/current_chunk.mp4`로 갱신
- 하지만 실제 worker env의 state는 그대로 유지

결과적으로 보장해야 하는 것:

- 같은 scene에서 여러 번 반복 시험 가능
- `chunk_idx` 증가 없음
- stable `chunk_videos` 증가 없음

## 8.3 Run Chunk

의도:

- 현재 수정된 action chunk를 실제 environment에 적용한다.

세부 동작:

- 수정값이 있으면 temp JSON을 통해 worker에 전달
- 실제 env에서 해당 chunk를 실행
- stable chunk video를 저장
- 그 이후 scene에서 다음 policy chunk를 자동으로 다시 계산
- 다시 `awaiting_chunk_execution` 상태로 복귀

결과:

- 실제 상태가 다음 장면으로 진행됨
- `chunk_idx` 증가
- stable `chunk_videos/*.mp4` 증가

## 8.4 Reset Edits

의도:

- 현재 편집 중인 action table을 policy 원본 값으로 되돌린다.

## 9. 상태 머신

중요 phase는 아래와 같다.

- `setup`
  - env와 첫 scene 준비 중
- `awaiting_chunk_execution`
  - 현재 scene에서 chunk가 준비되었고, edit / simulate / run 가능
- `simulating_chunk`
  - dry-run preview 실행 중
- `executing_chunk`
  - 실제 env에서 chunk 실행 중
- `planning_next_chunk`
  - 방금 실행한 후 다음 policy chunk 생성 중
- `chunk_complete`
  - episode 종료 등으로 마지막 chunk 결과가 정리된 상태
- `finished`
  - task episode 종료
- `switching_task`
  - task 전환 중

## 10. Action Editor 입력 UX

현재 설계 의도는 다음과 같다.

- 숫자는 소수점 6자리까지 표시
- spinner 화살표를 제거하고 키보드 타이핑 중심으로 수정
- 같은 chunk를 편집 중이면 polling이 들어와도 DOM을 갈아끼우지 않음
- 그래서 입력 focus가 쉽게 풀리지 않아야 함

구현상 포인트:

- `<input type="text" inputmode="decimal">`
- 동일 chunk / 동일 editable 상태일 때 table rerender 방지
- `Reset Edits` 때만 강제 rerender

## 11. Task Metadata 설계

dashboard 서버는 type별 task catalog를 subprocess로 생성하고 캐시한다.

catalog는 다음 정보를 가진다.

- `suite_task_counts`
- `task_taxonomy_by_suite`

생성 방식:

- 선택한 OpenPI venv의 Python 사용
- RLinf repo 및 LIBERO package 경로를 `PYTHONPATH`에 주입
- `libero_eval.py`의 `_import_libero_stack`, `_load_taxonomy_lookup` 재사용

또한 dashboard는 시작 시점에 각 type import 가능 여부를 검사하고, 실제로 사용 가능한 type만 UI에 노출한다.

이렇게 해야 웹 UI가 RLinf 내부 정보만으로 metadata를 일관되게 읽을 수 있다.

## 12. 내부 API

대표 API는 다음과 같다.

- `GET /api/state`
  - live 상태, trace tail, summary, session info
- `GET /api/task_catalog?libero_type=<type>`
  - suite별 task count, plus taxonomy metadata
- `POST /api/set_task`
  - task / type 변경
- `POST /api/simulate`
  - 비파괴 simulate preview
- `POST /api/run_chunk`
  - 실제 chunk 실행

## 13. 설계상 제한 사항

- taxonomy는 LIBERO-Plus에서만 의미 있게 표시된다.
- type 변경 시에는 worker 재시작과 모델 재로딩이 발생할 수 있다.
- task metadata catalog는 첫 로딩이 다소 무거울 수 있다.
- action 수정량이 작으면 영상상 차이가 육안으로 작아 보일 수 있다.
- 현재 UI는 task 이름 검색 기능까지는 제공하지 않고, task id 기준으로 동작한다.

## 14. 검증 기준

현재 구현에서 중요하게 보는 검증 기준은 다음과 같다.

- `Set Task` 후 시작 scene이 즉시 live preview에 보일 것
- `Set Task` 후 첫 chunk가 `awaiting_chunk_execution` 상태로 준비될 것
- `Simulate Chunk`는 same-scene dry-run일 것
- `Run Chunk`는 실제 env를 전진시키고 다음 chunk를 자동 계획할 것
- action editor는 편집 중 focus out이 자주 발생하지 않을 것
- plus에서 taxonomy가 표시될 것
- suite별 task count가 표시될 것

## 15. 향후 개선 후보

- task id 대신 task name 검색
- taxonomy별 필터링
- 현재 frame와 simulate 결과 frame의 side-by-side 비교
- action diff 강조 표시
- live video와 completed video의 더 명확한 시각 구분

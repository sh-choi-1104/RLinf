# LIBERO 데모 콘솔 사용법

## 1. 목적

이 문서는 RLinf 안에 구현된 LIBERO 데모 웹 UI를 콘솔에서 실행하고, 포트포워딩으로 브라우저에서 접속하고, tmux 세션을 관리하는 방법을 정리한다.

대상 환경은 다음을 가정한다.

- 원격 서버에서 작업 중
- RLinf repo는 `/workspace/RLinf`
- 결과 저장 루트는 `/data/inference_results/liberoplus_action_debug`
- 모델 경로는 `/data/models/pi05_libero_finetuned_v044`
- 학습 / 평가 / 디버그 worker는 기본적으로 `CUDA_VISIBLE_DEVICES=0`만 사용
- EGL 렌더링도 기본적으로 `MUJOCO_EGL_DEVICE_ID=0`, `EGL_DEVICE_ID=0`으로 고정

## 2. 주요 구성

사용되는 tmux 세션은 두 개다.

- `libero_debug_dashboard`
  - 웹 서버
- `libero_debug_worker`
  - policy / env 실행 worker

내부 구현상 dashboard의 session/run-state backend는
`toolkits/eval_scripts_openpi/libero_debug_dashboard_backend.py`로 분리되어 있다.

## 3. 데모 서버 실행

repo 루트로 이동한다.

```bash
cd /workspace/RLinf
```

dashboard를 실행한다.

```bash
tmux kill-session -t libero_debug_dashboard 2>/dev/null || true
tmux new-session -d -s libero_debug_dashboard \
  "bash -lc 'cd /workspace/RLinf && \
   python3 toolkits/eval_scripts_openpi/libero_debug_video_dashboard.py \
     --run_dir /data/inference_results/liberoplus_action_debug/__dashboard_bootstrap__ \
     --host 0.0.0.0 \
     --port 8765 \
     --session_name libero_debug_worker'"
```

설명:

- `--run_dir`는 부트스트랩용 placeholder다.
- 실제 run directory는 dashboard가 worker를 띄우면서 자동으로 갱신한다.
- `--session_name`은 dashboard가 제어할 worker tmux 세션 이름이다.
- 기본 OpenPI Python은 `/workspace/RLinf/.venv-openpi-liberoplus/bin/python` 이다.
- worker 자체는 launcher에서 기본적으로 `CUDA_VISIBLE_DEVICES=0`으로 뜬다.
- 렌더링도 launcher에서 기본적으로 GPU 0으로 고정된다.

## 4. 세션 확인

현재 tmux 세션을 본다.

```bash
tmux ls
```

정상이라면 대략 다음 둘이 보여야 한다.

```text
libero_debug_dashboard
libero_debug_worker
```

dashboard 로그를 최근 몇 줄 확인한다.

```bash
tmux capture-pane -pt libero_debug_dashboard -S -80
```

worker 로그를 최근 몇 줄 확인한다.

```bash
tmux capture-pane -pt libero_debug_worker -S -120
```

직접 attach 하고 싶다면:

```bash
tmux attach -t libero_debug_worker
```

detach:

```text
Ctrl-b d
```

## 5. 웹 접속 URL

dashboard가 올라오면 브라우저에서 다음 URL로 접속한다.

```text
http://127.0.0.1:8765/
```

원격 서버에서 작업 중이라면 로컬에서 포트포워딩이 필요하다.

## 6. SSH 포트포워딩

사용자 SSH config에 `Host SNU-104`와 `ProxyJump SNU-10`이 이미 있다면, 로컬 PC에서 아래처럼 접속하면 된다.

### 6.1 가장 단순한 경우

원격 호스트에서 dashboard 포트가 바로 보이는 경우:

```bash
ssh -N -L 8765:127.0.0.1:8765 SNU-104
```

이후 로컬 브라우저에서:

```text
http://127.0.0.1:8765/
```

### 6.2 Docker 컨테이너 내부에서 실행 중이고 포트 publish가 안 된 경우

먼저 원격 서버에서 컨테이너 IP를 확인한다.

```bash
docker ps
docker inspect -f '{{range.NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container_name_or_id>
```

예를 들어 컨테이너 IP가 `172.17.0.3`이면, 로컬 PC에서:

```bash
ssh -N -L 8765:172.17.0.3:8765 SNU-104
```

이후 로컬 브라우저에서:

```text
http://127.0.0.1:8765/
```

### 6.3 권장 방식

가능하면 컨테이너를 실행할 때 포트를 publish하는 편이 관리가 쉽다.

예:

```bash
docker run -p 8765:8765 ...
```

그 경우 로컬 포워딩은 다시 단순해진다.

```bash
ssh -N -L 8765:127.0.0.1:8765 SNU-104
```

## 7. 웹 UI 기본 사용 흐름

### 7.1 Set Task

왼쪽 `Run Controls`에서 다음을 고른다.

- `Libero Type`
- `Suite`
- `Task ID`
- `Trial Index`
- `Model Ckpt`

그 다음 `Set Task`를 누른다.

참고:

- `Libero Type` dropdown은 현재 환경에서 실제 import 가능한 type만 보여준다.
- 예를 들어 `liberopro`가 설치되지 않았으면 `pro`는 dropdown에 나타나지 않는다.
- `Task ID`는 기본적으로 비어 있으며, 직접 선택해야 한다.
- `Model Ckpt`는 현재 worker가 사용할 checkpoint 경로다.
- checkpoint 경로를 바꾸고 `Set Task`를 누르면 worker가 해당 모델로 다시 로드된다.
- 별도 설정이 없으면 worker는 GPU 0에서만 실행된다.
- `Reset`은 현재 worker와 dashboard history를 함께 비우고 idle 상태로 되돌린다.
- live video/preview 파일은 중간 파일을 읽지 않도록 atomic replace 방식으로 갱신된다.

의도:

- 같은 type이면 worker 안에서 task만 switch
- type이 바뀌면 worker 재기동
- 시작 scene과 첫 chunk를 자동 준비

정상 상태:

- progress가 `Chunk ready. Edit actions, Simulate, or Run.` 로 바뀜
- live video가 시작 scene을 보여줌
- live video 오른쪽에 현재 chunk planning에 사용된 `agentview` / `wrist` observation 이미지가 표시됨
- obs state 8개 값도 라벨과 함께 표시됨

### 7.2 Simulate Chunk

의도:

- 현재 장면에서 action을 시험만 해본다
- 실제 env는 전진하지 않는다

동작:

- action table 수정
- `Simulate Chunk` 클릭
- `live/current_chunk.mp4`가 수정값 기반 preview로 갱신
- 우측 observation 패널은 같은 chunk를 계획할 때 사용한 입력 이미지를 유지
- live pane은 브라우저 안정성을 위해 GIF preview를 기본으로 표시한다

### 7.3 Run Chunk

의도:

- 현재 수정된 action을 실제 env에 적용한다

동작:

- action table 수정
- `Run Chunk` 클릭
- 실제 chunk 실행
- stable chunk video 저장
- 다음 chunk 자동 생성

### 7.4 Reset Edits

현재 action table을 그 chunk에서 모델이 처음 출력한 원본값으로 되돌린다.

## 8. 콘솔에서 API 직접 호출하기

브라우저 대신 콘솔에서 상태를 볼 수 있다.

### 8.1 현재 상태 조회

```bash
curl -s http://127.0.0.1:8765/api/state | python3 -m json.tool
```

### 8.2 task metadata 조회

`plus` 기준 suite별 task count와 taxonomy catalog 조회:

```bash
curl -s "http://127.0.0.1:8765/api/task_catalog?libero_type=plus" | python3 -m json.tool
```

### 8.3 task 전환

```bash
curl -s -X POST http://127.0.0.1:8765/api/set_task \
  -H 'Content-Type: application/json' \
  -d '{"libero_type":"plus","suite_name":"libero_goal","task_id":1716,"trial_idx":0,"model_path":"/data/models/pi05_libero_finetuned_v044"}' \
  | python3 -m json.tool
```

### 8.4 simulate

현재 action plan을 `api/state`에서 가져와서 다시 `api/simulate`에 넘기는 식으로 사용할 수 있다.

예시:

```bash
python3 - <<'PY'
import json, urllib.request

base = "http://127.0.0.1:8765"

with urllib.request.urlopen(base + "/api/state") as resp:
    state = json.load(resp)

live = state["live_status"]
payload = {
    "chunk_idx": live["chunk_idx"],
    "actions": live["current_plan_actions"],
}

req = urllib.request.Request(
    base + "/api/simulate",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as resp:
    print(resp.read().decode())
PY
```

### 8.5 run chunk

```bash
python3 - <<'PY'
import json, urllib.request

base = "http://127.0.0.1:8765"

with urllib.request.urlopen(base + "/api/state") as resp:
    state = json.load(resp)

live = state["live_status"]
payload = {
    "chunk_idx": live["chunk_idx"],
    "actions": live["current_plan_actions"],
}

req = urllib.request.Request(
    base + "/api/run_chunk",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req) as resp:
    print(resp.read().decode())
PY
```

## 9. 상태 파일 위치

현재 run directory 아래에서 자주 보는 파일은 다음과 같다.

- `live/status.json`
- `live/current_chunk.mp4`
- `debug_trace.jsonl`
- `debug_summary.json`
- `chunk_videos/*.mp4`

예:

```bash
RUN_DIR=/data/inference_results/liberoplus_action_debug/<run_root>/<exp_name>
cat "${RUN_DIR}/live/status.json"
tail -n 20 "${RUN_DIR}/debug_trace.jsonl"
ls -lah "${RUN_DIR}/chunk_videos"
```

## 10. 자주 쓰는 진단 명령

### dashboard가 응답하는지 확인

```bash
curl -sv http://127.0.0.1:8765/api/state > /dev/null
```

### 8765 포트 listen 확인

```bash
ss -lntp | rg 8765
```

### worker 최근 로그 확인

```bash
tmux capture-pane -pt libero_debug_worker -S -120
```

### dashboard 최근 로그 확인

```bash
tmux capture-pane -pt libero_debug_dashboard -S -120
```

## 11. 종료

dashboard 종료:

```bash
tmux kill-session -t libero_debug_dashboard
```

worker 종료:

```bash
tmux kill-session -t libero_debug_worker
```

## 12. 권장 데모 순서

현장에서 가장 설명이 쉬운 데모 순서는 아래다.

1. 포트포워딩 연결
2. 웹 접속
3. `Set Task`
4. start scene과 첫 chunk 확인
5. action 값 일부 수정
6. `Simulate Chunk`로 같은 scene dry-run 확인
7. `Run Chunk`로 실제 상태 전진 확인
8. `Completed Chunk Preview`에서 stable chunk video 확인

이 순서를 따르면 "같은 입력 장면 테스트"와 "실제 실행"의 차이를 가장 직관적으로 보여줄 수 있다.

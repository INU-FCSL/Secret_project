# MicroDog 개발 규칙

## Language
- 모든 사용자 대상 설명, 분석 결과, 작업 보고, 표 제목, 결론은 한국어로 작성한다.
- 일본어와 중국어를 사용하지 않고, 답변 중간에 다른 언어로 전환하지 않는다.
- 코드, 변수명, 파일명, MuJoCo API, XML tag, Git 명령어 등 기술적으로 필요한 영어 식별자는 유지할 수 있다.
- 이전 대화나 파일의 언어와 관계없이 사용자 대상 출력은 한국어로 통일한다.

## Git
- 작업 전 `git status`를 확인한다.
- 예상하지 못한 사용자 변경사항은 덮어쓰지 않고 먼저 보고한다.
- force push를 금지한다.
- 사용자 지시 없이 Git history를 변경하지 않는다.
- 중요한 수정 전후에 `git diff`를 확인한다.
- commit과 push는 사용자가 지시한 경우에만 수행한다.

## Project
- `/home/fcsl/Secret_project`를 MicroDog의 메인 개발 repository로 사용한다.
- `/home/fcsl/robot_ws/mujoco/microdog`는 초기 원본 reference이며, 기존 reference 파일은 수정하지 않는다.
- MicroDog의 구조는 floating base + 4개의 3-DOF leg + 3-DOF head + 2-DOF tail이다.

## Development
- 개발 순서: Baseline → Model validation → Collision cleanup → Neutral standing pose → Actuator validation → RL interface → Standing RL → Walking RL → Sim-to-real.
- MuJoCo 실행에는 `/home/fcsl/robot_ws/mujoco/.venv/bin/python`을 사용한다.
- 사용자 지시 없이 새 패키지를 설치하지 않는다.
- 변경 목적을 분리하고, collision cleanup에서는 불필요한 내부 자기충돌만 최소한으로 제거한다.
- foot, leg, torso, head, tail과 floor 사이의 collision은 유지한다.
- 모델 변경 시 모델 차원, 질량, joint order, actuator mapping과 kinematics의 회귀를 확인한다.

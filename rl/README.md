# MicroDog 기립 학습 초기 환경

`feature/standing-rl`에서 검증된 `microdog.xml`을 직접 읽는다. XML과 기존 실행 스크립트는 변경하지 않는다. mjlab의 `Simulation`이 MuJoCo Warp 물리를 수행하고, rsl_rl의 PPO가 정책을 학습한다. MicroDuck task나 정책을 불러와 사용하지 않는다.

## 실행 환경

기존 `/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python`을 사용한다. 새 패키지 설치나 reference 수정이 필요하지 않다. 확인한 버전은 Python 3.12.3, PyTorch 2.9.1, mjlab 1.3.0, MuJoCo 3.10.0, MuJoCo Warp 3.8.1, rsl_rl 5.0.1이다. 기존 standalone 검증 가상환경의 MuJoCo 3.14.0과 버전 차이가 있으므로 GPU 결과를 별도로 확인한다.

```bash
cd /home/fcsl/Secret_project
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m unittest discover -s tests -v
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.train --num-envs 16 --iterations 3 --log-dir /tmp/microdog_standing_smoke
```

첫 명령이 통과한 뒤 두 번째 명령을 실행한다. 기본 학습은 16개 환경, 16 step rollout, 3 iteration이다. checkpoint, loss와 TensorBoard 기록은 지정 경로에만 저장하고 외부 서비스에 업로드하지 않는다. 최초 실행에는 Warp kernel compile 시간이 추가될 수 있다.

## 제어와 관측

물리는 0.002초, 제어는 decimation 10으로 0.02초다. XML의 `neutral_standing` keyframe에서 초기 `qpos`, `ctrl`과 기준 관절각을 읽는다.

정책 action은 FL, FR, RL, RR 각각 hip roll, hip pitch, knee 순서의 12개다. 정규화 action을 [-1,1]로 제한하고 scale `(0.004363323, 0.034906585, 0.034906585)` rad를 다리마다 적용한다. Head/Tail 목표는 0이다.

급변을 완화하기 위해 0.75초 시정수의 필터를 사용한다.

```text
alpha = 1 - exp(-0.02 / 0.75)
applied_action += alpha * (clipped_requested_action - applied_action)
target = neutral + action_scale * applied_action
```

최종 target은 joint range와 actuator ctrlrange의 교집합으로 제한한다. 이는 이전 0.75초 선형 Ramp와 다른 필터이며 zero/random action으로 별도 검증한다. 실제 배포 시에도 동일한 필터가 필요하다.

actor와 critic은 동일한 42차원 관측을 사용한다.

| 순서 | 값 | 차원 |
|---|---|---:|
| 0:3 | 몸통 좌표계의 projected gravity | 3 |
| 3:6 | 몸통 좌표계 angular velocity | 3 |
| 6:18 | 실제 다리 관절각 − 기준각 | 12 |
| 18:30 | 다리 관절 속도 | 12 |
| 30:42 | 직전 step에 실제 적용한 정규화 action | 12 |

마지막 항목은 필터 상태를 관측 가능하게 만든다. 원래 정책 요청은 `requested_actions`, 적용값은 `previous_actions`, 실제 target은 `joint_targets`에서 추적한다. action 변화 비용은 현재와 직전 step의 실제 적용 action 차이를 사용한다. 필터 적용과 observation 정의를 바꾸면 정책을 다시 검증해야 한다.

actor에는 세계 위치, 절대 quaternion, 접촉력, base 높이를 넣지 않는다. projected gravity는 향후 IMU 자세 추정으로 대체할 수 있지만 센서 추정 오차는 아직 모델링하지 않는다. PPO actor/critic 관측 정규화는 활성화한다.

## 보상

`g`는 projected gravity, `dq`는 기준각 대비 편차, `v`는 다리 관절 속도, `da`는 현재와 직전 step의 실제 적용 action 차이, `tau`는 actuator 토크다. `mean`은 다리 12개 평균이다.

| 항목 | 수식 | weight |
|---|---|---:|
| 기립 | `exp(-sum(g_xy²)/0.2²) * clamp(-g_z,0,1)` | +1.5 |
| 높이 | `exp(-max(abs(z-0.1927)-0.005,0)²/0.02²)` | +0.5 |
| 관절 자세 | `exp(-mean(dq²)/0.1²)` | +0.2 |
| 관절 속도 비용 | `mean(v²)` | −0.02 |
| action 변화 비용 | `mean(da²)` | −0.01 |
| 노력 비용 | `mean((tau/0.52)²)` | −0.01 |

합계에 제어 간격 0.02초를 곱한다. 비용은 비음수 함수에 음수 weight를 적용한다. 뒤집힌 자세의 기립 점수는 0이다. 정상 평형 주변 높이 ±5 mm에는 높이 비용을 추가하지 않는다. 발 접촉은 진단과 비정상 접촉 종료에 사용하며 gait reward는 없다.

## 종료와 reset

20초에 timeout을 발생시킨다. 높이 0.10 m 미만, 기울기 45° 초과, 발 이외 형상의 지면 접촉, 유한하지 않은 물리 상태는 실패 종료다. 문턱은 정상 높이 약 0.193 m와 작은 기립 기울어짐에 충분한 여유를 둔다.

종료 환경만 중립 keyframe으로 자동 reset한다. timeout은 실패 종료와 구분해 PPO에 전달한다. 종료 직전 관측과 진단은 `extras`에 보존하고 반환 관측은 reset 후 상태다. 잘못된 action shape이나 NaN action은 입력 오류로 처리한다.

다리 actuator 토크와 포화, 자기접촉·비정상 지면 접촉은 physics substep마다 검사한다. 발 접지는 제어 step 종료 상태에서 기록한다. 외란·reset noise·domain randomization은 비활성 상태다.

## 다음 검증

현재 구조는 기립 유지와 학습 연결 확인용이다. 외란 복원 능력이나 보행 성능을 검증한 정책이 아니다. 다음 단계에서 작은 초기 자세 변화와 외란 평가를 추가하고, 이후 마찰·질량·actuator strength·damping·지연·관측 noise를 순차 검증한다. ONNX 배포·Walking reward·velocity command는 구현하지 않았다.

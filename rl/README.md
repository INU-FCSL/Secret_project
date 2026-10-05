# MicroDog Standing Task V2

`feature/standing-rl`에서 검증된 `microdog.xml`을 직접 읽는다. XML과 기존 실행 스크립트는 변경하지 않는다. mjlab의 `Simulation`이 MuJoCo Warp 물리를 수행하고, rsl_rl의 PPO가 정책을 학습한다. MicroDuck task나 정책을 불러와 사용하지 않는다.

## 실행 환경

기존 `/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python`을 사용한다. 새 패키지 설치나 reference 수정이 필요하지 않다. 확인한 버전은 Python 3.12.3, PyTorch 2.9.1, mjlab 1.3.0, MuJoCo 3.10.0, MuJoCo Warp 3.8.1, rsl_rl 5.0.1이다. 기존 standalone 검증 가상환경의 MuJoCo 3.14.0과 버전 차이가 있으므로 GPU 결과를 별도로 확인한다.

```bash
cd /home/fcsl/Secret_project
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m unittest discover -s tests -v
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.train --num-envs 16 --iterations 3 --log-dir /tmp/microdog_standing_smoke
```

첫 명령이 통과한 뒤 두 번째 명령을 실행한다. 기본 학습은 Stage 0의 16개 환경, 16 step rollout, 3 iteration이다. checkpoint, loss와 TensorBoard 기록은 지정 경로에만 저장하고 외부 서비스에 업로드하지 않는다. 최초 실행에는 Warp kernel compile 시간이 추가될 수 있다.

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

모델별 자연 평형에서 `g_ref`와 `z_ref`를 보정한다. 목표 자세는 `StandingCfg.reference_standing_orientation`의 rad 단위 roll/pitch/yaw로, 높이는 `reference_standing_height`로 별도 지정할 수 있다. 미지정 값은 `neutral_standing`을 35초 안정화한 보정 결과에서 가져온다. 파일 내용 hash를 보정 cache의 키에 포함한다. 목표 pitch 수치를 모델 공통 상수로 하드코딩하지 않는다.

actor에는 세계 위치, 절대 quaternion, 접촉력, base 높이를 넣지 않는다. projected gravity는 향후 IMU 자세 추정으로 대체할 수 있지만 센서 추정 오차는 아직 모델링하지 않는다. PPO actor/critic 관측 정규화는 활성화한다.

## 보상

`g`는 projected gravity, `dq`는 기준각 대비 편차, `v`는 다리 관절 속도, `da`는 현재와 직전 step의 실제 적용 action 차이, `tau`는 actuator 토크다. `mean`은 다리 12개 평균이다.

| 항목 | 수식 | weight |
|---|---|---:|
| 자세 추종 | `exp(-sum((g-g_ref)²)/0.05²) * clamp(dot(g,g_ref),0,1)` | +1.5 |
| 높이 | `exp(-max(abs(z-z_ref)-0.005,0)²/0.02²)` | +0.5 |
| 관절 자세 | `exp(-mean(dq²)/0.1²)` | +0.2 |
| 관절 속도 비용 | `mean(v²)` | −0.02 |
| action 변화 비용 | `mean(da²)` | −0.01 |
| 노력 비용 | `mean((tau/0.52)²)` | −0.01 |

합계에 제어 간격 0.02초를 곱한다. 비용은 비음수 함수에 음수 weight를 적용한다. 뒤집힌 자세의 기립 점수는 0이다. 정상 평형 주변 높이 ±5 mm에는 높이 비용을 추가하지 않는다. 발 접촉은 진단과 비정상 접촉 종료에 사용하며 gait reward는 없다.

## 종료와 reset

20초에 timeout을 발생시킨다. 높이 0.10 m 미만, 기울기 45° 초과, 발 이외 형상의 지면 접촉, 유한하지 않은 물리 상태는 실패 종료다. 문턱은 정상 높이 약 0.193 m와 작은 기립 기울어짐에 충분한 여유를 둔다.

종료 환경만 중립 keyframe으로 자동 reset한다. timeout은 실패 종료와 구분해 PPO에 전달한다. 종료 직전 관측과 진단은 `extras`에 보존하고 반환 관측은 reset 후 상태다. 잘못된 action shape이나 NaN action은 입력 오류로 처리한다.

다리 actuator 토크와 포화, 자기접촉·비정상 지면 접촉은 physics substep마다 검사한다. 자기접촉이 발생한 제어 step은 실패 종료하고 timeout과 구분하여 PPO에 전달한다. 바닥·발 접촉은 정상이며 XML에서 제외한 접촉 관계는 그대로 유지한다. 활성화된 비바닥 접촉은 허용하지 않는다. 깊이나 지속 시간 문턱은 두지 않는다. 발 접지는 제어 step 종료 상태에서 기록한다. Stage 0에서는 외란과 초기 오차가 없으며 Stage 1 이상에서는 아래 초기 오차를 적용한다. 질량·마찰 등 물리 계수의 무작위화는 아직 구현하지 않았다.

## 초기 오차와 외력 단계

`StandingCfg.stage` 또는 학습·평가의 `--stage`로 단계를 명시한다. 자동 승급은 없다.

| 단계 | 초기 자세 및 관절 오차 | 외력 |
|---|---|---|
| 0 | 중립 keyframe | 없음 |
| 1 | roll/pitch ±2°, yaw 0; hip roll ±0.1°, hip pitch/knee ±0.5° | 없음 |
| 2 | 단계 1과 같음 | 몸통에 x 또는 y 방향 ±1 N, 0.15초 |
| 3 | 자세와 관절 위치 오차를 단계 1의 1.5배로 확대 | ±2 N, 0.15초; 진단용 |

단계 1 이상에서 관절 속도는 ±0.01 rad/s, 몸통 roll/pitch 각속도는 ±0.02 rad/s다. 몸통 선속도와 yaw 각속도는 0이다. 초기 발 관통을 피하기 위해 가장 낮은 발의 바닥면이 지면에 닿도록 몸통 높이를 보정한다. 물리 모델, 기구학, keyframe 자체는 변경하지 않는다.

외력은 세계 좌표계의 `xfrc_applied`로 몸통 질량중심에 가하며 토크는 0이다. 각 episode의 2~3초 사이에서 시작한다. force 적용 여부는 각 0.002초 substep에서 확인한다. reset 시 남은 외력을 지운다. `push_force`는 물리 시험에서만 크기를 별도로 지정할 수 있다.

물리 sweep에서 1 N은 자기접촉 없이 복구했으나 1.5~2 N에서는 일부 자기접촉이 발생했다. 4 N에서는 넘어짐이 나타났다. 따라서 단계 3은 검증된 안전 학습 단계가 아니다. 단계 1의 성공률 99% 이상, 실패 종료율 1% 이하, 자기접촉 0을 확인한 뒤 단계 2를 선택한다. 단계 3을 학습에 쓰기 전에는 자기접촉 없이 복구 가능한 범위를 다시 검증해야 한다. 단계 2의 16 episode 결과만으로 자동 승급하지 않는다.

초기 오차와 외력 일정은 전용 seeded generator에서 생성한다. 정책 구성으로 바뀌는 전역 난수 상태와 분리한다. 정책 비교는 동일한 seed, 환경 수, 단계, episode 길이를 사용한다. 학습 seed는 42, 기본 평가 seed는 2026이다.

## Pilot 학습과 평가

```bash
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.train --num-envs 64 --iterations 100 --stage 2 --episode-seconds 10 --seed 42 --log-dir /tmp/microdog_standing_pilot
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.evaluate --stage 2 --seed 2026 --output /tmp/standing_zero.json
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.evaluate --stage 2 --seed 2026 --random --output /tmp/standing_random.json
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.evaluate --stage 2 --seed 2026 --checkpoint /tmp/microdog_standing_pilot/checkpoint_100.pt --output /tmp/standing_ppo.json
```

`checkpoint_0.pt`는 학습 전 모델이며 `checkpoint_25/50/75/100.pt`는 실제 완료된 PPO update 수를 뜻한다. 라이브러리가 생성하는 `model_*.pt`의 0부터 시작하는 iteration 번호와 구분한다. `metrics.json`에는 update마다 평균 step reward, 완료된 episode의 return과 길이, 실패율, 보상 항목, loss, entropy, learning rate, 자기접촉·지면 이상 접촉 수, 포화 표본 수, peak torque를 기록한다. 아직 완료된 episode가 없는 update의 episode 평균은 `null`이다. `environment.json`에는 환경 설정, 모델 hash, Git 기준점과 GPU를 기록한다.

평가는 환경마다 첫 episode만 집계한다. 조기 종료 뒤 자동 reset된 후속 episode를 섞지 않는다. 16개 환경의 10초 episode에서 생존율·실패율·return, 최대/평균 절대 roll/pitch, 높이 오차, 네 발 접지 비율, 포화 비율, 자기접촉과 비정상 지면 접촉, 초기 및 외란 복구 시간을 기록한다. 발 접지는 제어 step 종료 시점, 접촉 사건과 포화는 모든 물리 substep을 검사한다. 접촉 수의 단위는 해당 사건이 발생한 환경별 제어 step이며 독립된 접촉 사건의 개수가 아니다.

V2 복구는 `|roll-roll_ref|, |pitch-pitch_ref| < 0.1°`, 보정된 목표 높이 오차 <5 mm, 네 발 접지를 0.2초 연속 유지하는 것으로 정의한다. 목표는 reward와 동일한 보정 자세다. 이 문턱은 시뮬레이션 평가용이며 실기 문턱은 센서 오차를 측정한 뒤 검증해야 한다. 기존 수평 0.6° 기준 복구 시간은 별도 지표로 남긴다. 외력이 끝난 뒤 처음 성립한 구간의 시작까지를 복구 시간으로 기록한다. 초기 복구도 같은 기준을 쓰며 episode 시작을 기준으로 잰다. 성공률은 생존, 초기 및 외란 복구, 자기접촉·비정상 접촉 없음, actuator 포화 표본 비율 1% 미만을 모두 요구한다. 생존율과 성공률은 다를 수 있다.

## 한계와 다음 검증

현재는 제한된 기립 복구 실험이며 보행 정책이 아니다. 자기접촉은 V2에서 실패 종료와 평가 지표에 반영한다. 따라서 중간 정책에서 접촉이 발견되면 보상만 좋아졌다는 이유로 통과시키지 않는다. 고정된 PD 제어기만으로도 작은 외란에서 빠르게 회복하기 때문에 PPO의 역할을 별도로 증명해야 한다.

0.75초 필터의 90% 응답 시간은 약 1.73초다. 0.15초 외력 동안 목표 변화의 약 18%만 적용되므로 빠른 반응의 한계가 있다. 이번 pilot에서는 action scale과 필터를 바꾸지 않았다. 다음 검증에서는 여러 학습·평가 seed, 접촉 방지 제약, 외란 중 최대 기울기와 회복 지표를 확인한다. 이후 실기 기반 마찰·질량·actuator strength·damping·지연·관측 noise를 검증한다. ONNX 배포, Walking reward, velocity command는 구현하지 않았다.

## 제어 권한 진단

추가 학습 없이 지정된 CPU Python으로 자연 평형과 action 응답을 검사한다.

```bash
/home/fcsl/robot_ws/mujoco/.venv/bin/python -m rl.control_authority --suite all --output /tmp/microdog_authority
/home/fcsl/robot_ws/mujoco/.venv/bin/python -m unittest discover -s tests -p test_control_authority.py -v
```

`--suite authority`는 단일 관절 24개 시험과 대칭 조합 32개 시험이다. `smoothing`, `disturbances`, `controllers`, `validation`으로 개별 절차를 실행할 수 있다. 결과와 모든 시간 이력은 출력 경로에 저장한다. 전체 진단 결과에는 수백 MB의 공간이 필요하다. 기존 환경의 action scale, smoothing, observation, reward와 termination은 변경하지 않는다.

각 시험은 zero-action으로 35초 안정화한 동일한 물리 상태에서 독립적으로 시작한다. 마지막 5초의 평형 변동을 측정하고, action 입력 후 0.1/0.2/0.5/1/2초의 자세, 관절 목표와 실제 위치, 각속도, 토크와 접촉을 저장한다. 초기 기울기 시험에서는 초기 시점도 최대 오차에 포함한다. 모든 물리 substep에서 자기접촉, 깊이, 지속 시간, 접지 손실과 포화를 기록한다.

앞뒤 knee 차등 명령은 pitch를, 좌우 knee 차등 명령은 roll을 제어할 수 있었다. 양의 knee 명령은 해당 다리를 접는 방향이며, 앞쪽을 접으면 양의 pitch, 왼쪽을 접으면 음의 roll 응답이 나타났다. hip pitch와 knee를 함께 쓰는 조합은 hip pitch를 knee 명령의 반대 부호 절반으로 주어 발의 수평 이동을 줄인다. 조합의 실제 응답과 foot Jacobian을 함께 저장한다.

수평 목표와 자연 평형 복귀는 별도로 평가한다. 자연 평형 pitch가 약 +0.400°이므로 0.6° 판정만으로는 작은 외란 제어를 구분하기 어렵다. 진단에서는 0.6/0.2/0.1° 오차 문턱을 비교하며 높이 5mm와 네 발 접지 0.2초 조건을 함께 유지한다. 0.1°는 측정한 시뮬레이션 정상상태 변동보다 충분히 크지만 실기 센서 정확도를 보장하는 값은 아니다.

자연 평형을 기준으로 하는 scripted 제어기는 projected gravity에서 구한 roll/pitch와 몸통 각속도만 사용한다. 단일 관절을 독립적으로 보정하는 대신 두 knee 차등 조합의 측정된 응답 행렬로 목표를 계산한다. 외력의 방향과 크기를 제어기에 전달하지 않는다. 이는 PPO 성능의 증명이 아니라 action의 제어 가능성을 확인하는 기준 제어기다.

0.5~0.75 N, 0.5~1초 외력에서 자기접촉·포화·접지 손실 없이 능동 제어의 개선을 확인했다. 0.75초 필터에서도 최대·누적 오차는 줄었으나 외력이 끝난 뒤 복구가 늦어질 수 있었다. 0.15초 필터는 진단에서 최대·누적 오차와 복구 시간을 함께 개선했다. 기본 필터는 여전히 0.75초다. 다음 단계에서 안전성 검증을 유지하며 별도 설정으로 비교하는 것을 추천한다.

기존 PPO의 자기접촉은 수십 마이크로미터 깊이로 여러 물리 step 동안 지속됐으며 단순한 한 step 수치 접촉으로 볼 근거가 없다. CAD 확인 전에는 관통 발생 즉시 실패 종료하는 방식을 우선 추천한다. 접촉 깊이와 시간 문턱을 만들어 알려진 간섭을 허용하지 않는다. 진단 이후 Standing V2에서 관통을 포함한 활성 자기접촉의 실패 종료를 구현했다.

기립 복구의 다음 목표는 자연 평형을 기준으로 외란 중 오차와 복구를 줄이는 것이다. 0° 몸통 수평 유지가 필요하면 별도 목표로 평가한다. actor의 42차원 관측은 이번 자세 제어에 필요한 정보를 제공했으나 제자리 복귀를 보장하지 않는다. 위치 복귀를 목표로 추가하기 전에는 실기에서 이용 가능한 속도·위치 추정 방법을 먼저 정해야 한다.

## Standing V2 실행과 평가 범위

V2의 주 목표는 자연 평형 대비 자세 오차를 줄이는 것이다. yaw 추종은 추가하지 않는다. projected gravity는 roll/pitch 목표만 반영한다. 기존 action 12차원과 observation 42차원, 네트워크 및 PPO hyperparameter는 유지한다. `--smoothing-tau`로 0.75/0.30/0.15/0.05초를 선택할 수 있으며 기본은 0.75초다.

| V2 단계 | 외력 크기 | 지속 시간 |
|---|---:|---:|
| A | 0.5 N | 0.5초 |
| B | 0.5 N | 1.0초 |
| C | 0.75 N | 0.5초 |
| D | 0.75 N | 1.0초 |

학습에서는 episode마다 네 방향 중 하나를 무작위로 선택한다. 평가는 16개 환경에서 +x/−x/+y/−y를 각각 4개씩 배정한다. 정책별로 동일한 seed와 초기 오차, 외력 시작 시점을 사용한다. reset 범위는 기존 Stage 1을 유지한다.

```bash
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.train --num-envs 64 --iterations 100 --v2-stage A --smoothing-tau .15 --episode-seconds 10 --seed 314 --log-dir /tmp/microdog_v2/ppo
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.evaluate --v2-stage A --smoothing-tau .15 --seed 2026 --output /tmp/v2_zero.json
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.evaluate --v2-stage A --smoothing-tau .15 --seed 2026 --scripted --output /tmp/v2_scripted.json
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.evaluate --v2-stage A --smoothing-tau .15 --seed 2026 --checkpoint /tmp/microdog_v2/ppo/checkpoint_100.pt --output /tmp/v2_ppo.json
```

학습 전에 zero-action과 scripted 기준 제어기로 초기 오차와 각 외력을 결합한 안전성을 검증한다. scripted 제어기는 학습 action이나 보상에 사용하지 않는다. 실패 종료 이유는 `fall`, `self_collision`, `invalid_state`, `abnormal_ground_contact`, `timeout`으로 기록한다.

핵심 자세 오차는 목표 대비 roll/pitch 오차 벡터의 크기이며 단위는 도다. `peak_orientation_error`는 외력 시작 이후 episode가 끝날 때까지의 최대 오차를 환경별로 계산한 평균이다. `integrated_orientation_error`는 같은 구간의 오차 적분으로 단위는 도·초다. 외력 적용 중의 최대·적분 오차와 전체 episode 지표도 별도로 저장한다. 초기 기울기가 외란 응답 비교를 지배하지 않도록 구간을 구분한다. 생존·접촉 조건이 나빠진 정책은 오차나 return만으로 개선 판정을 하지 않는다.

평가 seed는 2026/2027/2028, 학습 seed는 314다. 각 정책과 checkpoint를 48 episode로 비교한다. 모델·standalone 파일은 수정하지 않는다. 학습 결과가 개선되지 않으면 iteration을 자동으로 늘리지 않는다.

### V2 초기 실험 결과

Stage A에서 64개 환경, 100 update, rollout 16, 학습 seed 314를 사용했다. 네트워크와 PPO 설정은 기존과 같았다. 초기 오차와 네 단계 외력을 결합한 zero-action/scripted 안전성 평가 384 episode를 먼저 통과했다. 기준 제어기는 PPO 학습에 사용하지 않았다.

평가 seed 2026/2027/2028의 48 episode 평균에서 zero-action의 최대·누적 자세 오차는 0.31859°/0.19566 도·초, scripted 기준은 0.14598°/0.07068 도·초였다. 반면 최종 PPO는 0.76452°/5.09488 도·초로 악화됐다. 생존율은 100%지만 0.1° 복구 기준의 성공률은 0%였다. 이 정책을 기립 복구 성공 정책으로 판정하지 않는다.

학습 중 자기접촉 3건은 모두 실패 종료로 처리했고, 평가 checkpoint들의 자기접촉과 포화는 0이었다. scripted 제어기의 return 개선은 약 0.03%에 불과했다. 목표의 일치만으로 정책 학습이 보장되지 않으며, 다음 단계에서는 보상 대비와 정책 update·정규화·기여도 할당을 점검해야 한다. 성능 악화의 원인을 하나로 확정하지 않는다. 이번 실험 뒤 추가 iteration이나 재학습은 수행하지 않았다.

## PPO 학습 신호 진단

동결된 V2 checkpoint의 보상·행동·관측 정규화, 소수 갱신의 실제 advantage·KL·clipping·gradient, 동일 상태의 유한 시간 보상, 임시 지도 회귀를 계측한다. production 보상과 PPO 설정은 변경하지 않는다. 진단은 새 100회 PPO 학습을 실행하지 않으며 기존 checkpoint를 덮어쓰지 않는다.

```bash
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.diagnose_ppo --checkpoints /tmp/microdog_v2/ppo --output /tmp/microdog_ppo_diagnostics --part all
```

`all`은 독립 checkpoint 복사본의 PPO 갱신 총 10회와 임시 지도 회귀를 포함한다. `--part`로 개별 진단을 선택할 수 있다. 원시 배열·임시 회귀망·JSON은 지정한 output에 저장하고 같은 경로의 이전 진단 결과는 덮어쓴다. 모델, 원본 checkpoint, production 정책은 수정하지 않는다. 상세한 측정값·한계·V3 최소 변경안은 [PPO 진단 보고서](PPO_DIAGNOSTICS.md)에 정리했다.

## Standing V3-A: 관측 정규화 준비 후 고정

기본 `running` 경로는 V1/V2의 동작을 유지한다. `--normalization warmup-frozen`을 선택하면 초기 normalizer를 고정한 확률정책으로 실제 관측을 수집하고, 전체 자료의 평균·모집단 분산을 두 normalizer에 한 번 적용한다. 이후 rsl_rl의 `until=count` 제한으로 학습 모드에서도 통계 갱신을 차단한다. `eps=0.01`과 기존 checkpoint tensor 형식은 유지한다. 고정 모드는 `infos.normalization`에 저장하며 평가 및 진단 loader가 이를 복원한다.

```bash
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.train --num-envs 64 --iterations 100 --v2-stage A --smoothing-tau .15 --episode-seconds 10 --seed 2718 --normalization warmup-frozen --warmup-steps 500 --warmup-seed 12718 --log-dir /tmp/microdog_v3a/ppo
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.compare_standing --v2 /tmp/microdog_v2/ppo --v3 /tmp/microdog_v3a/ppo --output /tmp/microdog_v3a/evaluation
```

통계 수집 중 가중치·통계 불변, 반복 관측 일치, 실제 rollout의 갱신 전 KL을 검사한 후에만 PPO 학습한다. 모든 학습 갱신에서 통계와 갱신 전 KL을 다시 확인한다. 전체 통계를 float64로 계산하고 저장 buffer 형식으로 변환하여 작은 중력 분산의 소실을 방지한다. 보상·action·관측 정의·필터·외란·PPO hyperparameter는 유지하며 seed만 이번 실험의 명시값으로 설정한다.

100회 실험은 갱신 전 KL 문제를 제거했지만 최종 정책의 zero 대비 복구 개선을 달성하지 못했다. 추가 iteration은 실행하지 않았다. 상세 결과와 다음 비교 후보는 [Standing V3-A 보고서](STANDING_V3A.md)에 기록했다.

## Standing V3-B: 자세 보상 scale 비교와 25회 판정

`StandingCfg.orientation_reward_scale` 및 `--orientation-reward-scale`로 scale을 선택한다. 기본값은 기존 `0.05`이며 V3-B에서는 `0.01`을 명시한다. 나머지 보상항·weight·물리·PPO 설정·정규화 준비 방식은 유지한다. 기존 checkpoint 평가는 기본값으로 실행할 수 있으며, 버전별 return을 비교할 때는 모든 제어기에 같은 scale을 적용한다.

```bash
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.reward_validation --trajectories /tmp/microdog_v3a/evaluation --output /tmp/microdog_v3b/preflight
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.train --num-envs 64 --iterations 100 --v2-stage A --smoothing-tau .15 --episode-seconds 10 --seed 31415 --normalization warmup-frozen --warmup-steps 500 --warmup-seed 41415 --orientation-reward-scale .01 --gate-after-25 --v2-baseline /tmp/microdog_v2/ppo --v3a-baseline /tmp/microdog_v3a/ppo --log-dir /tmp/microdog_v3b/ppo
```

학습 상한은 100회다. 먼저 25회만 실행하고 별도 프로세스에서 세 seed의 동일 48 episode를 평가한다. 이때 학습 환경·optimizer·난수 상태를 보존한다. zero보다 최대 오차가 1.5배, 누적 오차가 2배를 넘으면서 국소 자세 feedback 부호가 틀리거나 요청 clipping이 25%를 넘으면 조기 실패로 판정한다. 이 조건을 충족하지 않고 안전한 개선 징후가 있을 때만 남은 75회를 연장한다. 구체적인 조건과 결과는 `gate.json`에 기록한다.

저장 trajectory 재점수화는 물리를 다시 실행하지 않고 보존된 자세와 나머지 보상항을 사용한다. 실제 평형 기준 민감도와 네 외란 방향의 1·5·10·25·50 step 순서도 학습 전에 검사한다. 출력 자료와 checkpoint는 `/tmp`에 보관하며 실험을 다시 실행할 때는 새로운 출력 경로를 선택한다. 상세 결과는 [Standing V3-B 보고서](STANDING_V3B.md)에 기록한다.

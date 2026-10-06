# Standing V6: Advantage·credit 검증과 imitation 진단

timeout bootstrap 오류를 수정했다. 새로 시작한 V6-Fix25는 실패했지만, scripted actor를 재현한 뒤 PPO를 5회 갱신한 정책은 세 seed 모두에서 zero를 이겨 SUCCESS 기준을 충족했다. 이 시점에서 추가 학습과 state-dependent std 실험을 중단했다. 다만 PPO 갱신 중 누적 자세 오차와 평형 명령 편향이 증가했다. 좋은 정책의 발견 문제와 갱신·보상 신호의 품질 문제를 함께 남긴 결과다.

[V5 커밋]

- commit: `5ac6a128efdb2336b3b62d14e4780185d021c325`
- 메시지: `Add standing V5 policy distribution diagnostics`
- 예상 15개 파일을 SHA256과 Git diff로 확인하고 47개 테스트·CPU 회귀·16개 checkpoint 검증 후 일반 push했다.
- branch: `feature/standing-rl`; upstream: `origin/feature/standing-rl`.
- push 직후 작업 트리는 clean이었고 HEAD와 원격 feature branch가 일치했다.
- `main`과 원격 `main`은 `f2df8dda498aa098d56582bd43c0661dc78e1be2`를 유지한다.
- 이후 V6 변경은 stage·commit·push하지 않았다. 기존 V5 문서는 당시 작성한 기록이다.

[Advantage·credit 검증]

실제 설치된 rsl_rl 5.0.1의 `PPO.act/process_env_step/compute_returns/update`와 환경의 auto-reset 경로를 읽고 실행했다. actor와 critic은 파라미터를 공유하지 않는다.

| 처리 단계 | 실제 tensor shape 또는 처리 |
|---|---|
| 관측 | actor/critic 각각 `[64,42]`; state 30D 정규화, applied action 12D identity |
| 분포 | latent μ·σ `[64,2]`; σ는 상태와 무관한 축별 학습 scalar |
| 정책 명령 | `a=tanh(u)` `[64,2]`; storage도 bounded action 저장 |
| 물리 입력 | basis `[12,2]` → 12D smoothing → 12개 leg target; head/tail 0 |
| value | 수집 시 `[64,1]` |
| reward/done/timeout | 환경 `[64]`, storage `[32,64,1]` |
| GAE·return | storage `[32,64,1]`; 16개 block, 32,768 transition 기록 |
| PPO minibatch | `[1024,42]` 관측, `[1024,2]` action; 2 epoch × 2 minibatch |

`gamma=0.99`, `lambda=0.95`, `clip=0.2`를 유지했다. 수정 경로의 수식은 다음과 같다.

```text
r_boot = r + gamma * true_timeout * V(terminal_observation)
m = 1 - done
delta_t = r_boot_t + gamma * m_t * V_next - V(s_t)
A_t = delta_t + gamma * lambda * m_t * A_next
return_t = A_t + V(s_t)
A_norm = (A - mean(A)) / (std(A) + 1e-8)
ratio = exp(log_prob_new(a) - log_prob_old(a))
L_surrogate = mean(max(-A_norm*ratio, -A_norm*clip(ratio,1-eps,1+eps)))
```

정규화는 각 32×64 rollout 전체에 적용하며 `std`는 PyTorch 기본 표본 표준편차다. failure·self-collision은 bootstrap하지 않는다. timeout은 terminal 관측 value를 한 번 더하고 GAE 연결을 끊는다. 다음 episode의 reset 관측으로 advantage가 이어지지 않는다. 실패와 timeout이 겹치면 실패가 우선한다.

손계산 테스트는 비종료 2-step return `[4.564,4.7]`, failure return `0.2`, timeout return `3.8`, 실패·timeout 동시 발생을 확인했다. 실제 수집에서 timeout 64건·failure 1건을 기록했다. 수정 뒤 손계산 GAE와 실제 raw advantage의 최대 차이는 `2.3842e-7`이다.

`rollout.npz`에 V, GAE target, bootstrap 없는 32-step truncated MC, 올바른 terminal/cutoff bootstrap을 사용한 MC, TD residual, raw/normalized advantage를 모두 저장했다. 아래 RMSE·bias는 bootstrap MC 기준이고 EV는 GAE target 기준이다. 독립적인 무한 horizon 정답이 아니며 critic 오차의 확정적 원인 분해로 해석하지 않는다.

| 구간 | 표본 | value bias | RMSE | EV (GAE) | EV (bootstrap MC) | raw advantage 평균 / 표준편차 |
|---|---:|---:|---:|---:|---:|---:|
| 외란 전 | 8982 | +0.024535 | 0.559944 | -2.655514 | -3.547184 | -0.017668 / 0.554878 |
| 외란 중 | 1600 | +0.073557 | 0.336674 | -0.491826 | -1.670027 | -0.054632 / 0.239767 |
| 종료 후 첫 1초 | 3200 | +0.069154 | 0.161704 | -0.409227 | -1.484097 | -0.045682 / 0.116347 |
| 종료 1초 이후 | 18986 | +0.084473 | 0.098020 | 0.424539 | 0.130324 | -0.055309 / 0.026811 |

| 구간 | normalized advantage 평균 / 표준편차 | TD 평균 / 표준편차 | bootstrap 없는 MC 대비 bias / RMSE |
|---|---:|---:|---:|
| 외란 전 | +0.005191 / 1.032271 | +0.023482 / 0.538372 | +4.013052 / 4.072308 |
| 외란 중 | -0.060014 / 1.316995 | -0.006652 / 0.047755 | +4.214498 / 4.241027 |
| 종료 후 첫 1초 | -0.081425 / 0.984310 | -0.004353 / 0.031033 | +4.205972 / 4.222857 |
| 종료 1초 이후 | +0.016326 / 0.953824 | -0.005859 / 0.006728 | +4.282430 / 4.295879 |

외란 전·중·직후 critic의 EV가 음수다. 짧은 MC의 마지막 value를 같은 critic으로 bootstrap한 한계가 있으므로 별도의 장기 return 정확도를 증명하지 않는다.

[Empirical Q 검사]

대표 상태 6개와 실제 rollout 상태 22개, 총 28개를 검사했다. 실제 상태는 외란 전 6개·중 4개·직후 6개·안정 구간 6개다.

각 상태에서 9×9 grid 81개와 PPO mean/sample, scripted, wrong, zero, 중앙차분 4개, mean 반복 1개를 합쳐 91개 분기를 실행했다. 첫 명령을 1회 적용하고 이후에는 zero 명령을 사용했다. horizon은 1/5/10/25/50 policy step, 즉 0.02/0.1/0.2/0.5/1초다. γ0.99 discounted reward, peak tilt, integrated tilt, 평균 orientation error를 저장했다. 50-step 안에 종료된 분기는 없었다.

복원 항목은 time, qpos/qvel, act/ctrl, qacc_warmstart/qacc, applied force, equality/mocap 상태, 12D smoothing/request/target, episode counter, push 시점/벡터, RNG다. `forward()`로 파생 contact·inertia를 재계산하고 관측 일치를 확인했다. 해당 모델에는 plugin/userdata 상태가 없고 act/mocap/equality 배열은 비어 있다. 다른 명령을 실행한 뒤 동일 snapshot의 5-step 분기를 다시 실행해 qpos/qvel/filter/reward가 허용오차 내에서 일치하는 테스트가 통과했다.

다음 순위는 81개 grid 중 해당 특수 명령보다 return이 큰 명령 수에 1을 더한 값이다. 중앙차분은 현재 mean 부근 action ±0.05이며 clipping이 있으면 실제 간격으로 나눈다.

| 대표 상태 | 50-step 최적 grid (pitch, roll) | PPO mean 순위 | scripted 순위 | wrong 순위 | zero 순위 | dQ/dpitch | dQ/droll |
|---|---|---:|---:|---:|---:|---:|---:|
| 자연 평형 | [-1.0, 0.0] | 14 | 27 | 27 | 27 | -3.338e-05 | -1.431e-05 |
| −pitch 외란 | [1.0, 0.0] | 69 | 1 | 73 | 38 | +4.947e-04 | +0.000e+00 |
| +pitch 외란 | [-1.0, 0.0] | 51 | 2 | 73 | 37 | -6.580e-04 | +1.431e-05 |
| +roll 외란 | [-0.5, 1.0] | 28 | 6 | 77 | 39 | -2.265e-05 | +5.698e-04 |
| −roll 외란 | [-0.75, -1.0] | 46 | 6 | 77 | 37 | -5.960e-06 | -5.829e-04 |
| 복구 중 | [-1.0, 0.0] | 14 | 47 | 14 | 28 | -3.457e-05 | -3.219e-05 |

| horizon | 유효 실제 표본 | normalized advantage·Q 차이 상관 | 부호 일치율 | Q gradient·Δμ 평균 cosine | 양의 정렬 비율 |
|---|---:|---:|---:|---:|---:|
| 1 | 18 / 22 | 0.207605 | 55.56% | -0.603932 | 9.09% |
| 5 | 22 / 22 | 0.185427 | 45.45% | -0.614913 | 9.09% |
| 10 | 22 / 22 | 0.208977 | 45.45% | -0.614419 | 4.55% |
| 25 | 22 / 22 | 0.218458 | 45.45% | -0.635322 | 4.55% |
| 50 | 22 / 22 | 0.211009 | 45.45% | -0.638106 | 9.09% |

50 step의 raw advantage 상관은 0.055711, raw 부호 일치율은 40.91%다. 실제 표본의 평균 dQ/da는 pitch +6.672415e-04, roll -1.797350e-04이며 실제 PPO update의 평균 Δμ는 pitch -0.039760, roll +0.012471다. 비교할 때 `dQ/dμ=(dQ/da)*(1-a²)`를 사용했다.

Δμ는 외란이 포함된 실제 rollout block의 storage·optimizer를 복사한 뒤 native PPO를 1회 갱신해 측정했다. 각 표본의 action/raw advantage/normalized advantage/log_prob/ratio는 JSON과 snapshot에 저장했다. 갱신 전 KL=0·clip=0이고 표본 ratio 범위는 약 0.999997~1.000002다.

동일 mean·zero 명령의 중복 분기로 수치 잡음을 측정했다. 부호 판정에는 `5 × max(중복 return 차이,1e-7)`보다 큰 Q 차이만 사용했다.

여기서 부호는 `Q(sample)-Q(mean-action)`와 비교했다. zero continuation의 유한 Q와 실제 stochastic continuation의 GAE advantage는 서로 다른 양이므로 45.45%를 PPO 수식 오류의 직접 증명으로 해석하지 않는다. 표본은 단일 checkpoint·학습 seed의 22개 상태이며 다른 seed·정책 전체로 일반화할 수 없다.

평형에서는 pitch −1 grid의 return이 zero보다 약 2.29e−5 높지만 누적 tilt는 0.00864 대 0.00067°·s로 더 크다. 작은 return 차이와 자세 오차의 순서가 다를 수 있다.

[Actor gradient 기여]

전체 actor gradient는 각 phase 표본만 남긴 surrogate를 전체 32,768개 표본 수로 나눠 계산했다. 추가로 mean head score의 표본별 근사 기여 `−A_norm*(atanh(a)−μ)/σ²`를 기록했다. 아래 두 norm은 서로 다른 지표다. head bias gradient 부호의 반대가 단순 gradient descent 방향이며, 실제 Adam·hidden layer 갱신에 의한 Δμ와 동일하지는 않다.

| 구간 | 전체 actor gradient norm | mean head gradient 합 (pitch, roll) | 표본별 head norm 합 |
|---|---:|---|---:|
| 외란 전 | 0.169630 | +1267.16, +125.80 | 31243.16 |
| 외란 중 | 0.040825 | +259.25, -53.22 | 8341.11 |
| 종료 후 첫 1초 | 0.047386 | +269.11, -318.85 | 11041.84 |
| 종료 1초 이후 | 0.810401 | +4473.56, -5808.11 | 71568.20 |

안정 구간의 전체 gradient norm은 외란 중의 약 19.9배다. 표본 비율·상쇄·network Jacobian 영향이 함께 포함된다. 평상시 기여가 더 크지만 phase별 gradient 모두를 물리적으로 잘못됐다고 단정하지 않는다.

같은 rollout에서 raw/normalized advantage 전체 gradient의 cosine은 0.494837다. 정규화가 방향을 상당히 바꾼다. phase별 평균 부호도 달라질 수 있다. production의 advantage normalization은 변경하지 않았다.

[구현 오류]

native `PPO.process_env_step`는 timeout reward에 terminal value 대신 이미 저장된 현재 state value `V(s_t)`를 더했다. 환경은 auto-reset 이전의 `terminal_observation`을 제공하므로 이 값으로 교정할 수 있었다. `rl.timeout_bootstrap.TerminalBootstrapPPO`가 정확한 terminal value를 더한 뒤 native timeout 보정을 제거해 중복 bootstrap을 막는다. GAE recurrence·reward 함수·실패 종료는 그대로다.

- 수정 전 64개 timeout reward 오차: 평균 −0.00088186, 범위 −0.01491105~+0.01862047.
- 수정 후 최대 절대 오차: 4.1544e−7. raw advantage 손계산 오차: 2.3842e−7.
- V6 학습은 `--terminal-bootstrap`을 명시한다. checkpoint metadata로 교정 경로를 복원한다. 기존 metadata가 없는 checkpoint는 과거 native 경로를 복원해 비교 의미를 보존한다. 향후 학습 명령에도 이 옵션이 필요하다.
- gamma/GAE 부호, failure·self-collision·reset 연결에서 추가 구현 오류는 확인하지 못했다.
- V6-Fix: 새 정책 64 env × rollout32 × 25회. peak 0.618535°, integrated 3.836558°·s, 복구 없음, FAIL. V5-A25보다 약 0.97%/1.04% 악화해 100회로 연장하지 않았다.
- V6-Fix 학습 중 자기충돌 2건은 실패 종료로 처리했다. 최종 48 episode 평가의 자기접촉·비정상접촉·포화는 0이었다.

[Imitation 진단]

42D 관측과 2D scripted basis 명령을 4개 seed에서 수집했다. 훈련 seed 61026/61027/61028의 초기 trajectory 192개·관측 96,000개, holdout seed 61029의 초기 trajectory 64개·관측 32,000개를 분리했다. reset perturbation, 네 외란 방향, onset/active/recovery, 평형 부근을 포함한다.

초기 3번째 policy step에서 자기접촉으로 auto-reset된 환경이 훈련 2개·holdout 2개 있었다. 초기 경계를 다시 실행해 종료 이유를 확인했다. 따라서 실제 episode 구간 수는 훈련 194개·holdout 66개다. 학습/검증 seed는 섞이지 않았다. 이 초기 실패는 48 episode의 최종 평가 안전성 결과와 구분한다.

같은 42→64→64→2 ELU actor로 `MSE(tanh(mu), scripted_action)`를 4,000회 학습했다. phase별 batch를 균등하게 뽑았고 std0.3은 사전학습 중 고정했다. PPO optimizer는 초기 상태이고 critic은 V5-A checkpoint0의 정상 초기 가중치로 별도 준비했다. actor/critic normalizer 32,000개 통계와 identity mask를 그대로 유지했다.

holdout MSE: 8.60066284e-05. reset / 외란 전 / 외란 중 / 복구 / 안정 구간 MSE: 1.196552e-03, 1.214277e-05, 2.745096e-04, 8.950742e-05, 1.364140e-06.

| PPO 갱신 | peak (°) | integrated (°·s) | 복구 (s) | 동일 holdout Δμ 평균 (pitch, roll) | 국소 gain (roll, pitch) |
|---|---:|---:|---:|---|---|
| 0 | 0.147681 | 0.071731 | 0.010082 | +0.000000, +0.000000 | +2.384763, -2.776011 |
| 1 | 0.149268 | 0.077794 | 0.010082 | -0.004537, +0.011329 | +2.385756, -2.788984 |
| 2 | 0.149500 | 0.074231 | 0.009666 | +0.003272, -0.002692 | +2.383272, -2.788744 |
| 3 | 0.150257 | 0.103532 | 0.012166 | -0.015978, -0.041528 | +2.381479, -2.786126 |
| 4 | 0.149621 | 0.125574 | 0.018832 | -0.056425, -0.023774 | +2.394680, -2.783032 |
| 5 | 0.151738 | 0.132660 | 0.020916 | -0.056417, -0.039584 | +2.387666, -2.773207 |

모든 갱신에서 세 seed 각각의 peak·integrated가 zero보다 낮고 생존율 100%, 자기접촉 0, 비정상접촉 0, 포화 0을 유지했다. 복구도 성공했다. 따라서 빠른 붕괴 분기에 해당하지 않는다. 가장 좋은 PPO 갱신 checkpoint는 integrated 기준 1회이며, 순수 imitation 0회가 더 좋다.

5회 후 integrated는 초기보다 84.94% 커졌다. 같은 평형 관측의 bounded 명령은 초기 `[+0.002324,−0.000603]`에서 `[−0.047434,−0.038396]`으로 이동했다. 국소 피드백 부호는 계속 roll 양수·pitch 음수다. 정책 유지와 성능 악화를 함께 보고하며 장기 유지가 입증됐다고 결론 내리지 않는다.

저장된 동일 평가 trajectory의 보상항을 float64로 합산했을 때, 5회−0회의 episode 평균 변화는 다음과 같다.

| 보상항 | 변화 |
|---|---:|
| `upright` | -4.56563469e-04 |
| `height` | +0.00000000e+00 |
| `pose` | +5.81885019e-04 |
| `joint_velocity` | +8.00451488e-07 |
| `action_rate` | +1.01702627e-08 |
| `effort` | +3.11295335e-06 |

저장 reward 합의 평균 변화는 +1.29550618e-04다. pose 점수 증가가 upright 감소를 상쇄하는 관측 근거다. 환경의 float32 누적 return과 작은 수치 차이가 있으므로 근소한 aggregate return 변화만으로 유의한 개선을 주장하지 않는다. 보상은 수정하지 않았다.

[State-dependent exploration]

현재 std는 μ network와 독립된 두 학습 scalar이며 모든 상태에서 동일하다. 처음 0.3, 5회 후 pitch0.295356·roll0.298105다. scripted의 안정 구간 target RMS는 pitch0.003288·roll0.002408, 외란 중에는 0.364323·0.403237이다. 상태에 따른 필요 correction 크기의 차이는 크지만 latent σ와 bounded target RMS를 같은 단위의 정확한 탐색 분산으로 동일시하지 않는다. smoothing과 tanh의 효과도 포함된다.

좋은 정책을 5회 유지해 exploration/discovery 후보를 시험할 조건은 생겼다. 그러나 PPO 1~5가 이미 사용자 SUCCESS 기준을 충족했고 이후 불필요한 실험 중단 지시가 적용돼 state-dependent std 구현·새 25회·100회 학습은 하지 않았다. 이 원인이 확정되거나 state-dependent std가 효과적이라고 주장하지 않는다.

[기준선 비교]

seed2026/2027/2028, 각각 16 episode, Stage A0.5 N×0.5초,10초, 같은 초기 관측·외란 시점을 확인했다. peak·integrated는 기존 평가 정의의 외란 시작 이후 지표다. zero/scripted는 기존 12D 기준 제어기이며 PPO2D basis와 구분한다. V3-E75는 저장된 동일 조건의 V5 평가를 재사용했고 checkpoint 복원을 다시 확인했다.

| 정책 | peak (°) | integrated (°·s) | 복구 (s) |
|---|---:|---:|---:|
| zero | 0.318583 | 0.195665 | 0.126749 |
| scripted | 0.145980 | 0.070683 | 0.010082 |
| V3-E75 | 0.327888 | 0.415812 | 0.188416 |
| imitation 0회 | 0.147681 | 0.071731 | 0.010082 |
| V6-Fix25 | 0.618535 | 3.836558 | 복구 없음 |
| imitation → PPO 1회 | 0.149268 | 0.077794 | 0.010082 |
| imitation → PPO 5회 | 0.151738 | 0.132660 | 0.020916 |

[최종 원인 판정]

- 구현 오류: terminal bootstrap 오류가 확인됐으나 이를 고치는 것만으로 초기 정책 발견 문제가 해결되지는 않았다.
- critic: 외란 관련 구간의 value 예측이 좋지 않고 단기 credit의 신뢰도가 낮다. 독립적인 장기 return 검증은 아직 필요하다.
- advantage·갱신: 정규화 전후 gradient 방향 차이, 평상시 기여 우세, zero-continuation Q와 업데이트의 음의 정렬이 관측됐다. continuation 차이 때문에 이 지표만으로 PPO 수식 오류를 단정하지 않는다.
- discovery·exploration: 같은 관측과 actor 구조로 좋은 scripted 정책을 재현하고 PPO가 5회 유지해 표현력 부족보다 좋은 초기 정책 발견 문제가 지지된다. 상태별 탐색 크기 차이를 시험할 근거는 생겼지만 효과는 미검증이다.
- objective: 좋은 정책에서도 upright 감소를 pose 증가가 상쇄하고 integrated가 85% 증가했다. 따라서 주원인을 탐색 하나로 확정할 수 없으며 보상 순위와 자세 목표의 일치가 추가로 필요하다.
- 이번 실험에서 가장 가능성 높은 조합은 약한 자세 가치 대비·critic/정규화 신호의 품질·초기 policy discovery 문제다. 좋은 정책을 유지할 수 없는 보편적인 PPO 결함은 확인하지 못했다.

[회귀 검사]

전체 unittest52개 통과. 기존/V6 checkpoint18개에서 distribution·optimizer·고정 정규화 복원 통과.12개 평가 trace의 초기 관측·외란 시점 일치, finite 관측·bounded 명령 확인. actor/critic 분리, basis·의미 정규화·자기충돌 종료를 유지했다.

CPU 지정 Python의 10초 standalone 초기화 경로에서 `nq24/nv23/nu17/mass1.173 kg`, 최종 높이 0.192719714 m, 최대 leg torque0.102678368 N·m, saturation0, self-contact0, NaN없음, warning7종모두 0. GUI viewer는 실행하지 않았다.

XML·ctrlrange·neutral keyframe·joint ordering·actuator mapping·collision·geometry·kinematics·기존 standalone 파일은 V5 commit과 byte 단위로 같다. 새 package 설치, Walking 구현, main merge, destructive reset은 수행하지 않았다.

[Git 변경]

수정: `rl/README.md`, `rl/config.py`, `rl/train.py`, `rl/ablation.py`.
신규: `rl/timeout_bootstrap.py`, `rl/credit_audit.py`, `rl/imitation_diagnostic.py`, `rl/STANDING_V6.md`, `tests/test_gae_bootstrap.py`, `tests/test_gpu_state_snapshot.py`.
V6 변경은 unstaged다. `git diff --check` 통과. 현재 HEAD·upstream은 V5 commit을 유지하고 원격 feature/main hash도 조회해 확인했다. V6 stage·commit·push는 하지 않았다.

[다음 단계]

성공한 imitation0회와 PPO1회 checkpoint를 기준선으로 보존한다. 다음 변경의 우선순위는 같은 continuation·복수 sample의 Q/advantage 비교, 장기 MC와 critic 정확도, pose/upright 보상 순위의 검증이다. 검증된 terminal bootstrap 옵션을 계속 명시하고, 근거 없이 보상·std·학습률을 동시에 조정하지 않는다. state-dependent std는 후속 승인 범위의 별도 단일 변경 후보이며 이번에는 실행하지 않았다.

[마지막 질문 10개]

1. 좋은 action과 advantage 부호가 일치하는가? zero-continuation의 `Q(sample)−Q(mean)` 기준 50-step 일치율 45.45%로 낮다. 실제 stochastic continuation advantage의 정답과 동일한 비교는 아니다.
2. actor gradient가 return 개선 방향인가? 검사한 실제 상태 22개에서 평균cosine−0.6381, 양의 정렬 9.09%다. 유한 horizon·continuation 차이를 고려해야 한다.
3. critic/GAE/bootstrap 오류가 있는가? timeout에 현재 value를 더하던 명확한 오류를 수정했다. 추가 GAE 부호·reset 연결 오류는 확인하지 못했다. critic 예측 품질 문제는 별도로 남는다.
4. 평상시 transition이 더 큰 잘못된 gradient를 만드는가? 안정 구간 전체 gradient norm은 외란 중의 약 19.9배다. 부정적 정렬과 함께 의심 근거가 있으나 모든 평상시 sample이 잘못됐다는 결론은 아니다.
5. scripted를 actor가 재현하는가? 가능하다. holdout MSE8.60e−5이고 48 episode의 peak/integrated가 scripted에 가깝다.
6. 좋은 imitation 정책을 PPO가 유지하는가?5회까지 세 seed에서zero보다 좋게 유지한다. integrated는 85% 증가해 부분 악화도 관측된다.
7. 유지한다면 discovery가 주원인인가? 좋은 정책 발견 문제를 지지하지만 critic·보상 정렬까지 배제해 단일 주원인으로 확정하지 않는다.
8. 파괴한다면 credit/update가 주원인인가? 이번에는 zero 기준의 빠른 붕괴가 없었다. 좋은 정책의 점진적 악화는 credit/objective 검증이 필요하다는 근거다.
9. state-dependent exploration 근거가 생겼는가? 조건과 상태별 correction 크기 차이라는 근거는 생겼다. SUCCESS 이후 중단 지시로 실제 효과를 시험하지 않았다.
10. 가장 먼저 고칠 것은? terminal bootstrap 교정 경로를 유지하고, 추가 알고리즘 변경 전에 자세 오차와 실제 보상 순위의 불일치 및 같은 continuation의 critic/advantage 정확도를 먼저 검증한다.

[자료와 재현]

자료는 `/tmp/microdog_v6`에 있다. `/tmp`의 영속 보관은 보장되지 않는다. 핵심 파일: `results.json`, `regression.json`, `before/after/rollout.npz`, `before/after/audit.json`, `after/landscape.json`, `after/landscape_arrays.pt`, `after/q_comparison.json`, `imitation/pretraining.json`, `imitation/updates.json`, `imitation/decision.json`, checkpoint와 평가 trace.

- 그래프: `/tmp/microdog_v6/q_landscape.png`, `/tmp/microdog_v6/imitation_ppo.png`.
- 단일 training seed42424이며 평가 seed3개만 사용했다. 다른 강도·형상·실기·Walking으로 일반화하지 않는다.
- 수정 전후 진단을 같은 설정·seed로 재실행했으며 done/timeout mask는 일치했다. 독립 GPU 실행의 최대 관측 차이는 3.69e−4, action 차이는 1.72e−3으로, bit 단위로 같은 trajectory는 아니다. 각 Q grid 내부는 동일 snapshot에서 분기했다.

```bash
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.credit_audit --checkpoint /tmp/microdog_v5/v5a/ppo/checkpoint_25.pt --output /tmp/microdog_v6_repeat/before
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.credit_audit --checkpoint /tmp/microdog_v5/v5a/ppo/checkpoint_25.pt --output /tmp/microdog_v6_repeat/after --terminal-bootstrap
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.train --num-envs 64 --iterations 100 --seed 42424 --warmup-seed 52424 --normalization warmup-frozen --warmup-steps 500 --v2-stage A --episode-seconds 10 --smoothing-tau .15 --orientation-reward-scale .05 --rollout 32 --previous-action-normalization identity --standing-basis /tmp/microdog_v4/subspace/basis.json --policy-distribution squashed --action-mapping identity --terminal-bootstrap --ablation-name v6fix --ablation-parent /tmp/microdog_v5/v5a/evaluation/summary.json --log-dir /tmp/microdog_v6_repeat/fix/ppo
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.imitation_diagnostic --output /tmp/microdog_v6_repeat/imitation
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m unittest discover -s tests -v
```

재현 출력은 새 경로로 지정한다. imitation 진단은 기본 V5-A0 normalizer와 `/tmp/microdog_v6/fix/evaluation/summary.json`의 zero 기준선을 사용한다. 평가 subprocess가 학습 환경·optimizer·RNG를 변경하지 않도록 분리했다. checkpoint 저장 전 logger writer 초기화 문제는 진단 스크립트에서 수정했고, 저장돼 있던 사전학습 actor를 재사용해 4,000회 학습을 반복하지 않았다.

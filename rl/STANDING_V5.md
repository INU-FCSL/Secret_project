# Standing V5: 정책 분포와 평균 편향 진단

V4는 commit·push했다. V5-A/B는 각각 25회에서 종료했으며 V5 변경은 stage·commit·push하지 않았다. 성공 정책은 없고 기존 V3-E75가 PPO 중 가장 좋은 물리 오차를 유지한다.

[V4 커밋]

- branch: `feature/standing-rl`
- commit: `e1aeae593ca8af7058afdfc49dc5a32bf0f06cc5`
- 메시지: `Add standing V4 action-space diagnostics`
- 승인된 13개 파일의 SHA256·Git 목록을 이전 manifest와 대조했다. 37개 테스트와 CPU 10초 회귀 통과 후 일반 `git push`를 수행했다.
- push 직후 HEAD와 `origin/feature/standing-rl`이 일치하고 working tree가 clean이었다.
- `main`과 `origin/main`은 `f2df8dda498aa098d56582bd43c0661dc78e1be2`를 유지한다.

[현재 Distribution 진단]

실제 rsl_rl 5.0.1 `GaussianDistribution`, `PPO.act`, `PPO.update` 소스와 wrapper를 읽었다. 상태별 MLP 출력 μ, 상태와 무관한 축별 `std_param` σ를 사용한다. σ는 직접 scalar parameter이며 학습되는 `log_std` parameter는 없다.

`u ~ Normal(μ, σ) → storage.actions=u → env clip(u) → 12×2 basis → 12D smoothing → ACTION_SCALE → joint target`

log probability는 `sum(log Normal(u;μ,σ))`, entropy는 Gaussian analytic entropy, PPO ratio는 `exp(log_prob_new(u)-log_prob_old(u))`다. basis 다음에 smoothing이 적용된다. 기존 raw latent와 결정적인 clip을 묶은 정책 최적화는 수학적으로 가능한 구조다. 다만 같은 로봇 명령에 여러 원시값이 대응하고 raw entropy가 실제 명령 다양성을 나타내지 못한다.

- V4-B25 σ(pitch, roll): `+0.280938, +0.266443`.
- `u=1.2,2.0,3.0`은 모두 첫 bounded 성분 `1`이다. 두 번째 성분을 고정한 log probability: `-21.882170, -45.096718, -85.517937`.

| 지표 | 원시 pitch / roll | bounded pitch / roll |
|---|---:|---:|
| 분산 | +0.148307, +0.282271 | +0.104347, +0.067139 |
| 표본 쌍 RMS 거리 | +0.478163, +0.635440 | +0.393262, +0.322953 |
| 고정 폭 histogram entropy (nat) | +3.159065, +3.288435 | +2.639384, +0.661239 |

동일 24,000개 관측에서 seed 1729의 확률 표본을 비교했다. histogram bin 폭은 0.0625다. 이 값은 이산 entropy이고 Gaussian conditional differential entropy 0.245664 nat와 직접 비교하지 않는다. 12D joint command는 rank 2 부분공간에 있으므로 주변 관절별 histogram을 12D joint entropy로 해석하지 않는다.

| V4-B25 관측 구간 | μ_pitch | μ_roll |
|---|---:|---:|
| 외란 전 | -0.385098 | +1.043843 |
| 외란 중 | -0.649089 | +1.514202 |
| 종료 후 첫 1초 | -0.714581 | +1.605774 |
| 종료 1초 이후 | -0.816936 | +1.690108 |

평가 checkpoint의 σ는 모든 상태에서 같다. 외란 뒤 μ가 0으로 복귀하지 않는 현상이 확인되지만, 이 고정 checkpoint의 시간 경로로 σ 변화가 μ drift를 일으켰다고 결론 내릴 수 없다.

기존 25회 학습의 축별 세부 gradient는 저장돼 있지 않았다. V4-B25 checkpoint의 optimizer를 복원한 독립 복사본에서 시작 0초·6초의 32-step rollout과 한 번의 업데이트를 각각 계측했다. 원래 checkpoint를 수정하지 않았다. advantage와 Normal mean score를 곱한 평균은 각각 `[+0.044548,+0.110720]`, `[-0.351232,+0.050796]`이다. 두 창에서 roll을 같은 방향으로 미는 신호가 보이지만 전체 V4 학습의 gradient 이력을 재구성한 것은 아니다.

V5-A/B에서는 25 update × 4 minibatch의 clipping 전 mean head bias gradient, 축별 weight gradient norm, 실제 scalar std gradient를 저장했다. 다음 값은 전체 minibatch 평균이다.

| 후보 | head bias gradient (pitch, roll) | head weight gradient norm | std gradient |
|---|---:|---:|---:|
| V5-A | +0.046808, -0.011597 | +0.491759, +0.323658 | +0.097450, +0.024634 |
| V5-B | +0.037963, -0.006220 | +0.488332, +0.366392 | +0.092202, +0.022954 |

head bias gradient가 항상 같은 부호인 것은 아니다. V5-A의 양수 비율은 pitch 57%, roll 47%이며, head bias만으로 drift를 설명할 수 없다. hidden layer와 head weight, previous action에 대한 피드백을 함께 봐야 한다. actor와 critic은 파라미터를 공유하지 않는다. critic은 advantage를 통해 actor 학습에 영향을 주며 직접 공유 gradient 경로는 없다.

[V5-A Squashed Gaussian]

- `u ~ Normal(μ,σ)`, `a=tanh(u)`로 실제 PPO action과 storage action을 bounded `a`로 통일했다.
- log probability: `sum(log Normal(atanh(a);μ,σ) - log(1-tanh(atanh(a))²))`.
- Jacobian은 `2*(log(2)-u-softplus(-2u))`로 안정적으로 계산한다.
- 기존 storage의 bounded action과 `(μ,σ)`를 유지해 raw latent용 새 buffer를 추가하지 않았다.
- float32에서 정확히 ±1로 반올림되는 꼬리만 한 ULP 안쪽으로 guard한다. rollout·update 모두 같은 inverse와 density 경로를 사용한다. 실제 평가 확률 표본에서 guard 발동은 0이었다.
- 환경 `identity`는 유한한 bounded 입력을 검사한 뒤 그대로 반환한다. tanh나 clip을 다시 적용하지 않는다.
- deterministic evaluation은 `tanh(μ)`다.
- entropy는 `H(Normal)+log Jacobian(u)`의 관측당 한 reparameterized 표본 추정치를 사용한다. 이는 bounded entropy의 MC estimator이며 μ·σ gradient를 보존한다. 같은 forward에서 반복 조회는 같은 tensor를 사용해 계측이 RNG나 entropy gradient를 바꾸지 않는다.
- 동일한 가역 tanh 아래 KL은 원래 Normal의 analytic KL과 같다. 수치 guard 꼬리에서는 이 KL이 underlying continuous distribution 기준이라는 한계가 있다.

공통 설정은 검증된 V4-B basis·scaling, 관측 42D, 실제 previous applied joint action 12D identity, 500/50 Hz·decimation 10, 10초 episode, Stage A 0.5 N×0.5초, smoothing 0.15초, orientation scale 0.05, rollout 32, 64×500 warm-up 후 freeze다. 네트워크 64×64 ELU, epoch 2, minibatch 2, γ0.99, λ0.95, clip0.2, entropy coefficient0.005, 초기 σ0.3, adaptive LR를 유지했다. training/warm-up seed는 42424/52424다.

범위·grid별 finite density·Torch transformed density와 일치·ratio/KL·inverse·경계 gradient·MC entropy·native storage 계측의 8개 distribution 테스트가 통과했다. actual preflight KL=0, ratio mean=1, clip fraction=0이며 모든 학습 update 전에도 KL=0이다.

| 후보 | checkpoint 0 최대 오차 (°) | checkpoint 0 누적 오차 (도·초) | checkpoint 25 최대 | checkpoint 25 누적 | 복구 |
|---|---:|---:|---:|---:|---|
| V5-A | 0.323032 | 0.647871 | 0.612610 | 3.796983 | 복구 없음 |
| V5-B | 0.323049 | 0.647874 | 0.615972 | 3.814298 | 복구 없음 |

V5-A는 parent V4-B25 대비 최대 12.65%, 누적 17.53% 개선했다. 명확한 악화로 중단한 것이 아니라 최대 개선이 15% 승격 기준에 못 미쳐 `FAIL`로 종료했다. 세 seed 모두 zero를 이기지 못했다. 100회 연장과 checkpoint 50/75/100 생성은 수행하지 않았다. 후보 내 물리 오차 최적 checkpoint는 25이며 성공 정책으로 채택하지 않는다.

| 후보 | roll 국소 gain | pitch 국소 gain | μ_pitch | μ_roll | σ_pitch | σ_roll |
|---|---:|---:|---:|---:|---:|---:|
| V5-A25 | -0.008579 | +0.015133 | -1.504664 | +0.338968 | 0.232664 | 0.268962 |
| V5-B25 | -0.010558 | +0.014338 | -1.544429 | +0.333438 | 0.230414 | 0.270413 |

gain은 bounded knee 차등 명령/자세 오차 1° 기준이다. 기존 물리 측정에서 필요한 부호는 roll 양수·pitch 음수다. 두 후보 모두 이 부호를 학습하지 못했다.

[평균 편향과 대칭성]

현재 관측의 |roll/pitch error|<0.05°와 3D angular speed<0.005 rad/s를 동시에 만족하는 trajectory 표본은 V4-B25 12개, V5-A25와 V5-B25 각각 1개였다. V5의 표본 1개로 평균 분포나 분산을 일반화하지 않는다. 지정 CPU MuJoCo에서 실제 자연 평형 상태를 구성한 독립 probe를 추가했다. 이 상태의 reference pitch 오차는 약 −1.31e−8°이고 scripted 2D 명령은 사실상 0이다.

| 후보 | 평형 μ (pitch, roll) | 평형 tanh(μ) |
|---|---:|---:|
| V5-A25 | -0.388517, +0.253803 | -0.370081, +0.248491 |
| V5-B25 | -0.378347, +0.239861 | -0.361271, +0.235365 |

| 후보 / 축 | 공통 offset | 보정 성분 | odd symmetry error | latent gain / ° |
|---|---:|---:|---:|---:|
| V5-A25 / roll | +0.248477 | -0.002139 | 0.496955 | -0.042784 |
| V5-A25 / pitch | -0.370008 | +0.010322 | 0.740015 | +0.206432 |
| V5-B25 / roll | +0.235351 | -0.002553 | 0.470702 | -0.051069 |
| V5-B25 / pitch | -0.361197 | +0.010642 | 0.722394 | +0.212847 |

오차 ±0.05°의 대칭 pair를 같은 평형 joint/filter 상태에서 검사했다. V5-A 공통 offset/보정 성분의 절댓값 비율은 roll 약116배, pitch 약36배다. zero에 가까운 scripted 평형 명령과도 큰 차이가 있어 V5-B의 조건이 충족됐다.

[V5-B Mean Regularization]

- 실행: V5-A가 SUCCESS가 아니고 평형 및 대칭 probe에서 편향 근거가 충분해 수행했다.
- coefficient: `0.0009273962086400983`. 후보 한 개만 사용했고 sweep하지 않았다.
- 초기 policy loss magnitude `abs(surrogate-0.005*entropy)=0.0015945186`, mean(μ²) `0.0515804980`를 측정했다.
- 목표 초기 penalty `0.0000478356`는 policy loss의 3%다. V5-B에서 실제 첫 update 평균 penalty는 `0.0000479328`이었다.
- environment reward는 그대로다. native PPO의 zero_grad 직후 penalty gradient를 계산하고 native loss.backward의 gradient와 더한다. 결합 gradient는 기존 actor clipping·Adam을 통과한다. coefficient0의 native 결과 일치와 명시적 전체 objective gradient와의 일치(최대 오차0)를 테스트했다.
- 동일 seed·환경·distribution·네트워크에서 mean regularization 하나만 추가했다. A/B 초기 network weights는 정확히 같다. 독립 GPU warm-up의 mean/std 차이 최대는 각각 1.84e−7/2.94e−7이며 물리 rollout의 미세 수치 차이를 고려해야 한다.
- V5-B25는 A25 대비 최대 0.55%, 누적 0.46% 악화했다. 평형 probe action 절댓값은 pitch 약2.38%, roll 약5.28% 줄었지만 전체 latent μ 편향은 오히려 증가했고 피드백 부호도 개선되지 않았다.
- 판정 FAIL. 100회 연장 없음. checkpoint25가 이 후보의 유일한 학습 평가 checkpoint다.

| 후보 | 전체 update 평균 KL | 평균 clip fraction | 최대 clip fraction | 최종 update entropy |
|---|---:|---:|---:|---:|
| V5-A | 0.010822 | 13.46% | 56.45% | -1.929279 |
| V5-B | 0.011303 | 14.00% | 55.18% | -2.037361 |

첫 실제 minibatch ratio≈1·clip0이고 갱신 전에는 항상 clip0이다. 최적화 뒤 일부 minibatch의 clipping은 높으므로 학습 변화량을 추가 진단할 여지가 있다. 분포 재평가 시점의 불일치로 생긴 clipping은 관측되지 않았다. entropy는 bounded differential entropy이므로 음수도 가능하다.

[Zero vs Scripted vs PPO]

evaluation seed2026/2027/2028 각각16개, 총48개 episode다. 새 기준 정책 평가와 V5 평가에서 initial observation과 push start는 정확히 같았다. 힘 크기·시간·균형 잡힌 방향도 같은 설정을 유지했다.

| 정책 | 최대 오차 (°) | 누적 오차 (도·초) | 복구 (초) |
|---|---:|---:|---:|
| zero | 0.318572 | 0.195662 | 0.126749 |
| scripted | 0.145980 | 0.070687 | 0.010082 |
| V3-E75 | 0.327888 | 0.415812 | 0.188416 |
| V4-B25 | 0.701328 | 4.603903 | 복구 없음 |
| V5-A25 | 0.612610 | 3.796983 | 복구 없음 |
| V5-B25 | 0.615972 | 3.814298 | 복구 없음 |

V5의 best는 A25다. 전체 PPO best는 여전히 V3-E75다. 두 정책 모두 zero보다 최대·누적 오차를 동시에 낮추는 성공 기준을 통과하지 못했다.

| seed | zero 최대 / 누적 | V5-A25 최대 / 누적 | V5-B25 최대 / 누적 | zero 대비 판정 |
|---|---:|---:|---:|---|
| 2026 | 0.316630 / 0.194330 | 0.613246 / 3.802844 | 0.616345 / 3.819383 | 두 후보 모두 실패 |
| 2027 | 0.319481 / 0.196755 | 0.614067 / 3.819541 | 0.617203 / 3.837846 | 두 후보 모두 실패 |
| 2028 | 0.319605 / 0.195900 | 0.610516 / 3.768564 | 0.614369 / 3.785665 | 두 후보 모두 실패 |

[Distribution 지표]

| 후보 | 확률 raw latent 평균 | 확률 raw latent 표준편차 | bounded 표본 평균 | \|a\|≥0.99 (pitch/roll) |
|---|---:|---:|---:|---:|
| V5-A25 | -1.502042, +0.338742 | +0.420512, +0.292562 | -0.869038, +0.307039 | 0.0083% / 0.0000% |
| V5-B25 | -1.541832, +0.333211 | +0.412825, +0.296539 | -0.877875, +0.301942 | 0.0083% / 0.0042% |

위 원시 표준편차는 상태 간 μ 변동을 포함한 표본의 표준편차이며 학습된 σ와 다르다. deterministic |tanh(μ)|≥0.9는 A pitch81.83%/roll0.071%, B pitch83.925%/roll0.0875%다. deterministic |action|≥0.99는 두 후보 모두0이다. 경계 포화 감소만으로 큰 지속 명령이나 잘못된 피드백이 해결되지는 않았다.

[안전성]

- 두 후보의 checkpoint25 평가 각각48 episode: 생존100%, self collision0, abnormal ground0, actuator torque saturation0. 최대 leg torque는 A0.134665, B0.134642 N·m이다.
- 네 발 접지 비율: A99.8083%, B99.8083%. 초기 reset 정착 구간의 일시 발 접촉 손실을 포함한다.
- 각 후보 warm-up self-collision termination1건, 학습 self-collision termination2건. 종료 처리는 유지하며 안전 성공으로 지우지 않았다.
- GPU zero·random squashed·네 corner 시험: NaN/invalid·abnormal ground·torque saturation0. corner(+1,−1)에서 seed2026 초기 env11의 self contact1건이 재현됐다. 500Hz step29/30에서 fl_upper–torso 약2.08/1.54 µm 침투였다. 전체 최대 torque0.159477 N·m이다.
- 지정 CPU MuJoCo의 평형 시작 zero·random squashed·네 corner 10초 시험: self·abnormal ground·saturation·NaN·MuJoCo warning0, 네 발 접지100%, 최대 torque0.160338 N·m. GPU의 초기 perturbation에서 나타난 corner 문제와 구분해 기록한다.
- 모델·collision·basis·힘 제한을 바꾸지 않았다. 평가의 안전 통과가 모든 reset 상태와 corner의 무접촉을 보장하지는 않는다.

[회귀 검증]

- nq24, nv23, nu17, mass1.173 kg 유지.
- XML·ctrlrange·neutral_standing·collision·actuator mapping·geometry·kinematics와 standalone 파일 모두 V4 commit과 byte 단위 동일.
- reward·reference·control authority·environment 물리 구현·basis 설계 및 scaling 동일.
- 모든 canonical/native V5 checkpoint에서 32,000 관측의 첫30차원 실제 통계 일치·freeze·previous applied action12D identity·42D 관측·2D 출력 유지.
- V5 checkpoint8개와 기존 V1/V2/V3/V4 checkpoint8개를 loader로 복원했다. V5 optimizer 및 distribution·regularization metadata를 복원했고 canonical25/native24 output이 정확히 같다.
- 12개 trace, 총576개 episode의 초기 관측과 외란 시작 일치, finite·bounded output 확인.
- 전체47개 테스트 통과. standalone 10초 지정 CPU 회귀에서 height0.1927197m, peak torque0.1026784 N·m, saturation/self/warning0. GUI viewer는 직접 시험하지 않았다.
- 새 package 설치·Walking 구현·state-dependent std 구현 없음.

[Git Diff]

V5 관련 15개 변경은 unstaged이며 이번 요청에서 commit·push하지 않는다. `git diff --check` 통과. HEAD와 upstream은 V4 commit e1aeae5에 그대로 있고 main은 f2df8dd다.

수정: `rl/README.md`, `rl/ablation.py`, `rl/action_mapping.py`, `rl/compare_standing.py`, `rl/config.py`, `rl/diagnose_ppo.py`, `rl/evaluate.py`, `rl/normalization.py`, `rl/train.py`.

추가: `rl/STANDING_V5.md`, `rl/distribution_diagnostics.py`, `rl/distributions.py`, `rl/mean_regularization.py`, `tests/test_mean_regularization.py`, `tests/test_squashed_distribution.py`.

[최종 판단]

SUCCESS 정책 없음. V5의 최적은 A25, 전체 PPO 최적은 V3-E75다. 물리적으로 유효한2D basis에서도 잘못된 피드백과 필터 상태에 따른 지속 μ 편향이 남는다. bounded distribution 구현 수정은 필요했지만 이 학습 문제를 해결하지는 못했다. 약한 mean penalty는 평형 probe를 조금 줄였어도 실제 자세 제어를 개선하지 못했다.

[다음 단계]

추가 hyperparameter tuning을 중단했다. 우선순위는 advantage sign와 state–action credit, 그다음 state-dependent exploration, 마지막으로 imitation pretraining+PPO 진단이다. shared actor/value gradient는 없으므로 critic의 advantage 경로·filter memory·추정 편향을 먼저 분리한다. 동일 state에서 scripted/반대/zero의 짧은 미래 return과 actor score gradient의 방향이 맞는지 검사하는 진단을 다음 후보로 추천한다. 이번에는 구현하지 않았다. 성공 정책이 없으므로 multi-training-seed 성공 검증과 Walking으로 진행하지 않는다.

[질문에 대한 답]

1. 기존 PPO는 원시 Gaussian u에 대해 학습했다. 실제 명령은 clip·basis·smoothing 뒤의 다른 값이다. latent 정책으로는 유효하지만 명령 다양성과 entropy가 다르다.
2. true squashed distribution에서는 bounded a·storage·log probability·deterministic evaluation이 일치한다. 모든 update 전 ratio≈1/KL0을 확인했다.
3. roll μ 편향은 크게 줄었지만 pitch μ 편향은 커졌다. 두 축 평균 절댓값은 줄었어도 persistent bias가 해결되지는 않았다.
4. V5-A는 세 seed 모두 zero보다 최대·누적 오차가 크다.
5. V5-A 평형 대칭 probe에서 common offset이 보정 성분보다 두 축 모두 훨씬 컸다.
6. scripted 평형 명령≈0, 실제 평형의 큰 actor 명령, 대칭성에서 큰 common offset이 mean penalty의 물리·학습 근거였다.
7. V5-B는 평형 probe 편향을 조금 줄였지만 전체 trajectory μ 편향과 자세 제어는 개선하지 못했다.
8. 성공 정책이 없으므로 아직 multi-training-seed 성공 검증으로 넘어가지 않는다.
9. advantage/credit를 우선하며 exploration, imitation-pretraining을 뒤에 둔다.

[재현 및 결과 파일]

실행 Python은 기존 GPU RL Python `/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python`이며 CPU MuJoCo는 `/home/fcsl/robot_ws/mujoco/.venv/bin/python`이다. 기존 환경의 버전은 rsl_rl5.0.1, mjlab1.3.0, torch2.9.1, GPU MuJoCo3.10, CPU MuJoCo3.14다.

```bash
cd /home/fcsl/Secret_project
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.train \
  --num-envs 64 --iterations 100 --seed 42424 --warmup-seed 52424 \
  --normalization warmup-frozen --warmup-steps 500 --v2-stage A \
  --episode-seconds 10 --smoothing-tau .15 --orientation-reward-scale .05 \
  --rollout 32 --previous-action-normalization identity \
  --standing-basis /tmp/microdog_v4/subspace/basis.json \
  --policy-distribution squashed --action-mapping identity \
  --ablation-name v5a --ablation-parent /tmp/microdog_v4/v4b/evaluation/summary.json \
  --log-dir /tmp/microdog_v5/v5a/ppo
```

V5-B 명령은 위와 동일하며 `--mean-regularization .0009273962086400983`, `--ablation-name v5b`, parent `/tmp/microdog_v5/v5a/evaluation/summary.json`, log dir `/tmp/microdog_v5/v5b/ppo`로 바꾼다. `--iterations 100`은 상한이며 현재 후보는 gate에서25회 종료됐다.

- [현재 분포 진단](/tmp/microdog_v5/diagnostics/current_distribution.json)
- [평균 응답 그래프](/tmp/microdog_v5/distribution_response.png)
- [축별 gradient 이력](/tmp/microdog_v5/gradient_summary.json)
- [V5-A 평가](/tmp/microdog_v5/v5a/evaluation/summary.json)
- [V5-B 평가](/tmp/microdog_v5/v5b/evaluation/summary.json)
- [회귀 근거](/tmp/microdog_v5/regression.json)
- [전체 테스트 로그](/tmp/microdog_v5/final_tests.log)

개발 중 사전 검사에서 deterministic output을 μ로 재사용한 기존 검사 가정이 검출돼 학습 전에 수정했다. baseline evaluator에도 identity를 전달하면 기존 scripted 명령의 최종 clip이 빠지는 문제가 검출돼 원래 full-joint clip baseline을 유지하도록 수정했다. 해당 검사/평가 실패 산출물은 `/tmp/microdog_v5/v5a_preflight_rejected`와 `v5a_baseline_rejected`에 분리 보관했다. 최종 V5-A/B 결과에는 수정 후 완료된 실행만 사용했다.

checkpoint·trajectory·임시 진단 스크립트는 `/tmp`에 두며 Git에는 포함하지 않는다. 장기 보존이 필요하면 `/tmp` 외부에 실험 artifact를 별도로 보관해야 한다.

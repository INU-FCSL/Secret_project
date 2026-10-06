# Standing V4 정규화와 action subspace 진단 보고서

V4-A는 previous action 입력의 과도한 정규화 증폭을 제거했지만 25회에서 PARTIAL 승격 기준에 못 미쳤다. 물리 분석은 reduced basis 실행 조건을 충족했다. V4-B는 불필요한 명령 성분을 크게 줄였으나 25회에서 자세 오차가 악화됐다. **두 후보 모두 25회에서 중단했고 추가 자동 튜닝을 종료했다. 전체 최적 PPO는 여전히 V3-E75이며 zero-action을 이긴 PPO는 없다.**

## [V3-C/E Commit]

작업 전 branch·HEAD와 승인된 10개 파일을 확인했다. 저장된 source hash와 모두 일치했고 git diff·공백 검사, 전체 31개 테스트 및 CPU 물리 회귀를 통과한 뒤 stage·commit·일반 push했다.

```text
commit = bf53990e760124671ba8a766b18a8b2308691a9c
message = Add standing V3 action and rollout ablations
branch = feature/standing-rl
upstream = origin/feature/standing-rl
```

push 직후 local HEAD와 upstream 일치, working tree clean을 확인했다. 포함 파일은 기존 승인 목록의 README·compare_standing·config·env·evaluate·train·STANDING_V3CDE·ablation·action_mapping·test_action_mapping 10개다. 이후 V4 관련 변경은 stage·commit·push하지 않았다.

## 공통 실행 및 판정

| 항목 | 설정 |
| --- | --- |
| training / warm-up seed | 42424 / 52424 |
| evaluation | 2026 / 2027 / 2028, 각각 16 episode |
| 학습 환경 / rollout | 64개 / 32 step |
| episode / smoothing | 10초 / 0.15초 |
| 외란 | Stage A 0.5 N × 0.5초 |
| reward / mapping | orientation scale 0.05 / clip |
| warm-up | 64 × 500 = 32,000개 실제 관측 |
| PPO | 64×64 ELU, 기존 epoch·mini-batch·gamma·lambda·학습률 규칙·action std·entropy weight 유지 |
| 학습 표본 | V4-A와 V4-B 각각 25 update × 64 × 32 = 51,200개 |

V4-A는 V3-E 설정에서 새 가중치로 시작했다. V3-E checkpoint 가중치를 이어 학습하지 않았다. 같은 seed·25 update의 V3-E25와 비교했다. V4-B도 같은 seed에서 새 정책으로 시작했으며 actor 출력만 12→2차원으로 바꿨다. 출력 차원 변화로 분포의 총 entropy 및 난수 소비는 달라질 수 있지만 PPO hyperparameter를 추가 조정하지 않았다.

SUCCESS는 세 seed 모두에서 최대·누적 오차가 zero보다 작고 안전한 경우다. PARTIAL은 같은 update의 parent 대비 두 오차가 각각 15% 이상 감소하고 안전성과 feedback·경계·지속 편향·복구 중 하나 이상의 개선을 확인한 경우다. V4-A는 소폭 개선됐지만 승격 기준 미달로 FAIL, V4-B는 실제 오차 악화로 FAIL이다. 두 경우 모두 100회로 연장하지 않았다. 자기접촉·비정상 지면 접촉 0 및 포화 1% 미만을 평가 안전 조건으로 사용했고 실제 후보 평가는 포화 0이었다.

## [V4-A Previous Action Input]

관측 30:42는 정책 raw 출력이 아니라 **직전 step에 실제 적용한 12차원 smoothed joint action**이다. 필터 내부 상태를 나타내므로 raw Gaussian이나 basis latent로 바꾸지 않았다. 관측은 계속 42차원이다.

```text
기존: 전체 42차원 → full empirical normalization
V4-A: state 30차원 → 기존 warm-up/frozen normalization
      previous smoothed action 12차원 → 정확한 identity
actor와 critic에 동일하게 적용
```

`SemanticNormalization`은 기존 buffer 이름·42차원 tensor 형식을 보존하고 forward에서 앞 30차원만 정규화한다. 뒤 12차원의 통계 buffer는 변환에 사용하지 않는다. raw 분포는 별도 진단으로 기록한다. 기본 모드는 running이며 기존 V1/V2/V3 checkpoint는 기존 전체 정규화로 복원한다. V4 metadata에 identity 모드와 `[True]*30+[False]*12` mask를 저장한다.

warm-up 32,000개, eps=0.01, 사전 exact KL=0·PPO clipping=0을 확인했다. warm-up actor previous 입력 범위는 -0.472133~0.459453이며 원시 값과 정확히 일치했다. 실제 MLP 입력 hook, 앞 30차원의 기존 normalizer 결과 일치, actor/critic 일치, PPO 중 통계 고정 및 checkpoint 재로딩을 단위 테스트했다.

| 정책 | 최대 오차 ° | 누적 오차 도·초 | 복구 초 | 복구 조건 성공률 | return | 최대 토크 N·m |
| --- | --- | --- | --- | --- | --- | --- |
| V3-E25 parent | 0.739230 | 4.463790 | 복구 실패 | 0.00% | 21.154892 | 0.175822 |
| V4-A25 | 0.667770 | 4.034943 | 복구 실패 | 0.00% | 21.343269 | 0.159074 |

최대 9.67%, 누적 9.61% 개선됐지만 각각 15% 미만이다. **FAIL(승격 기준 미달)**이며 최적 학습 checkpoint는 25회다. 100회 연장 없이 action subspace 분석으로 이동했다. 입력 처리의 구조적 타당성과 학습 성공을 구분했다.

| 정책 | raw 범위 초과 | 관절 applied 경계 | 관절별 평균 절댓값의 평균 |
| --- | --- | --- | --- |
| V3-E25 | 64.03% | 60.64% | 0.831488 |
| V4-A25 | 30.68% | 28.95% | 0.570588 |

| previous action actor 입력 | 범위 |
| --- | --- |
| 관절별 평균 | -0.9720 ~ 0.9694 |
| 관절별 std | 0.0291 ~ 0.1434 |
| 전체 최소~최대 | -0.999999762 ~ 0.999999762 |
| warm-up 경험적 범위 밖 | 63.96% |

raw와 actor 입력의 mean/std/min/max가 정확히 같다. ±10~13으로 증폭되던 문제가 제거됐다. 다만 초기 warm-up의 경험적 범위를 벗어나는 raw 관측은 여전히 존재하며, state 30차원을 포함한 전체 관측 분포 이동까지 해결됐다는 뜻은 아니다.

| 국소 feedback | roll | pitch |
| --- | --- | --- |
| 자세 명령 변화 / ° | -0.019788138 | -0.023597726 |
| 각속도 명령 변화 / (rad/s) | 0.172980467 | -0.141512895 |

knee 차등 명령의 기대 자세 feedback 부호는 roll 양수, pitch 음수다. V4-A25의 roll 자세 feedback은 반대이고 pitch는 기대 방향이다.

## [Action Subspace Analysis]

지정 CPU Python과 MuJoCo 3.14.0에서 XML을 그대로 compile했다. 자연 평형과 ±x/±y 방향 0.5 N·0.25초 작은 외란으로 만든 네 상태를 사용했다. 각 상태에서 동일한 초기 물리 상태로 복원하고 joint command ±0.05의 중앙 차분을 계산했다. 기존 0.15초 필터와 action scale을 유지했다. 응답 시점은 0.2/0.5/1/2초다. roll·pitch는 degree/action, height는 mm/action이다. 서로 다른 단위의 height 행을 orientation SVD에 섞지 않았다.

개별 orientation Jacobian은 2×12라 rank 상한 2가 출력 차원에 의해 정해진다. 그 사실만으로 12차원이 불필요하다고 판단하지 않았다. 다섯 상태·네 응답 시간을 쌓은 **40×12 Jacobian**에서 상대 1% 문턱의 rank는 5이며, 상위 2개 방향이 제곱 특이값 합의 **97.65%**를 설명했다. 95% dominant dimension은 2다. 완전한 rank-2 모델이라는 뜻은 아니다.

| 순서 | 특이값 | 민감도 에너지 비율 |
| --- | --- | --- |
| 1 | 1.849542958 | 56.18% |
| 2 | 1.589238386 | 41.48% |
| 3 | 0.297987198 | 1.46% |
| 4 | 0.202678772 | 0.67% |
| 5 | 0.113008630 | 0.21% |
| 6 | 0.012850950 | 0.00% |
| 7 | 0.005148660 | 0.00% |
| 8 | 0.002628876 | 0.00% |
| 9 | 0.000689167 | 0.00% |
| 10 | 0.000580703 | 0.00% |
| 11 | 0.000149886 | 0.00% |
| 12 | 0.000121003 | 0.00% |

| 관절 | 첫 방향 | 둘째 방향 |
| --- | --- | --- |
| FL_hip_roll | 0.00164 | 0.00000 |
| FL_hip_pitch | 0.05606 | -0.09771 |
| FL_knee | 0.45515 | -0.56206 |
| FR_hip_roll | 0.00164 | -0.00000 |
| FR_hip_pitch | -0.05606 | -0.09771 |
| FR_knee | -0.45515 | -0.56206 |
| RL_hip_roll | -0.02358 | -0.00001 |
| RL_hip_pitch | 0.01143 | -0.18029 |
| RL_knee | 0.53760 | 0.37687 |
| RR_hip_roll | -0.02358 | 0.00001 |
| RR_hip_pitch | -0.01143 | -0.18029 |
| RR_knee | -0.53760 | 0.37687 |

첫 방향은 주로 좌우 knee 차등, 둘째 방향은 주로 앞뒤 knee 차등이다. 앞뒤 다리의 기하학 차이와 hip 기여도 포함된다. SVD 벡터의 전체 부호는 임의이며 물리 feedback 부호는 실제 응답 측정으로 판단한다.

| 관절 | roll °/action | pitch °/action | height mm/action |
| --- | --- | --- | --- |
| FL_hip_roll | -0.000766 | -0.000000 | 0.000000 |
| FL_hip_pitch | -0.024396 | 0.034320 | -0.121609 |
| FL_knee | -0.206884 | 0.218497 | -0.346490 |
| FR_hip_roll | -0.000766 | 0.000000 | -0.000000 |
| FR_hip_pitch | 0.024396 | 0.034320 | -0.121609 |
| FR_knee | 0.206884 | 0.218497 | -0.346490 |
| RL_hip_roll | 0.011039 | 0.000000 | 0.000000 |
| RL_hip_pitch | -0.004292 | 0.073827 | -0.110322 |
| RL_knee | -0.245427 | -0.146275 | -0.246165 |
| RR_hip_roll | 0.011039 | -0.000000 | -0.000000 |
| RR_hip_pitch | 0.004292 | 0.073827 | -0.110322 |
| RR_knee | 0.245427 | -0.146275 | -0.246165 |

유효 성분은 상위 2개의 우측 특이벡터가 만드는 직교 subspace로의 투영이다. 에너지는 물리적 일이나 열량이 아니라 **정규화 명령 제곱합의 평균**이다. 아래는 외력 시작 이후 applied trajectory의 비율이며 전체 구간·requested·bounded 통계도 JSON에 보존했다. weak/null은 측정한 국소 roll/pitch 반응의 약한 방향을 뜻한다. height·발 위치·관절 자세·접촉·토크 효과까지 없다는 뜻은 아니다. common 비율과 weak 비율은 겹칠 수 있으므로 더하지 않는다.

| 제어기 | 총 명령 제곱합 평균 | 유효 성분 | weak 성분 | common 성분 |
| --- | --- | --- | --- | --- |
| scripted | 0.012910 | 92.69% | 7.31% | 0.00% |
| v3e75 | 8.518294 | 1.21% | 98.79% | 51.11% |
| v4a25 | 5.700142 | 20.93% | 79.07% | 14.18% |
| v4b25 | 1.563099 | 93.45% | 6.55% | 0.00% |

실행 전 정한 V4-B 조건은 dominant dimension≤4에서 95% 자세 민감도 설명, scripted 유효 성분≥70%, V3-E/V4-A 각각 weak 성분≥30%, 국소 시험 안전성이다. 네 조건을 모두 충족했다. 120개 중앙 차분 물리 시험에서 자기접촉·비정상 지면 접촉·포화·발 접지 손실·warning은 없었다.

## [V4-B Reduced Standing Action]

실행했다. Standing 전용 진단으로 pitch knee 차등과 roll knee 차등의 두 모드를 선택했다. 추가 height 또는 hip 모드를 넣을 필요는 이번 물리 자료에서 확인되지 않았다. hidden network 64×64 ELU와 PPO 설정은 유지했고 actor 출력만 2차원이다. 관측은 42차원이며 previous action은 실제 12차원 필터 상태를 identity로 전달한다.

측정된 1초 full-mode 제어 권한은 pitch=0.729544°, roll=0.904620°다. 두 모드의 최댓값 합을 1로 제한하면서 제어 권한을 맞춰 pitch scale=0.553567497, roll scale=0.446432503를 정했다. 균등화한 국소 권한은 각각 0.403852°/basis command다.

```text
a_pitch, a_roll = clip(raw policy action, -1, 1)
FL knee = +pitch_scale*a_pitch + roll_scale*a_roll
FR knee = +pitch_scale*a_pitch - roll_scale*a_roll
RL knee = -pitch_scale*a_pitch + roll_scale*a_roll
RR knee = -pitch_scale*a_pitch - roll_scale*a_roll
hip roll / hip pitch = 0 offset
위 12차원 명령 → 기존 smoothing → 기존 safe action scale
```

행별 basis 절댓값 합≤1이므로 2차원 cube 전체가 기존 12차원 safe range 안에 있다. 각 축 ±1 및 네 결합 모서리를 다섯 상태에서 검사한 40개 시험이 안전했다. 관절 physical scale인 hip roll ±0.25°, hip pitch/knee ±2°는 변경하지 않았다. 아래는 평형에서 1초에 측정한 실제 응답이다.

| basis 명령 | roll 변화 ° | pitch 변화 ° | height 변화 mm | 최대 토크 N·m |
| --- | --- | --- | --- | --- |
| [1, 0] | 0.000000 | 0.413468 | -0.133517 | 0.107216 |
| [-1, 0] | 0.000000 | -0.394959 | 0.089958 | 0.114541 |
| [0, 1] | -0.403957 | -0.002877 | -0.017519 | 0.110641 |
| [0, -1] | 0.403957 | -0.002877 | -0.017519 | 0.110641 |
| [1, 1] | -0.413334 | 0.410087 | -0.151412 | 0.109126 |
| [1, -1] | 0.413334 | 0.410087 | -0.151412 | 0.109126 |
| [-1, 1] | -0.394052 | -0.397271 | 0.072901 | 0.125354 |
| [-1, -1] | 0.394052 | -0.397271 | 0.072901 | 0.125354 |

2차원 raw Gaussian·storage·log-prob을 유지했다. storage는 `[32,64,2]`이며 pre-update KL=0, 유한한 returns/advantages와 native PPO update를 검증했다. latent clipping 뒤 고정된 B 행렬로 12차원 joint command를 만든다. zero/scripted 공통 비교는 기존 12차원 제어기를 사용해 기준 자체를 바꾸지 않았다. 두 축의 raw latent 통계와 joint로 투영한 requested 통계는 구분한다.

| 정책 | 최대 오차 ° | 누적 오차 도·초 | 복구 초 | 복구 조건 성공률 | return | 최대 토크 N·m |
| --- | --- | --- | --- | --- | --- | --- |
| V4-A25 parent | 0.667770 | 4.034943 | 복구 실패 | 0.00% | 21.343269 | 0.159074 |
| V4-B25 | 0.701339 | 4.603921 | 복구 실패 | 0.00% | 21.330860 | 0.141872 |

V4-A25보다 최대 오차 5.03%, 누적 오차 14.10% 악화됐다. **FAIL**, 100회 연장 없음, 최적 학습 checkpoint는 25회다. 25회 이후 추가 자동 학습·튜닝은 수행하지 않았다.

| basis latent | raw 평균 | raw std | raw 최소 | raw 최대 | 양 경계 | 음 경계 |
| --- | --- | --- | --- | --- | --- | --- |
| pitch_knee | -0.690277 | 0.262909 | -1.821267 | 3.859022 | 0.33% | 0.05% |
| roll_knee | 1.511205 | 0.459051 | -4.501431 | 2.710633 | 90.64% | 0.53% |

관절 applied 경계는 0.00%지만 2차원 latent raw 범위 초과는 45.71%이고 roll latent의 양 경계 비율은 90.64%다. 관절 경계가 사라졌다는 이유만으로 policy 내부 포화가 사라졌다고 판정하지 않는다.

| 국소 feedback | roll | pitch |
| --- | --- | --- |
| 자세 명령 변화 / ° | -0.000780713 | 0.064543052 |
| 각속도 명령 변화 / (rad/s) | 0.005693794 | 0.048746373 |

roll은 대부분 clipping으로 국소 응답이 평평하며 평균 자세 feedback은 반대 부호다. pitch 자세 feedback은 명확히 기대 방향과 반대다. 공통·약한 성분을 없애는 것만으로 올바른 자세 feedback이 학습되지 않았다.

## [Zero vs Scripted vs Best PPO]

전체 최적 PPO는 여전히 V3-E75다. 현재 코드로 다시 48 episode 평가한 값은 아래와 같다. 기존 보고값과 약 1e-5 수준의 차이는 GPU 물리 연산의 수치 변동 범위이며 결론은 같다.

| 제어기 | 최대 오차 ° | 누적 오차 도·초 | 복구 초 | 복구 조건 성공률 | return | 최대 토크 N·m |
| --- | --- | --- | --- | --- | --- | --- |
| zero | 0.318582 | 0.195655 | 0.126749 | 100.00% | 21.907418 | 0.141832 |
| scripted | 0.145980 | 0.070686 | 0.010082 | 100.00% | 21.914278 | 0.144727 |
| V3-E75 최적 PPO | 0.327904 | 0.415811 | 0.188416 | 100.00% | 21.705914 | 0.148775 |
| V4-A25 | 0.667770 | 4.034943 | 복구 실패 | 0.00% | 21.343269 | 0.159074 |
| V4-B25 | 0.701339 | 4.603921 | 복구 실패 | 0.00% | 21.330860 | 0.141872 |

| seed | zero 최대 ° | V3-E75 최대 ° | zero 누적 도·초 | V3-E75 누적 도·초 | V4-A25 최대 / 누적 | V4-B25 최대 / 누적 |
| --- | --- | --- | --- | --- | --- | --- |
| 2026 | 0.316607 | 0.325056 | 0.194308 | 0.416176 | 0.665040 / 4.019741 | 0.702895 / 4.618447 |
| 2027 | 0.319541 | 0.328400 | 0.196764 | 0.417284 | 0.666313 / 4.058309 | 0.703022 / 4.626535 |
| 2028 | 0.319598 | 0.330258 | 0.195891 | 0.413973 | 0.671957 / 4.026779 | 0.698100 / 4.566780 |

V4-A/V4-B 각각의 zero 기준은 해당 평가 디렉터리에도 별도로 보존했다. 모든 비교의 최초 초기 관측·외란 시작 시점을 확인했다. 어느 새 후보도 세 seed 중 한 seed에서조차 최대·누적 오차를 동시에 zero보다 줄이지 못했다. 학습 전 checkpoint 0은 학습 성공 후보에서 제외한다.

## [Action Bias]

| 제어기 | common | 앞뒤 차등 | 좌우 차등 | 대각 모드 | 관절 applied 경계 |
| --- | --- | --- | --- | --- | --- |
| scripted | 0.00% | 56.41% | 43.59% | 0.00% | 0.00% |
| V3-E75 | 51.11% | 11.44% | 36.80% | 0.64% | 53.49% |
| V4-A25 | 14.18% | 28.29% | 32.81% | 24.72% | 28.95% |
| V4-B25 | 0.00% | 49.09% | 50.91% | 0.00% | 0.00% |

공통·앞뒤·좌우·대각 모드는 4개 다리에 대한 직교 변환이며 hip roll/hip pitch/knee별로 따로 계산했다. 모든 모드의 제곱 에너지를 합하면 원래 12차원 에너지가 보존된다. V3-E는 common 성분도 크지만 V4-A의 대부분은 common이 아니라 weak 방향이다. V4-B의 common·대각 성분 0은 basis 구조에 의해 보장되는 결과다. 학습 성공의 증거로 사용하지 않았다. 약한 성분은 V4-B에서 크게 줄어도 자세 성능은 악화됐다.

### V4A checkpoint 0: 관절별 명령

raw/requested는 12차원 Gaussian 정책의 결정적 출력이다.

| 관절 | raw/requested 평균 | raw std | applied 평균 | applied std | 양 경계 | 음 경계 |
| --- | --- | --- | --- | --- | --- | --- |
| FL_hip_roll | 0.1525 | 0.0387 | 0.1503 | 0.0184 | 0.00% | 0.00% |
| FL_hip_pitch | 0.1048 | 0.0469 | 0.1034 | 0.0189 | 0.00% | 0.00% |
| FL_knee | 0.1236 | 0.0868 | 0.1208 | 0.0339 | 0.00% | 0.00% |
| FR_hip_roll | 0.1072 | 0.0719 | 0.1050 | 0.0216 | 0.00% | 0.00% |
| FR_hip_pitch | 0.0337 | 0.0363 | 0.0333 | 0.0150 | 0.00% | 0.00% |
| FR_knee | 0.0134 | 0.0326 | 0.0133 | 0.0158 | 0.00% | 0.00% |
| RL_hip_roll | -0.1719 | 0.0458 | -0.1694 | 0.0306 | 0.00% | 0.00% |
| RL_hip_pitch | -0.1140 | 0.0291 | -0.1125 | 0.0139 | 0.00% | 0.00% |
| RL_knee | -0.0889 | 0.0576 | -0.0878 | 0.0291 | 0.00% | 0.00% |
| RR_hip_roll | 0.0911 | 0.0473 | 0.0898 | 0.0258 | 0.00% | 0.00% |
| RR_hip_pitch | 0.0598 | 0.0394 | 0.0590 | 0.0204 | 0.00% | 0.00% |
| RR_knee | -0.0279 | 0.0667 | -0.0276 | 0.0265 | 0.00% | 0.00% |

각 joint type의 모드 계수 평균은 common / 앞뒤 / 좌우 / 대각 순서다.

| 관절 종류 | common 평균 | 앞뒤 차등 평균 | 좌우 차등 평균 | 대각 평균 |
| --- | --- | --- | --- | --- |
| hip_roll | 0.08786 | 0.16743 | -0.10694 | 0.15224 |
| hip_pitch | 0.04157 | 0.09507 | -0.05068 | 0.12080 |
| knee | 0.00940 | 0.12476 | 0.02365 | 0.08388 |

previous action의 actor 입력 통계:

| 관절 | 입력 평균 | 입력 std | 입력 최소 | 입력 최대 |
| --- | --- | --- | --- | --- |
| FL_hip_roll | 0.1500 | 0.0196 | -0.0700 | 0.2263 |
| FL_hip_pitch | 0.1032 | 0.0195 | -0.0887 | 0.3168 |
| FL_knee | 0.1206 | 0.0343 | -0.0633 | 0.4416 |
| FR_hip_roll | 0.1048 | 0.0221 | -0.0167 | 0.3336 |
| FR_hip_pitch | 0.0332 | 0.0151 | -0.1907 | 0.0946 |
| FR_knee | 0.0133 | 0.0158 | -0.0548 | 0.1731 |
| RL_hip_roll | -0.1690 | 0.0315 | -0.2880 | 0.0646 |
| RL_hip_pitch | -0.1123 | 0.0147 | -0.2154 | 0.0178 |
| RL_knee | -0.0876 | 0.0294 | -0.2694 | 0.2031 |
| RR_hip_roll | 0.0896 | 0.0262 | -0.1925 | 0.2051 |
| RR_hip_pitch | 0.0589 | 0.0206 | -0.0850 | 0.1978 |
| RR_knee | -0.0275 | 0.0265 | -0.2663 | 0.1910 |

### V4A checkpoint 25: 관절별 명령

raw/requested는 12차원 Gaussian 정책의 결정적 출력이다.

| 관절 | raw/requested 평균 | raw std | applied 평균 | applied std | 양 경계 | 음 경계 |
| --- | --- | --- | --- | --- | --- | --- |
| FL_hip_roll | 0.8802 | 0.1083 | 0.8672 | 0.1351 | 0.00% | 0.00% |
| FL_hip_pitch | 0.2010 | 0.0937 | 0.1972 | 0.0374 | 0.00% | 0.00% |
| FL_knee | 0.5451 | 0.0769 | 0.5360 | 0.0605 | 0.00% | 0.00% |
| FR_hip_roll | 1.0316 | 0.1242 | 0.9614 | 0.1368 | 83.04% | 0.00% |
| FR_hip_pitch | 0.1328 | 0.0539 | 0.1310 | 0.0285 | 0.00% | 0.00% |
| FR_knee | 0.1241 | 0.0613 | 0.1225 | 0.0370 | 0.00% | 0.00% |
| RL_hip_roll | -0.5376 | 0.0845 | -0.5294 | 0.0868 | 0.00% | 0.00% |
| RL_hip_pitch | -0.9913 | 0.0962 | -0.9647 | 0.1293 | 0.00% | 83.66% |
| RL_knee | 0.1838 | 0.0621 | 0.1811 | 0.0411 | 0.00% | 0.00% |
| RR_hip_roll | 0.4170 | 0.1013 | 0.4112 | 0.0801 | 0.00% | 0.00% |
| RR_hip_pitch | 1.1672 | 0.1251 | 0.9714 | 0.1270 | 89.97% | 0.00% |
| RR_knee | -1.2504 | 0.1372 | -0.9740 | 0.1181 | 0.00% | 90.70% |

각 joint type의 모드 계수 평균은 common / 앞뒤 / 좌우 / 대각 순서다.

| 관절 종류 | common 평균 | 앞뒤 차등 평균 | 좌우 차등 평균 | 대각 평균 |
| --- | --- | --- | --- | --- |
| hip_roll | 0.85521 | 0.97339 | -0.51738 | 0.42323 |
| hip_pitch | 0.16746 | 0.16075 | -0.93487 | 1.00116 |
| knee | -0.06717 | 0.72569 | 0.78428 | -0.37081 |

previous action의 actor 입력 통계:

| 관절 | 입력 평균 | 입력 std | 입력 최소 | 입력 최대 |
| --- | --- | --- | --- | --- |
| FL_hip_roll | 0.8654 | 0.1406 | -0.0108 | 0.9764 |
| FL_hip_pitch | 0.1968 | 0.0384 | -0.1297 | 0.3641 |
| FL_knee | 0.5349 | 0.0650 | 0.0000 | 0.7493 |
| FR_hip_roll | 0.9594 | 0.1434 | 0.0000 | 1.0000 |
| FR_hip_pitch | 0.1307 | 0.0291 | -0.1273 | 0.4224 |
| FR_knee | 0.1223 | 0.0374 | -0.0967 | 0.2742 |
| RL_hip_roll | -0.5283 | 0.0899 | -0.6498 | 0.1392 |
| RL_hip_pitch | -0.9627 | 0.1363 | -1.0000 | 0.0000 |
| RL_knee | 0.1807 | 0.0419 | -0.2300 | 0.2803 |
| RR_hip_roll | 0.4104 | 0.0822 | -0.3121 | 0.6721 |
| RR_hip_pitch | 0.9694 | 0.1342 | 0.0000 | 1.0000 |
| RR_knee | -0.9720 | 0.1259 | -1.0000 | 0.1249 |

### V4B checkpoint 0: 관절별 명령

V4-B의 requested는 2차원 raw latent를 B로 투영한 값이며 독립 12차원 Gaussian 출력이 아니다.

| 관절 | raw/requested 평균 | raw std | applied 평균 | applied std | 양 경계 | 음 경계 |
| --- | --- | --- | --- | --- | --- | --- |
| FL_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| FL_hip_pitch | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| FL_knee | 0.0325 | 0.0356 | 0.0321 | 0.0148 | 0.00% | 0.00% |
| FR_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| FR_hip_pitch | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| FR_knee | -0.0938 | 0.0257 | -0.0925 | 0.0108 | 0.00% | 0.00% |
| RL_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| RL_hip_pitch | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| RL_knee | 0.0938 | 0.0257 | 0.0925 | 0.0108 | 0.00% | 0.00% |
| RR_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| RR_hip_pitch | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| RR_knee | -0.0325 | 0.0356 | -0.0321 | 0.0148 | 0.00% | 0.00% |

각 joint type의 모드 계수 평균은 common / 앞뒤 / 좌우 / 대각 순서다.

| 관절 종류 | common 평균 | 앞뒤 차등 평균 | 좌우 차등 평균 | 대각 평균 |
| --- | --- | --- | --- | --- |
| hip_roll | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| hip_pitch | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| knee | 0.00000 | -0.06041 | 0.12465 | 0.00000 |

previous action의 actor 입력 통계:

| 관절 | 입력 평균 | 입력 std | 입력 최소 | 입력 최대 |
| --- | --- | --- | --- | --- |
| FL_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FL_hip_pitch | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FL_knee | 0.0321 | 0.0149 | -0.0154 | 0.2037 |
| FR_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FR_hip_pitch | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FR_knee | -0.0923 | 0.0116 | -0.1935 | 0.0596 |
| RL_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RL_hip_pitch | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RL_knee | 0.0923 | 0.0116 | -0.0596 | 0.1935 |
| RR_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RR_hip_pitch | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RR_knee | -0.0321 | 0.0149 | -0.2037 | 0.0154 |

### V4B checkpoint 25: 관절별 명령

V4-B의 requested는 2차원 raw latent를 B로 투영한 값이며 독립 12차원 Gaussian 출력이 아니다.

| 관절 | raw/requested 평균 | raw std | applied 평균 | applied std | 양 경계 | 음 경계 |
| --- | --- | --- | --- | --- | --- | --- |
| FL_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| FL_hip_pitch | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| FL_knee | 0.2925 | 0.1190 | 0.0357 | 0.0768 | 0.00% | 0.00% |
| FR_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| FR_hip_pitch | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| FR_knee | -1.0568 | 0.3350 | -0.7900 | 0.2276 | 0.00% | 0.00% |
| RL_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| RL_hip_pitch | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| RL_knee | 1.0568 | 0.3350 | 0.7900 | 0.2276 | 0.00% | 0.00% |
| RR_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| RR_hip_pitch | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00% | 0.00% |
| RR_knee | -0.2925 | 0.1190 | -0.0357 | 0.0768 | 0.00% | 0.00% |

각 joint type의 모드 계수 평균은 common / 앞뒤 / 좌우 / 대각 순서다.

| 관절 종류 | common 평균 | 앞뒤 차등 평균 | 좌우 차등 평균 | 대각 평균 |
| --- | --- | --- | --- | --- |
| hip_roll | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| hip_pitch | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| knee | 0.00000 | -0.75426 | 0.82570 | 0.00000 |

previous action의 actor 입력 통계:

| 관절 | 입력 평균 | 입력 std | 입력 최소 | 입력 최대 |
| --- | --- | --- | --- | --- |
| FL_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FL_hip_pitch | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FL_knee | 0.0358 | 0.0768 | -0.3040 | 0.3076 |
| FR_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FR_hip_pitch | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| FR_knee | -0.7882 | 0.2302 | -0.9171 | 0.2905 |
| RL_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RL_hip_pitch | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RL_knee | 0.7882 | 0.2302 | -0.2905 | 0.9171 |
| RR_hip_roll | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RR_hip_pitch | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| RR_knee | -0.0358 | 0.0768 | -0.3076 | 0.3040 |

## [Safety]

| 평가 정책 | 생존율 | 자기접촉 | 비정상 지면 접촉 | 토크 포화 | 네 발 접지 | 최대 토크 N·m |
| --- | --- | --- | --- | --- | --- | --- |
| V3-E75 | 100.00% | 0 | 0 | 0.00% | 99.81% | 0.148775 |
| V4-A25 | 100.00% | 0 | 0 | 0.00% | 99.81% | 0.159074 |
| V4-B25 | 100.00% | 0 | 0 | 0.00% | 99.81% | 0.141872 |

평가와 학습 탐색 중의 접촉을 구분한다. V4-A/V4-B 각각 warm-up 자기접촉 종료 1회, 실제 25회 학습 자기접촉 종료 2회였다. 모두 기존 실패 종료로 처리했다. 학습 비정상 지면 접촉·포화는 0이며 학습 최대 토크는 다음과 같다.

| 실험 | 학습 최대 토크 N·m | 정규화 count | 모든 갱신 전 KL |
| --- | --- | --- | --- |
| v4a | 0.161456 | 32000 | 0 |
| v4b | 0.145211 | 32000 | 0 |

## [Regression]

| 검사 | 결과 |
| --- | --- |
| 전체 테스트 | 37개 통과 |
| checkpoint 감사 | 기존·신규 14개 로딩 및 모드 복원 통과 |
| 모델 | nq=24 / nv=23 / nu=17 / mass=1.173 kg |
| XML·actuator | ctrlrange·neutral standing·collision·mapping·geometry·kinematics 유지 |
| standalone | run/view/animate 파일 불변; 지정 CPU Python의 headless 10초 회귀 통과 |
| CPU 회귀 | 높이 0.192719714 m / 최대 토크 0.102678368 N·m / 자기접촉·포화·warning 0 |
| 정규화 | 첫 30차원 통계가 수집 원시 자료와 일치하고 모든 checkpoint·PPO update에서 고정 |
| previous action | actor·critic·실제 MLP에서 12차원 identity 일치, metadata와 mask 재로딩 통과 |
| V3-E 가중치 | V4-A 초기 actor/critic 가중치가 V3-E 초기 가중치와 완전히 일치 |
| V4-B | 관측 42 / policy action 2 / 실제 관절 명령 12 / native raw Gaussian 저장 경로 유지 |
| 기존 정책 | V1/V2/V3-A/V3-B/V3-C/V3-E의 full-normalization 호환 및 V3-E75 재평가 통과 |
| 설치 / GUI | 새 패키지 설치 없음 / GUI 수동 실행 미검증 |

GPU 학습은 기존 microduck_rl 가상환경의 MuJoCo 3.10.0·Warp를 사용했다. Jacobian·basis 안전·standalone 회귀는 지정 `/home/fcsl/robot_ws/mujoco/.venv/bin/python`과 MuJoCo 3.14.0을 사용했다. CPU/GPU 물리 수치의 완전한 일치는 가정하지 않는다. 각 V4 후보는 자기 환경에서 동일 seed의 zero와 비교했고, 사전 물리 측정의 방향과 범위는 실제 GPU 환경의 단위 테스트와 rollout에서도 검증했다.

## [Git Diff]

```text
 M rl/README.md
 M rl/ablation.py
 M rl/compare_standing.py
 M rl/config.py
 M rl/env.py
 M rl/evaluate.py
 M rl/normalization.py
 M rl/train.py
?? rl/STANDING_V4.md
?? rl/action_diagnostics.py
?? rl/action_subspace.py
?? tests/test_semantic_normalization.py
?? tests/test_standing_basis.py
```

V4 변경 13개 파일은 모두 unstaged다. 최종 git status·git diff·git diff --check와 신규 파일 공백 검사를 확인했다. HEAD와 origin/feature/standing-rl은 V3-C/E commit `bf53990e760124671ba8a766b18a8b2308691a9c`에서 일치한다. main 및 origin/main은 `f2df8dda498aa098d56582bd43c0661dc78e1be2`를 유지한다. V4 commit·push, main merge와 destructive reset은 수행하지 않았다.

## [최종 판단]

previous action 정규화 증폭과 12차원 weak 방향 명령이라는 두 문제를 직접 확인하고 각각 분리해 줄였다. 그럼에도 PPO는 올바른 feedback과 zero 대비 자세 개선을 얻지 못했다. 따라서 이번 자료는 이 두 요인만으로 Standing 실패를 설명하기에 부족하다. 현재 최적 PPO는 V3-E75이며 새로운 성공 baseline으로 채택할 후보는 없다. reduced basis를 Walking action space로 채택하지 않았다.

## [다음 단계]

추가 자동 hyperparameter 튜닝을 종료한다. 다음은 별도 승인된 연구 단계에서 bounded/squashed stochastic policy, actor mean 제약, 상태에 따른 std, action magnitude 제약, symmetry-aware parameterization을 한 요인씩 검토하는 것이다. 특히 V4-B의 높은 roll latent 평균·포화와 잘못된 pitch feedback을 실제 PPO update 신호와 연결해 분석해야 한다. 이번 요청에서는 이 후보를 자동 구현하지 않았다.

사용자 질문에 대한 답:

1. **previous action OOD 입력은 해결됐는가?** 해당 12차원의 과도한 정규화 증폭은 해결됐다. 입력은 원시 applied action과 정확히 같고 [-1,1] 안에 있다. 전체 관측 분포 이동이 모두 해결됐다는 뜻은 아니다.
2. **이것만으로 자세 제어가 개선됐는가?** 같은 seed·25회에서 최대 9.67%, 누적 9.61% 소폭 개선됐지만 승격 기준과 zero 대비 성공에는 못 미쳤다.
3. **V3-E12D 명령의 유효 비율은?** 외란 이후 applied 명령의 제곱 에너지 기준 1.21%다. 이는 여러 상태·시간의 국소 orientation Jacobian 상위 2방향에 대한 투영 비율이다.
4. **지속 편향 대부분이 common/null 방향인가?** V3-E의 weak 성분은 98.79%, common은 51.11%였다. V4-A는 weak 79.07%, common 14.18%이므로 약한 방향과 common을 동일시하면 안 된다.
5. **reduced basis의 물리 근거는 충분했는가?** 기립 진단 실험의 근거는 충분했다. 상위 2방향이 97.65% 자세 민감도를 설명했고 scripted 유효 에너지가 92.69%였으며 결합 안전 시험도 통과했다.
6. **reduced basis PPO가 세 seed에서 zero를 이겼는가?** 아니다. V4-B25는 parent보다 자세 오차가 악화됐고 세 seed 모두 zero보다 두 오차가 컸다.
7. **성공 정책의 multi-training-seed 검증 단계인가?** 아직 아니다. 성공 조건을 충족한 PPO가 없다.
8. **다음은 distribution/actor parameterization을 바꿔야 하는가?** 별도 분석 단계로 검토하는 것을 추천한다. 불필요한 방향 제거만으로 해결되지 않았으므로 latent 편향·포화와 feedback 학습 원인을 먼저 분석하고 변경을 분리해 검증해야 한다.

## 재현 명령과 자료

```bash
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.train --num-envs 64 --iterations 100 --v2-stage A --smoothing-tau .15 --episode-seconds 10 --seed 42424 --normalization warmup-frozen --warmup-steps 500 --warmup-seed 52424 --orientation-reward-scale .05 --action-mapping clip --rollout 32 --previous-action-normalization identity --ablation-parent /tmp/microdog_v3cde/v3e/evaluation/summary.json --ablation-name v4a --log-dir /tmp/microdog_v4/v4a/ppo
/home/fcsl/robot_ws/mujoco/.venv/bin/python -m rl.action_subspace --output /tmp/microdog_v4/subspace
/home/fcsl/robot_ws/mujoco/.venv/bin/python -m rl.action_subspace --output /tmp/microdog_v4/subspace --design-basis
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.train --num-envs 64 --iterations 100 --v2-stage A --smoothing-tau .15 --episode-seconds 10 --seed 42424 --normalization warmup-frozen --warmup-steps 500 --warmup-seed 52424 --orientation-reward-scale .05 --action-mapping clip --rollout 32 --previous-action-normalization identity --standing-basis /tmp/microdog_v4/subspace/basis.json --ablation-parent /tmp/microdog_v4/v4a/evaluation/summary.json --ablation-name v4b --log-dir /tmp/microdog_v4/v4b/ppo
```

학습 상한 100회는 gate 통과 시에만 실행된다. 이번 실행은 둘 다 25회에서 중단됐다. 재실행 시에는 이전 자료와 섞이지 않는 새 출력 경로를 사용하고 action_subspace의 trajectory 입력 경로도 해당 새 자료로 맞춘다.

자료는 `/tmp/microdog_v4/{v4a,v4b}/ppo/`, 각 `evaluation/`, `reference/evaluation/`, `subspace/summary.json`, `subspace/basis.json`, `regression.json`에 있다. `manifest.json`, `source_snapshot/`, `tracked_diff.patch`에 최종 소스와 자료 hash를 보존했다. 평가 합계는 V4-A 240 + V4-B 240 + 기존 기준 재평가 192 = 672 episode이며 CPU 물리 시험은 별도다. `/tmp` 자료는 영구 보관을 보장하지 않는다.

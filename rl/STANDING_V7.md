# Standing V7 검증 보고서

이 문서의 실험 수치와 당시 판정은 V7 완료 시점의 기록이다. 이후 Walking V1 전환 단계에서 V7을 `Add standing V7 reward and critic diagnostics`로 commit·일반 push하고 중요 자료를 `artifacts/standing/local/`에 보존한다. 기본 Standing 물리·control interface와 scripted/imitation recovery 검증은 완료했으며, from-scratch 정책 발견과 유지 문제는 연구 항목으로 남긴다. 이 미해결 문제는 Walking 개발의 필수 선행조건이 아니다. 아래의 Walking 보류 판단은 당시 연구 범위의 판단으로 읽는다.

보상 총점과 자세 품질의 불일치, same-policy critic의 큰 value 편향을 확인했다. pose weight를 0.2에서 0.1로 낮추면 저장 trajectory의 보상 순위는 개선되지만, 좋은 imitation 정책의 PPO 성능 악화는 줄지 않았다. 이어서 보상 0.1을 고정하고 critic 초기화만 교정했으나 성능 악화가 더 커졌다. 이번에는 from-scratch 학습과 state-dependent exploration의 실행 조건을 충족하지 못했다.

확보된 성공은 imitation-assisted SUCCESS다. From-scratch PPO SUCCESS는 아직 없다. V6만 commit·일반 push했으며, V7 코드는 `feature/standing-rl`에 unstaged로 남긴다. Walking은 구현하지 않았다.

## [V6 커밋]

- commit: `e59a9ff54ab3fa930a10a8e47590abc747dbf008`
- 메시지: `Add standing V6 credit and imitation diagnostics`
- branch·upstream: `feature/standing-rl` → `origin/feature/standing-rl`
- 일반 push 성공. 직후 working tree는 clean이었고 local HEAD와 upstream이 일치했다.
- V6 예상 10파일의 manifest 일치, 52개 테스트, 18개 checkpoint 복원, 12개 평가 trace와 CPU 물리 회귀를 확인한 뒤 commit했다.
- `main`·`origin/main`: `f2df8dda498aa098d56582bd43c0661dc78e1be2`, 변경하지 않았다.
- 포함 파일: `rl/README.md`, `rl/ablation.py`, `rl/config.py`, `rl/train.py`, `rl/STANDING_V6.md`, `rl/credit_audit.py`, `rl/imitation_diagnostic.py`, `rl/timeout_bootstrap.py`, `tests/test_gae_bootstrap.py`, `tests/test_gpu_state_snapshot.py`.

## [보상 정렬 검증]

실제 `REWARD_WEIGHTS['pose']`는 **0.2**다. 9개 정책의 같은 초기 관측·외란 시점 48 episode를 대조했다. 모든 보상항은 이미 weight와 control-step dt가 적용된 저장값이며 float64로 합산했다. 외란 전, 외란 중 0.5초, 회복 첫 1초, 이후 안정 구간, 전체 episode를 분리했다. Spearman은 동률 평균 순위를 직접 구현했으며 새 package를 설치하지 않았다.

물리 순위: scripted → imitation 0 → imitation 2 → imitation 1 → imitation 3 → imitation 4 → imitation 5 → zero → V3-E75.

기존 return 순위: imitation 4 → imitation 5 → scripted → imitation 1 → imitation 0 → imitation 3 → imitation 2 → zero → V3-E75.

| pose weight | return 대 −누적 오차 Spearman | return 대 −최대 오차 Spearman | 0→5 return 변화 | 선정 조건 |
| --- | --- | --- | --- | --- |
| 0.2 | 0.333333 | 0.466667 | +0.000129245 | good_above_worse, deterioration_0_to_5 |
| 0.1 | 0.966667 | 1.000000 | -0.000161697 | 전부 충족 |
| 0.05 | 0.983333 | 0.983333 | -0.000307169 | 전부 충족 |
| 0.0 | 0.983333 | 0.983333 | -0.000452640 | pose_retained |

**선정 후보는 0.1**이다. scripted/imitation0가 더 나쁜 PPO 정책보다 높은 점수, 0→5 악화가 return 감소로 반영, 좋은 정책이 zero보다 높은 점수, pose 항 유지라는 네 조건을 모두 만족하는 가장 큰 후보값이다. 0.05는 조건을 만족하지만 추가 감소 근거가 부족하고, 0은 posture 항을 제거한다.

V6 imitation0→5의 upright 보상은 `−0.000456563`, pose 보상은 `+0.000581885`, 전체 보상은 `+0.000129245`로 변했다. 누적 자세 오차는 `0.071731→0.132660°·s`, **84.94% 증가**했다. pose 증가가 upright 감소를 초과하여 물리적 악화가 총점 증가로 평가되었다. 0→5 변화량을 보상항 합과 float32 저장 reward 합으로 계산하면 약 `3.1e−7` 차이가 있으므로 미세 변화는 float64 보상항 합으로 비교한다.

| 갱신 | Δupright | Δpose | Δheight | Δ관절 속도 | Δaction rate | Δeffort | Δ총점 | Δ최대 오차(°) | Δ누적 오차(°·s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0→1 | -0.00004058 | +0.00004039 | +0.00000000 | +0.00000002 | +0.00000004 | +0.00000020 | +0.00000006 | +0.001587 | +0.006063 |
| 1→2 | -0.00000321 | -0.00008894 | +0.00000000 | +0.00000010 | +0.00000000 | -0.00000043 | -0.00009248 | +0.000232 | -0.003563 |
| 2→3 | -0.00014649 | +0.00020489 | +0.00000000 | +0.00000020 | -0.00000000 | +0.00000109 | +0.00005969 | +0.000757 | +0.029301 |
| 3→4 | -0.00018268 | +0.00043133 | +0.00000000 | +0.00000038 | -0.00000005 | +0.00000227 | +0.00025125 | -0.000636 | +0.022042 |
| 4→5 | -0.00008360 | -0.00000579 | +0.00000000 | +0.00000011 | +0.00000002 | -0.00000002 | -0.00008928 | +0.002117 | +0.007086 |

### 구간별 기존 weighted 보상항

아래는 episode 평균 합계다. 모든 정책은 동일 48 episode이며, 전체 구간의 높이 항은 약5다. 원본 JSON에는 48개 episode 각각의 항도 보관했다.

| 정책 | 구간 | upright | height | pose | 관절 속도 | action rate | effort | 총점 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero | 외란 전 | 3.74288349 | 1.25083331 | 0.48205931 | -0.00017967 | 0.00000000 | -0.00021529 | 5.47538115 |
| zero | 외란 중 | 0.74422021 | 0.24999999 | 0.09557328 | -0.00001144 | 0.00000000 | -0.00005257 | 1.08972947 |
| zero | 회복 첫 1초 | 1.49920947 | 0.49999999 | 0.19242679 | -0.00001135 | 0.00000000 | -0.00008926 | 2.19153564 |
| zero | 안정 구간 | 8.99743068 | 2.99916660 | 1.15472540 | -0.00000001 | 0.00000000 | -0.00052936 | 13.15079331 |
| zero | 전체 episode | 14.98374385 | 4.99999989 | 1.92478478 | -0.00020247 | 0.00000000 | -0.00088649 | 21.90743956 |
| scripted | 외란 전 | 3.74445490 | 1.25083331 | 0.48190614 | -0.00017654 | -0.00000365 | -0.00021612 | 5.47679804 |
| scripted | 외란 중 | 0.74888881 | 0.24999999 | 0.09565068 | -0.00001047 | -0.00000054 | -0.00005239 | 1.09447609 |
| scripted | 회복 첫 1초 | 1.49990761 | 0.49999999 | 0.19242304 | -0.00000962 | -0.00000049 | -0.00008925 | 2.19223129 |
| scripted | 안정 구간 | 8.99749860 | 2.99916660 | 1.15466985 | -0.00000000 | -0.00000000 | -0.00052993 | 13.15080511 |
| scripted | 전체 episode | 14.99074993 | 4.99999989 | 1.92464971 | -0.00019663 | -0.00000468 | -0.00088768 | 21.91431053 |
| imitation 0 | 외란 전 | 3.74442373 | 1.25083331 | 0.48191161 | -0.00017625 | -0.00000298 | -0.00021606 | 5.47677335 |
| imitation 0 | 외란 중 | 0.74887317 | 0.24999999 | 0.09565173 | -0.00001043 | -0.00000052 | -0.00005238 | 1.09446157 |
| imitation 0 | 회복 첫 1초 | 1.49990672 | 0.49999999 | 0.19242497 | -0.00000965 | -0.00000049 | -0.00008924 | 2.19223230 |
| imitation 0 | 안정 구간 | 8.99749852 | 2.99916660 | 1.15467838 | -0.00000000 | -0.00000000 | -0.00052987 | 13.15081363 |
| imitation 0 | 전체 episode | 14.99070215 | 4.99999989 | 1.92466669 | -0.00019634 | -0.00000399 | -0.00088755 | 21.91428085 |
| imitation 1 | 외란 전 | 3.74441346 | 1.25083331 | 0.48191934 | -0.00017623 | -0.00000296 | -0.00021603 | 5.47677088 |
| imitation 1 | 외란 중 | 0.74884945 | 0.24999999 | 0.09565399 | -0.00001042 | -0.00000050 | -0.00005237 | 1.09444013 |
| imitation 1 | 회복 첫 1초 | 1.49990235 | 0.49999999 | 0.19242896 | -0.00000966 | -0.00000048 | -0.00008922 | 2.19223193 |
| imitation 1 | 안정 구간 | 8.99749632 | 2.99916660 | 1.15470479 | -0.00000000 | -0.00000000 | -0.00052973 | 13.15083797 |
| imitation 1 | 전체 episode | 14.99066157 | 4.99999989 | 1.92470708 | -0.00019632 | -0.00000395 | -0.00088735 | 21.91428092 |
| imitation 2 | 외란 전 | 3.74440833 | 1.25083331 | 0.48190529 | -0.00017615 | -0.00000297 | -0.00021606 | 5.47675174 |
| imitation 2 | 외란 중 | 0.74884995 | 0.24999999 | 0.09564953 | -0.00001041 | -0.00000050 | -0.00005239 | 1.09443617 |
| imitation 2 | 회복 첫 1초 | 1.49990180 | 0.49999999 | 0.19241961 | -0.00000965 | -0.00000047 | -0.00008926 | 2.19222202 |
| imitation 2 | 안정 구간 | 8.99749828 | 2.99916660 | 1.15464370 | -0.00000000 | -0.00000000 | -0.00053006 | 13.15077851 |
| imitation 2 | 전체 episode | 14.99065836 | 4.99999989 | 1.92461814 | -0.00019622 | -0.00000395 | -0.00088778 | 21.91418844 |
| imitation 3 | 외란 전 | 3.74432020 | 1.25083331 | 0.48196240 | -0.00017602 | -0.00000299 | -0.00021584 | 5.47672105 |
| imitation 3 | 외란 중 | 0.74883481 | 0.24999999 | 0.09566036 | -0.00001039 | -0.00000049 | -0.00005233 | 1.09443195 |
| imitation 3 | 회복 첫 1초 | 1.49988973 | 0.49999999 | 0.19244019 | -0.00000962 | -0.00000046 | -0.00008914 | 2.19223068 |
| imitation 3 | 안정 구간 | 8.99746713 | 2.99916660 | 1.15476009 | -0.00000000 | -0.00000000 | -0.00052938 | 13.15086444 |
| imitation 3 | 전체 episode | 14.99051187 | 4.99999989 | 1.92482303 | -0.00019603 | -0.00000395 | -0.00088669 | 21.91424813 |
| imitation 4 | 외란 전 | 3.74421870 | 1.25083331 | 0.48206257 | -0.00017572 | -0.00000302 | -0.00021547 | 5.47672037 |
| imitation 4 | 외란 중 | 0.74882102 | 0.24999999 | 0.09568350 | -0.00001036 | -0.00000051 | -0.00005218 | 1.09444145 |
| imitation 4 | 회복 첫 1초 | 1.49987337 | 0.49999999 | 0.19248483 | -0.00000956 | -0.00000047 | -0.00008889 | 2.19225926 |
| imitation 4 | 안정 구간 | 8.99741611 | 2.99916660 | 1.15502346 | -0.00000000 | -0.00000000 | -0.00052787 | 13.15107830 |
| imitation 4 | 전체 episode | 14.99032919 | 4.99999989 | 1.92525436 | -0.00019565 | -0.00000400 | -0.00088442 | 21.91449938 |
| imitation 5 | 외란 전 | 3.74421001 | 1.25083331 | 0.48205845 | -0.00017565 | -0.00000302 | -0.00021548 | 5.47670761 |
| imitation 5 | 외란 중 | 0.74877750 | 0.24999999 | 0.09568480 | -0.00001033 | -0.00000050 | -0.00005219 | 1.09439929 |
| imitation 5 | 회복 첫 1초 | 1.49985808 | 0.49999999 | 0.19248410 | -0.00000955 | -0.00000046 | -0.00008889 | 2.19224327 |
| imitation 5 | 안정 구간 | 8.99739999 | 2.99916660 | 1.15502122 | -0.00000000 | -0.00000000 | -0.00052788 | 13.15105993 |
| imitation 5 | 전체 episode | 14.99024559 | 4.99999989 | 1.92524857 | -0.00019554 | -0.00000398 | -0.00088444 | 21.91441010 |
| V3-E75 | 외란 전 | 3.73500390 | 1.25083331 | 0.43967148 | -0.00020666 | -0.00000853 | -0.00024661 | 5.42504689 |
| V3-E75 | 외란 중 | 0.74377836 | 0.24999999 | 0.08571790 | -0.00001163 | -0.00000006 | -0.00005517 | 1.07942939 |
| V3-E75 | 회복 첫 1초 | 1.49910374 | 0.49999999 | 0.17260479 | -0.00001141 | -0.00000007 | -0.00009199 | 2.17160506 |
| V3-E75 | 안정 구간 | 8.99561299 | 2.99916660 | 1.03558263 | -0.00000002 | -0.00000000 | -0.00052096 | 13.02984124 |
| V3-E75 | 전체 episode | 14.97349898 | 4.99999989 | 1.73357680 | -0.00022971 | -0.00000866 | -0.00091472 | 21.70592257 |

### 국소 보상 gradient

6개 대표 상태에서 첫 2D action을 평균 주변 ±0.1로 바꾸고 이후 imitation0 deterministic 정책으로 1/5/10/25/50 step을 진행했다. 아래는 50-step weighted return의 `[pitch, roll]` 중앙 차분이다. 확률적 same-policy MC와 구분한다.

| 상태 | upright gradient | pose gradient(0.2) | 두 gradient cosine |
| --- | --- | --- | --- |
| 자연 평형 | [6.37596e-07, 5.97126e-08] | [-1.76312e-05, 1.8244e-07] | -0.994625 |
| −pitch | [0.000230255, -9.66996e-07] | [-1.85373e-05, 7.69069e-07] | -0.999306 |
| +pitch | [-0.000265594, 1.0706e-05] | [-3.43624e-05, 2.32384e-06] | 0.999629 |
| +roll | [-1.69505e-06, 0.000236651] | [-1.92432e-05, 7.4415e-06] | 0.367350 |
| −roll | [9.41416e-06, -0.000228991] | [-1.83143e-05, -7.01821e-06] | 0.319176 |
| 회복 | [3.73604e-06, 5.51136e-07] | [-2.07962e-05, -1.4896e-06] | -0.997192 |

자연 평형·−pitch·회복의 cosine은 약−0.995/−0.999/−0.997로 강하게 충돌했다. pose weight 0.1은 같은 trajectory의 pose gradient를 절반으로 줄이지만 방향은 바꾸지 않는다. 특히 평형에서 pose gradient 크기는 upright보다 크므로 **9개 trajectory의 순위 개선을 모든 상태의 국소 목표 정렬로 확대 해석하지 않는다**.

## [Same-policy critic 검증]

- paired full-physics state **50개**, 다섯 구간 각각10개. imitation0의 stochastic rollout에서 수집한 같은 상태를 imitation0/1/5에 복원했다. 각 정책별 새로운 상태 분포를 따로 평가한 실험은 아니다.
- 후보9개: 평균, scripted, zero, 평균의 pitch±0.1·roll±0.1, 실제 PPO 표본 action, 첫 action도 확률 표본인 policy expectation.
- 상태·후보마다 독립 continuation **32 replicate**. 후보끼리는 common random numbers를 사용하고 이후 action은 해당 checkpoint의 stochastic squashed 정책이다.
- horizon32/64/원래10초 episode 잔여 구간. endpoint 관측의 critic bootstrap을 적용하고 failure에는 bootstrap하지 않는다.
- critic 오차가 target에도 유입되는 한계를 확인하기 위해, 진단 환경의 시간제한만40초로 늘리고 원래 경계 이후 최소512-step MC tail을 직접 더한 별도 장기 target도 계산했다. 원래 정책·물리·외란·시계를 유지하며 경계에서 reset하지 않는다. 추가 tail 끝은0으로 절단하므로 무한 horizon 정답은 아니다.
- 모든 branch의 failure는0이다. `states.pt`에는 물리·필터·외력·시계·난수 snapshot을 보존했다.

| 정책 | horizon | critic RMSE | 편향 V−MC | EV | Spearman |
| --- | --- | --- | --- | --- | --- |
| imitation 0 | 32 | 1.223673 | -1.221571 | -14.474208 | 0.308715 |
| imitation 0 | 64 | 2.106346 | -2.105188 | -18.845706 | 0.197023 |
| imitation 0 | remainder | 4.223773 | -4.218602 | -0.096982 | 0.054694 |
| imitation 0 | extended | 4.438269 | -4.437741 | -4686.532948 | 0.045282 |
| imitation 1 | 32 | 1.219300 | -1.217880 | -15.327264 | 0.306603 |
| imitation 1 | 64 | 2.100176 | -2.099371 | -19.541689 | 0.206242 |
| imitation 1 | remainder | 4.212279 | -4.207306 | -0.057599 | 0.070540 |
| imitation 1 | extended | 4.426258 | -4.425890 | -3251.786715 | 0.052197 |
| imitation 5 | 32 | 1.138017 | -1.137759 | -9.346219 | 0.438655 |
| imitation 5 | 64 | 1.962924 | -1.962731 | -27.444536 | 0.125762 |
| imitation 5 | remainder | 3.938473 | -3.934031 | -0.010149 | 0.024250 |
| imitation 5 | extended | 4.138519 | -4.138422 | -818.237494 | 0.006194 |

### 구간별 critic 정확도: episode 잔여 구간

| 정책 | 구간 | RMSE | 편향 | EV | Spearman |
| --- | --- | --- | --- | --- | --- |
| imitation 0 | 외란 전 | 4.387851 | -4.387668 | -22.210200 | -0.054545 |
| imitation 0 | 외란 중 | 4.311857 | -4.311059 | -25.803414 | 0.309091 |
| imitation 0 | 초기 회복 | 4.290644 | -4.288988 | -30.316021 | 0.066667 |
| imitation 0 | 후기 회복 | 4.216082 | -4.215713 | -0.419793 | -0.248485 |
| imitation 0 | 안정 구간 | 3.894995 | -3.889585 | -0.012063 | -0.030303 |
| imitation 1 | 외란 전 | 4.374414 | -4.374259 | -18.656144 | -0.103030 |
| imitation 1 | 외란 중 | 4.300846 | -4.300336 | -16.215002 | 0.296970 |
| imitation 1 | 초기 회복 | 4.279316 | -4.278154 | -21.029992 | 0.066667 |
| imitation 1 | 후기 회복 | 4.204617 | -4.204274 | -0.322798 | -0.187879 |
| imitation 1 | 안정 구간 | 3.884889 | -3.879508 | -0.009128 | -0.066667 |
| imitation 5 | 외란 전 | 4.075931 | -4.075748 | -23.720798 | 0.030303 |
| imitation 5 | 외란 중 | 4.039631 | -4.039583 | -0.758032 | 0.442424 |
| imitation 5 | 초기 회복 | 4.004533 | -4.004356 | -2.581427 | 0.115152 |
| imitation 5 | 후기 회복 | 3.926654 | -3.926410 | -0.001657 | 0.054545 |
| imitation 5 | 안정 구간 | 3.629205 | -3.624057 | -0.032243 | 0.018182 |

장기 return을 약4만큼 과소평가한다. extended target에서도 큰 편향이 남아 endpoint bootstrap만의 현상이 아님을 확인했다. extended EV의 큰 음수는 상태별 target 분산이 매우 작은 상황에서 계산되므로 RMSE·편향·순위 상관과 함께 해석해야 한다. critic은 충분히 정확하지 않다.

## [Advantage 정확도]

각 checkpoint의 복사본에 paired 상태50개와 반복 상태14개를 넣어 실제 `64 env×32 step` rollout 및 native PPO update를 수행했다. 표의 GAE와 Δμ는 이 진단 rollout의 실제 값이다. **과거 V6 학습 당시 update를 재구성한 값은 아니다.** 원본 checkpoint는 변경하지 않았다.

`A_MC=Q(실제 표본 action)−V_MC`이며 V_MC는 첫 action까지 stochastic인 policy expectation branch 평균이다. normalization은 전체64×32 rollout에 적용했다. 첫 step50개의 raw와 normalized 부호 일치율은 이번 표본에서 같고, Pearson·Spearman도 양의 affine 변환으로 거의 같다. paired MC 표준오차의2배보다 큰 advantage만 별도로 구분했다.

| 정책 | GAE 종류 | 부호 일치 | Pearson | Spearman | 2SE 초과 표본 부호 일치 | 해당 표본 수 |
| --- | --- | --- | --- | --- | --- | --- |
| imitation 0 | raw | 46.0% | -0.344415 | -0.238703 | 57.6% | 33 |
| imitation 0 | normalized | 46.0% | -0.344415 | -0.238703 | 57.6% | 33 |
| imitation 1 | raw | 50.0% | -0.349590 | -0.244658 | 58.1% | 31 |
| imitation 1 | normalized | 50.0% | -0.349590 | -0.244658 | 58.1% | 31 |
| imitation 5 | raw | 50.0% | -0.085419 | -0.120672 | 60.7% | 28 |
| imitation 5 | normalized | 50.0% | -0.085419 | -0.120672 | 60.7% | 28 |

| 정책 | MC Q 대 실제 Δμ cosine | 축별 부호 일치 | cosine>0 상태 비율 |
| --- | --- | --- | --- |
| imitation 0 | 0.427395 | 64.0% | 78.0% |
| imitation 1 | 0.530672 | 71.0% | 80.0% |
| imitation 5 | 0.385807 | 56.0% | 74.0% |

MC Q action gradient에는 `da/dμ=1−tanh(μ)²`를 적용하여 latent Δμ와 비교했다. 평균 cosine은 양수여서 update 전체가 항상 반대 방향이라고 결론 낼 수 없다. 그러나 advantage의 부호 일치46~50%, 음의 순위 상관, 상태별 불일치는 정확한 credit이 확보되지 않았음을 보여준다. 작은 advantage에는 MC 표본 잡음도 남는다.

## [선택 분기]

보상 충돌과 critic 부정확성이 동시에 확인되어 **A를 먼저 실행**했다. 재점수화에서 선정한0.1을 고정해 보상 순위가 대체로 정렬된 뒤에도 drift와 critic 문제가 남아, **B를 순차 실행**했다. A 대 V6 비교는 pose weight만, B 대 A 비교는 critic 초기화만 바꿨다. B 결과로 reward·critic을 동시에 교체한 효과를 단일 원인으로 해석하지 않는다. 국소 reward 충돌이 완전히 제거된 것도 아니다.

## [V7-A 보상 정렬]

실행했다. pose weight **0.2→0.1** 외의 reward·orientation scale0.05·물리·actor·std0.3·critic·normalizer·PPO optimizer 초기화는 V6 imitation0와 같다.64 env, rollout32, seed42424, 수정된 terminal bootstrap으로5회 fine-tuning했다. 각 checkpoint는 별도 프로세스에서 seed2026/2027/2028의48 episode를 평가했다. 정규화32000개 관측과 previous action identity를 고정했다.

### A의 checkpoint별 결과

| 갱신 | 최대 오차(°) | 누적 오차(°·s) | 회복(s) | return | 공통 관측 μ drift RMS | 평가 μ 평균 [pitch,roll] | 학습 σ [pitch,roll] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.147682 | 0.071729 | 0.010082 | 20.952028 | 0.000000 | [0.010078, 0.007022] | [0.3, 0.3] |
| 1 | 0.149226 | 0.077775 | 0.010082 | 20.952000 | 0.018161 | [0.008232, 0.010647] | [0.299385, 0.299462] |
| 2 | 0.149332 | 0.074132 | 0.009666 | 20.951974 | 0.017627 | [0.010483, 0.006799] | [0.299083, 0.299014] |
| 3 | 0.149851 | 0.103144 | 0.012582 | 20.951914 | 0.042858 | [0.004576, -0.00379] | [0.298089, 0.298954] |
| 4 | 0.149095 | 0.125282 | 0.018832 | 20.951874 | 0.062745 | [-0.008157, 0.000455] | [0.296248, 0.299053] |
| 5 | 0.151507 | 0.132832 | 0.022166 | 20.951799 | 0.075333 | [-0.008495, -0.004162] | [0.29519, 0.298098] |

### A의 전체 episode weighted 보상항

| 갱신 | upright | height | pose | 관절 속도 | action rate | effort | 항 합계 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 14.99070215 | 4.99999989 | 0.96233330 | -0.00019634 | -0.00000399 | -0.00088755 | 20.95194746 |
| 1 | 14.99066210 | 4.99999989 | 0.96235359 | -0.00019632 | -0.00000395 | -0.00088735 | 20.95192796 |
| 2 | 14.99066063 | 4.99999989 | 0.96230916 | -0.00019620 | -0.00000395 | -0.00088778 | 20.95188175 |
| 3 | 14.99051817 | 4.99999989 | 0.96241047 | -0.00019598 | -0.00000396 | -0.00088669 | 20.95184190 |
| 4 | 14.99033475 | 4.99999989 | 0.96262754 | -0.00019559 | -0.00000401 | -0.00088441 | 20.95187817 |
| 5 | 14.99024468 | 4.99999989 | 0.96262420 | -0.00019550 | -0.00000399 | -0.00088443 | 20.95178485 |

A5의 누적 오차0.132832는 V6 PPO5의0.132660보다 **0.13% 악화**했고 공통 관측 μ drift도0.074411→0.075333으로 증가했다. 세 seed zero 대비 SUCCESS와 정상 feedback은 유지했지만, **좋은 정책 유지 개선은 실패**했다. 선정 조건은 누적 오차15% 이상 감소와 μ drift 감소였으며 둘 다 충족하지 못했다. from-scratch25/50/75/100은 실행하지 않았다.

A3→4에서도 누적 오차는0.103144→0.125282로 커졌는데 weighted 총점은 약0.00003627 증가했다. 따라서0.1은 선정에 사용한 V6 trajectory 순위를 개선했을 뿐, 새 학습 trajectory의 모든 국소 악화를 점수 감소로 반영하지는 못한다.

## [V7-B critic]

실행했다. pose0.1과 actor를 고정한 paired50개 상태에서 정책당32 replicate, uniform1024-step bootstrap 독립 MC target을 만들었다. source environment ID로 train30/validation20을 분리하여 같은 환경의 시간 표본이 양쪽에 섞이지 않도록 했다. target은 약4.19이며 추가 할인 tail의 최대 절단 오차는 약0.00015다.

64×64와128×128 critic만 같은 자료·split에서 Adam0.001,3000회 fit하고 validation 최적 지점을 저장했다. 아래 비교는 작은50-state 자료에 대한 제한된 진단이다.

| critic 진단 | train RMSE | validation RMSE | validation 편향 | validation EV | validation Spearman |
| --- | --- | --- | --- | --- | --- |
| 현재64×64 | 0.002555 | 0.133844 | -0.037858 | -73091.945312 | 0.235427 |
| 후보128×128 | 0.001147 | 0.140206 | -0.025531 | -84290.351562 | 0.118842 |
| 현재64×64, output baseline 분리 | 0.000567 | 0.000477 | -0.000041 | 0.000000 | 계산 불가 |

큰 network가 validation을 개선하지 못했다. output head를0 weight와 train MC 평균 bias로 시작한64×64의 최적 checkpoint는 **학습 전 상수 head**였다. RMSE0.000477로 큰 offset은 제거하지만 EV≈0, 순위는 계산 불가다. 따라서 **state-dependent value를 정확히 fit했다거나 capacity 문제가 없다고 확정하지 않는다**.

선택한 단일 요인은 **critic warm-start**다.64×64 구조·actor·global std0.3·정규화·빈 PPO optimizer는 A 초기값과 같고 critic weights만 상수 baseline 진단 결과로 교체했다. B 대 A는 critic 초기화만의 비교다.

### B의 checkpoint별 결과

| 갱신 | 최대 오차(°) | 누적 오차(°·s) | 회복(s) | return | 공통 관측 μ drift RMS | 평가 μ 평균 [pitch,roll] | 학습 σ [pitch,roll] |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 0.147680 | 0.071730 | 0.010082 | 20.952029 | 0.000000 | [0.010076, 0.007022] | [0.3, 0.3] |
| 1 | 0.148646 | 0.072548 | 0.010082 | 20.952006 | 0.018125 | [0.010327, 0.009192] | [0.300052, 0.299367] |
| 2 | 0.147467 | 0.089938 | 0.010082 | 20.951882 | 0.030203 | [0.017393, 0.012639] | [0.300033, 0.299007] |
| 3 | 0.144721 | 0.131421 | 0.011332 | 20.951557 | 0.064808 | [0.029062, 0.019126] | [0.30005, 0.298471] |
| 4 | 0.143257 | 0.151823 | 0.015916 | 20.951461 | 0.090798 | [0.027992, 0.031982] | [0.299052, 0.297785] |
| 5 | 0.142266 | 0.186869 | 0.018416 | 20.951328 | 0.121958 | [0.028881, 0.043939] | [0.298892, 0.297292] |

### B의 전체 episode weighted 보상항

| 갱신 | upright | height | pose | 관절 속도 | action rate | effort | 항 합계 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 14.99070218 | 4.99999989 | 0.96233327 | -0.00019634 | -0.00000399 | -0.00088755 | 20.95194745 |
| 1 | 14.99068145 | 4.99999989 | 0.96233131 | -0.00019640 | -0.00000398 | -0.00088758 | 20.95192468 |
| 2 | 14.99074015 | 4.99999989 | 0.96219811 | -0.00019647 | -0.00000401 | -0.00088893 | 20.95184873 |
| 3 | 14.99072191 | 4.99999989 | 0.96197137 | -0.00019659 | -0.00000409 | -0.00089119 | 20.95160129 |
| 4 | 14.99064038 | 4.99999989 | 0.96198928 | -0.00019681 | -0.00000416 | -0.00089101 | 20.95153756 |
| 5 | 14.99037425 | 4.99999989 | 0.96195514 | -0.00019656 | -0.00000421 | -0.00089124 | 20.95123727 |

B5의 누적 오차0.186869는 A5보다 **40.68% 악화**했다. μ drift도0.075333→0.121958로 증가했다. 세 seed zero 대비 SUCCESS는 유지했지만 초기 imitation0에 가까운 성능을 유지하지 못했다. from-scratch25/50/75/100은 실행하지 않았다.

B5 자체의 stochastic policy로 같은50개 상태·32 replicate·1024-step MC target을 다시 만들었다. 최종 critic RMSE는 **0.004703**, 편향 **−0.003738**, EV **−28.253986**, Spearman **−0.234057**이다. 큰 value offset은 제거했지만 상태별 미세 가치 차이는 정확하게 표현하지 못했다. B5의 advantage 정확도를 MC로 다시 전부 측정한 실험은 수행하지 않았으므로, baseline 보정이 GAE 정렬까지 해결했다고 주장하지 않는다.

| B5 구간 | RMSE | 편향 | EV | Spearman |
| --- | --- | --- | --- | --- |
| 후기 회복 | 0.004225 | -0.004134 | -2259.000000 | -0.127273 |
| 외란 중 | 0.006523 | -0.003707 | -63.532425 | -0.793939 |
| 안정 구간 | 0.003731 | -0.003589 | -6511.085449 | 0.000000 |
| 외란 전 | 0.003423 | -0.002956 | -13.263347 | -0.369697 |
| 초기 회복 | 0.004956 | -0.004303 | -2718.318848 | -0.684848 |

## [V7-C 상태별 탐색]

실행하지 않았다. critic/advantage 정렬과 좋은 정책의 PPO 유지 조건이 충족되지 않아 exploration 원인으로 넘어갈 근거가 부족하다. 정책은 기존 global learned std2개를 유지한다. 상태별 logσ, clamp 변경,25/50/75/100회 학습은 구현·실행하지 않았다.

## [최적 정책]

48 episode, 동일 세 평가 seed 기준. 최대·누적 오차는 평형 기준 자세 오차다. 성공률은 survival만이 아니라 세 seed의 zero보다 최대·누적 오차가 모두 작은지로 판정한다.

| 정책 | 최대 오차(°) | 누적 오차(°·s) | 회복(s) | 구분 |
| --- | --- | --- | --- | --- |
| zero | 0.318583 | 0.195665 | 0.126749 | 비교 기준 |
| scripted | 0.145980 | 0.070683 | 0.010082 | 고정 제어기 |
| imitation0 | 0.147681 | 0.071731 | 0.010082 | imitation-assisted SUCCESS |
| V6 imitation PPO1 | 0.149268 | 0.077794 | 0.010082 | imitation-assisted SUCCESS |
| V6 imitation PPO5 | 0.151738 | 0.132660 | 0.020916 | imitation-assisted SUCCESS, drift |
| V3-E75 | 0.327888 | 0.415812 | 0.188416 | 기존 from-scratch 최적, zero 기준 미달 |
| V6-Fix25 | 0.618535 | 3.836558 | 계산 불가 | from-scratch 실패 |
| V7-A2 | 0.149332 | 0.074132 | 0.009666 | A의 갱신 정책 중 최적 |
| V7-B1 | 0.148646 | 0.072548 | 0.010082 | B의 갱신 정책 중 최적 |

A·B 각각의0~5 checkpoint를 모두 저장했다. 안전한 후보 중 누적→최대→회복 오차 순으로 선정한 전체 최적은 **둘 다 update0**이다. A0·B0의 actor tensor는 기존 imitation0와 정확히 같으며1e−6 수준 평가 차이는 GPU 수치 차이다. 새로운 PPO 갱신 정책만 보면 A2·B1이 각각 최적이다. 세 seed 모두 zero보다 좋고 feedback 부호가 정상임을 확인했다.

각 후보의 `best_checkpoint.pt`, `best_ppo_checkpoint.pt`를 원본과 byte 일치하는 복사본으로 저장하고, 선정 iteration은 `decision.json`에 기록했다. 기존 from-scratch 정책 중 비교 최적은 V3-E75이며 여전히 zero를 이기지 못한다. 이번 V7 from-scratch checkpoint는 없다.

## [최종 판정]

- **Imitation-assisted SUCCESS: 유지**. A/B의 모든0~5 정책은 세 seed zero 대비 성공하지만 PPO drift가 지속된다.
- **From-scratch PPO SUCCESS: 미확보**. V7은 연장 조건 미달로 새로운 from-scratch 학습을 실행하지 않았다.
- 보상 정렬 문제: 기존0.2의 trajectory 순위 오류는 확인했다.0.1은 측정 순위를 개선하나 국소 reward 충돌은 남는다.
- critic/advantage 문제: 기존 critic의 큰 과소평가와 GAE의 낮은 부호·순위 정확도를 확인했다. 상수 baseline 교정만으로 drift를 줄이지 못했다.
- exploration/discovery 문제: 충분한 critic 정렬과 정책 유지가 먼저 검증되지 않아 이번 실험으로 원인·개선 효과를 확정할 수 없다.
- 현재 가장 강한 근거는 **국소 reward 목표와 작은 action 이득의 credit을 PPO update가 충분히 보존하지 못한다는 것**이다. reward 또는 critic 중 하나만의 단독 원인으로 확정하지 않는다. entropy·분포 이동·optimizer 영향은 이번에 분리 실험하지 않았다.

## [안전성]

평가12개 checkpoint×48 episode에서 NaN/무한값, saturation, unexpected contact, self-contact는 모두0이다. 네 발 접지율은 약99.808%, survival100%다. 평가 최대 leg torque는 약0.1451 N·m이며 limit±0.52는 유지했다.

**확률적 학습 rollout에서는 A와 B 각각 self-collision 종료2회가 기록되었다.** 모두 첫 update에서 발생했고 이후4회는0이다. 평가 안전성과 구분하며, 접촉 종료의 구체적 초기 상태 원인은 별도 추적하지 않았다. invalid·abnormal ground·torque saturation은 학습에서도0이고 최대 torque는 A0.149965/B0.150413 N·m였다. self-collision termination을 비활성화하지 않았다.

CPU10초 static hold는 height0.192719714 m, peak torque0.102678368 N·m, saturation/self-contact/MuJoCo warning0이다. GUI viewer를 직접 열어 검증하지는 않았다.

## [회귀]

- `nq=24`, `nv=23`, `nu=17`, mass1.173 kg.
- XML·ctrlrange·neutral keyframe·collision·geometry·joint/actuator mapping·standalone 파일·AGENTS를 byte 비교로 보존했다.
- action mapping·2D basis·정규화 구현·squashed distribution·terminal bootstrap·evaluation 구현을 보존했다. env 변경은 pose weight 인자를 전달하는 한 줄뿐이다.
- 기존 Gaussian/squashed checkpoint4개와 V7 checkpoint12개, 총16개를 실제 복원했다. V6 commit 전에는 기존18개도 검증했다.
- optimizer 복원·초기 optimizer 빈 상태·normalizer count32000·freeze·previous action identity·초기 actor tensor·A critic 원본 일치·B critic 선정 weight 일치를 확인했다.
- V7 평가12개는 이전 zero와 초기 관측·외란 시점이 정확히 같다. 세 seed zero 개선·feedback 부호·안전성과 best 복사본을 확인했다.
- **전체55개 테스트 통과**. 기존52개에 pose 단일항·재점수화2개와 수치 MC credit 집계1개를 추가했다.
- 새 package 설치, reference 수정, main merge/reset, V7 commit/push는 하지 않았다.

## [Git 차이]

수정6개: `rl/README.md`, `rl/ablation.py`, `rl/config.py`, `rl/env.py`, `rl/rewards.py`, `rl/train.py`.

신규7개: `rl/STANDING_V7.md`, `rl/reward_alignment.py`, `rl/policy_credit.py`, `rl/standing_v7.py`, `rl/critic_fit.py`, `tests/test_reward_alignment.py`, `tests/test_policy_credit_metrics.py`.

기본 pose weight0.2는 호환성을 위해 유지하고 V7 profile에0.1을 명시했다. 새 CLI는 pose 후보와 critic warm-start를 지원하며, reward 변경은 pose항만이다. 기존 behavior는 기본값에서 유지한다. Python 진단·checkpoint·trace·그래프는 `/tmp/microdog_v7`에 저장했다.

`git status`, tracked 전체 `git diff`, untracked diff, `git diff --check`를 최종 확인한다. index는 비어 있고 V7의13개 파일은 모두 unstaged다. HEAD·upstream은 V6 commit `e59a9ff`에 그대로 있다. 최종 확인 자료는 `/tmp/microdog_v7/git_final.json` 및 diff 파일에 보관한다.

## [다음 단계]

추가 iteration이나 state-dependent std로 넘어가기 전에, 좋은 정책의 같은 상태에서 **작은 상태별 MC advantage와 GAE의 credit을 더 정확히 대조**할 필요가 있다. 다음 단일 실험은 critic의 state-dependent residual 및 phase별 학습 표본을 검증하고, baseline 보정 이후에도 남은 update 방향 오류를 분리하는 것이다. 보상0.1의 평형·회복 국소 목표 충돌도 따로 평가해야 한다. 한 실험에서 reward·critic·exploration을 동시에 바꾸지 않는다.

standing discovery와 좋은 정책 유지가 해결되지 않아 Walking으로 넘어갈 준비가 되지 않았다.

## [마지막 질문 답변]

1. **보상 총점 순위 일치:** 기존0.2는 불일치.0.1은 측정9개 trajectory의 Spearman을0.3333→0.9667로 개선했지만 전 상태 정렬 보장은 없다.
2. **pose의 악화 상쇄:** 있었다. pose+0.000581885가 upright−0.000456563을 초과했다.
3. **weight 변경 근거:** 물리 악화 대비 return 증가, 재점수화 선정 조건, 평형·회복 국소 gradient 충돌이다.
4. **same-policy critic 정확도:** 기존은 부정확했다. B는 큰 offset만 크게 줄였고 상태별 순위는 여전히 부정확하다.
5. **empirical advantage와 GAE:** 충분히 일치하지 않는다. 잔여 horizon 부호46~50%, Spearman−0.239/−0.245/−0.121이다.
6. **PPO gradient 장기 정렬:** 평균 cosine0.427/0.531/0.386으로 부분 정렬이나 일관된 상태별 정렬은 아니다.
7. **drift 원인:** reward·critic 문제가 둘 다 있으나 pose 감소와 baseline 교정 각각이 drift를 해결하지 못했다. 단독 원인 판정은 보류한다.
8. **좋은 정책의 장기 유지 개선:** 없다. A5는 V6와 비슷하게 악화했고 B5는 A5보다40.68% 더 나쁘다.
9. **from-scratch 최초 zero 승리:** 없다. V7 from-scratch는 실행 조건 미달로 수행하지 않았다.
10. **state-dependent exploration 효과:** 미실행으로 판단 불가. 도움이 됐다고 주장할 수 없다.
11. **성공 종류:** imitation-assisted SUCCESS만 확보했다. Pure/from-scratch PPO SUCCESS는 없다.
12. **Walking 준비:** 아직 아니다. Standing PPO의 좋은 정책 유지와 discovery를 더 해결해야 한다.

## [자료와 재현]

[한국어 결과 그래프](/tmp/microdog_v7/diagnostics.png)와 [PDF](/tmp/microdog_v7/diagnostics.pdf)를 저장했다.

원본 보상표 `/tmp/microdog_v7/reward/reward_audit.json`, MC 비교 `/tmp/microdog_v7/credit/policy_{0,1,5}/comparison.json`, A/B `decision.json`, critic fit 및 B5 직접 검증 JSON, checkpoint·trace·회귀·manifest를 `/tmp/microdog_v7`에 보관한다. `/tmp`는 영구 저장소가 아니며 checkpoint와 대용량 trace는 Git에 추가하지 않았다.

GPU Python은 `/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python`, CPU 물리 회귀는 `/home/fcsl/robot_ws/mujoco/.venv/bin/python`을 사용했다. GPU 환경은 torch2.9.1, rsl_rl5.0.1, mjlab1.3.0, MuJoCo3.10, Warp1.12, RTX4090이다.

완료한 실험의 재현 명령은 다음과 같다. 재실행할 때 기존 결과를 보존하고 새 출력 경로를 지정한다.

```bash
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.reward_alignment --output /tmp/microdog_v7/reward
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.policy_credit --output /tmp/microdog_v7/credit
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.standing_v7 --output /tmp/microdog_v7/v7a --pose-weight .1
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.critic_fit --output /tmp/microdog_v7/critic_fit
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m rl.standing_v7 --output /tmp/microdog_v7/v7b --critic-warm-start /tmp/microdog_v7/critic_fit/critic_64x64_centered.pt
/home/fcsl/robot_ws/mujoco/microduck_rl/.venv/bin/python -m unittest discover -s tests -v
```

1024-step B5 정확도 재검증은 `rl.critic_fit.collect_target`과 `rl.policy_credit.value_metrics`를 별도로 호출하여 저장했다. 국소 component gradient와 그림 생성 보조 script, source SHA256는 `/tmp/microdog_v7`에 보관한다. 같은 seed에서도 GPU 수치 차이가 있을 수 있다.

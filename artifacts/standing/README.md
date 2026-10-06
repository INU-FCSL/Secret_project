# Standing 자료 보존

Standing 기본 물리·control interface 검증은 완료했다. Scripted 및 imitation recovery는 성공했다. From-scratch PPO의 정책 발견과 좋은 정책의 장기 유지 문제는 미해결 연구 항목이며 Walking 개발을 차단하지 않는다.

`manifest.json`에 원본·보존 경로, byte 크기, SHA256, checkpoint 역할과 가능한 configuration을 기록했다. `local/`에는 V3-E75, scripted/zero trajectory, V6 imitation0/PPO1 및 V7 A/B checkpoint, 보고서, 그래프, metadata, SHA256 자료를 보관한다. 원본 `/tmp` 파일은 삭제하지 않았다.

`local/`은 Git에서 제외한다. Git clone만으로 binary가 복원되지는 않으므로 다른 머신으로 옮길 때 이 디렉터리도 별도로 복사해야 한다. SHA256로 복사본을 검증할 수 있다. 이 저장은 같은 디스크의 지속성 보존이며 별도 디스크 백업은 아니다.

checkpoint loader에는 `local/.../ppo/checkpoint_N.pt`를 전달한다. 같은 부모의 `environment.json`과 `config.json`을 함께 보존했다. 기존 진단 script는 과거 `/tmp` 경로를 기본으로 사용하므로 재현 시 입력 경로를 확인한다.

Standing 최종 상태:

- 기본 physics/control interface: 검증 완료
- Scripted 및 imitation recovery: 성공
- From-scratch PPO standing: 미해결
- Standing PPO 추가 튜닝: 중단, 연구 결과 보존
- Walking: 별도 task와 평가 기준으로 진행

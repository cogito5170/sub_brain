# 선행 연구 -- 1차 조사 (2026-10-01)

**확인 수준: 전부 [조각]이다.** 검색 결과 조각만 읽었다. 논문 페이지는 arxiv.org·huggingface.co·semanticscholar가 프록시에 막혀 하나도 열지 못했다. 아래 내용은 원문으로 확인할 때까지 인용하지 말 것.

## 가장 가까운 셋

1. **DeepRewind**, arXiv:2609.36344 (2026) [조각]. 에이전트의 인식 상태(근거·주장·가설·가정·약속)를 그래프로 두고, 위험한 약속을 막으며, 의존성을 따라 되돌린다(약속을 거두면 그것에만 기댄 정당화를 거두고, 아래쪽 항목은 낡음으로 표시한다). **근거 그래프 + 의존성 되돌림에서 매우 가깝다.** 조각에는 nogood, 약한 가정 선택, impasse 기반 LLM 호출, 소형 모델, ReAct 대비 토큰 비교가 보이지 않았다.
2. **CogRec**, arXiv:2512.24113 (2025) [조각]. Soar의 impasse가 날 때만 LLM에게 묻고, 답을 chunking으로 규칙으로 바꾼다. "막힐 때만 LLM"에서 가깝다. TMS는 없다.
3. **NeSyFS**, arXiv:2607.28942 (2026) [조각]. 관측 트리플로 지식그래프 믿음 상태를 유지하고, 반응형 행동이 반복해서 실패하면 느린 사고로 넘긴다. 우리의 STALL·REPEATED_ACTION과 닮았다.

## 그 밖

- NeuSymMS, arXiv:2605.17596 [조각]. LLM이 사실을 뽑고 CLIPS가 모순된 낡은 사실을 지운다. Doyle·de Kleer를 인용한다.
- CoALA, arXiv:2309.02427 [조각]. 언어 에이전트용 인지 아키텍처의 개념 틀이다.
- LLM 블랙보드: arXiv:2507.01701, 2510.01285, 2510.14312 [조각]. 정당화나 되돌림은 없다.
- ReWOO, arXiv:2305.18323 [조각]. 한 번에 계획하며, ReAct 대비 토큰을 크게 줄였다고 보고한다.
- LLMCompiler, arXiv:2312.04511 [조각]. 병렬 함수 호출을 한다.
- "Feedback That Backfires", arXiv:2608.23651 [조각]. 135M~1.7B 모델은 기록에 남은 실패 호출을 되풀이하기 쉽다고 한다.
- "Thinking Costs Tokens", arXiv:2608.27506 [조각]. 구조를 더하면 예산이 작을 때는 손해이고 클 때는 이득이라고 한다. **우리 측정(bench/REPORT.md)과 같은 방향이다.**

## 아직 못 본 곳

- 원문 전부. 특히 DeepRewind가 nogood를 쓰는지, 소형 모델과 토큰을 쟀는지.
- Google Scholar, DBLP, ACL Anthology, OpenReview, KR·IJCAI·AAAI의 TMS+LLM 논문, ACT-R+LLM.
- LLM + ASP/Prolog 믿음 수정. ReWOO·LLMCompiler 후속 연구의 소형 모델 평가.

**"이 조합(LLM이 쓰는 JTMS + nogood + impasse subgoal + ReAct 대비 소형 모델 토큰 측정)은 못 찾았다"는 말은 "없다"가 아니다.**

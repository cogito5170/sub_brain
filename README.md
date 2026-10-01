# sub_brain — LLM 옆에 붙는 보조-뇌

이 도구는 정책을 학습시키는 AI가 아닙니다(XCS+GA 같은 것이 아닙니다). LLM 옆에서 **사고에 필요한 상태를 관리**합니다.
무엇을 믿고 있는지, 왜 믿는지, 무엇이 모순인지, 어디서 막혔는지를 기록하고, **언제 LLM을 다시 부를 가치가 있는지**를 판단합니다.

```
                 LLM  (지식원 가운데 하나)
                  │  ops (JSON)            ▲  brief (막힌 자리 + 다음 연산)
                  ▼                        │
   ┌──────────────────────────────────────────────────┐
   │ Blackboard   GOAL · FACT · ASSUMPTION · CLAIM ·    │  Hearsay-II
   │              QUESTION · ACTION · RECHECK           │
   │ TMS          근거 사슬 · nogood · DDB 되돌림        │  Doyle 1979
   │ Impasse      막힘 감지 → subgoal                   │  SOAR
   │ Planner      STRIPS A*  (tool / cognitive 연산자)  │  Fikes & Nilsson 1971
   │ Verifier     GROUNDED / CONDITIONAL / UNPROVEN ... │
   │ Memory       episodic · semantic · procedural ·    │
   │              failure (세션을 넘어 남는다)           │
   │ CONTROL      다음 사고 작업 하나 + call_llm 여부    │  Hearsay-II scheduler
   └──────────────────────────────────────────────────┘
```

외부 의존성이 없습니다(파이썬 3.10 이상, 표준 라이브러리만 씁니다).

## 붙이는 방법 세 가지

### 1. MCP 서버: Claude Code 등 MCP 클라이언트에 도구로 붙이기

```bash
claude mcp add subbrain -- python3 -m subbrain mcp --session myproblem
# 또는 pip install -e . 후
claude mcp add subbrain -- subbrain mcp --session myproblem
```

| 도구 | 하는 일 |
|---|---|
| `subbrain_ops` | 칠판에 연산을 한꺼번에 적고, 바뀐 믿음과 되돌린 가정을 돌려줌 |
| `subbrain_next` | 다음 사고 작업 하나와 brief를 돌려줌. `call_llm=false`면 더 생각할 필요 없음 |
| `subbrain_why` | 왜 믿는지(IN) 또는 왜 못 믿는지(OUT)를 근거 사슬로 보여 줌 |
| `subbrain_verify` | 주장을 바깥에서 검사 |
| `subbrain_plan` | STRIPS 계획 |
| `subbrain_recall` | 장기 기억 검색 |
| `subbrain_state` | 칠판 전체 |
| `subbrain_close` | 에피소드와 교훈을 기억에 남기고(필요하면) 칠판을 비움 |

세션은 `~/.subbrain/sessions/<이름>.json`, 기억은 `~/.subbrain/memory.jsonl`에 저장됩니다. 위치는 `SUBBRAIN_HOME`으로 바꿀 수 있습니다.

### 2. 고리 드라이버: 보조-뇌가 LLM을 부르는 쪽

```python
from subbrain import AuxBrain, apply
from subbrain.loop import run, claude_cli      # anthropic_api(model) 도 있다

b = AuxBrain("session.json")
apply(b, [{"op": "operator", "name": "read_temperature", "add": ["temp_reading"], "doc": "온도계"},
          {"op": "goal", "text": "기계 A 를 계속 가동해도 되나", "requires": ["verdict"]}])
run(b, claude_cli(), tools={"read_temperature": lambda brain, op: [
    {"op": "fact", "id": "temp_reading", "text": "98도 (한계 85도)"}]})
```

`run`은 `next()`가 `call_llm=True`일 때만 LLM을 부릅니다. 등록된 도구로 풀리는 걸음은 도구로 처리하고, 목표가 다 서면 멈추고, LLM이 불렸는데도 믿음이 하나도 바뀌지 않는 일이 3번 이어지면 `STALL → ASK_USER`로 사람에게 넘기고 멈춥니다.

### 3. CLI

```bash
python3 -m subbrain demo                       # 센서·온도 예제 (LLM 호출 없음)
echo '{"ops":[{"op":"fact","text":"비가 온다","id":"rain"}]}' | python3 -m subbrain ops --session s
python3 -m subbrain next --session s
python3 -m subbrain why wet --session s
python3 -m subbrain run "기계 A 를 계속 가동해도 되나" --session s   # claude -p 로 고리를 돎
```

## LLM이 쓰는 말: 연산(ops)

````
```subbrain
{"ops": [
  {"op": "fact",   "id": "E1", "text": "센서가 정상이다"},
  {"op": "assume", "id": "E2", "text": "온도가 정상이다", "confidence": 0.7},
  {"op": "claim",  "id": "B1", "text": "A는 안전하다", "because": ["E1", "E2"]},
  {"op": "goal",   "text": "A 가동 판정", "requires": ["B1"]}
]}
```
````

`fact` · `assume` · `claim` · `rule` · `observe` · `retract` · `conflict` · `goal` · `abandon` · `question` · `act` · `use` · `remember` · `operator`.
노드는 id로도, 글귀 그대로로도 가리킬 수 있습니다. **`because`가 없는 주장은 믿지 않습니다**(OUT으로 남고 `UNSUPPORTED_CLAIM`이 됩니다). 무엇을 말했는지와 무엇을 믿는지를 이렇게 구분합니다.

## 무엇을 하는가: `python3 -m subbrain demo`

```
=== 도구 관측: 온도 정상(E2) = FALSE  -> E2 를 딛고 선 B1, C 가 무너진다
changed: E2 IN->OUT, B1 IN->OUT, C IN->OUT, not:E2 OUT->IN
IMPASSE CONFLICT  E2[A- c=0.7] 온도가 정상이다
RECHECK B1[C-] A는 안전하다; C[C-] A를 계속 가동해도 된다
NEXT REPLAN(E2) by llm -- 목표가 요구하는 노드가 CONTRADICTED -- 다른 길로 목표를 세우거나 목표를 고쳐라

=== LLM: 두 가정을 세운다 -- 그런데 둘은 함께 설 수 없다 (DDB)
IMPASSE CONFLICT  H1[A- c=0.4] 펌프 전원이 끊겼다
  memory[failure]: 함께 가정하면 모순: 펌프 전원이 끊겼다 / 펌프 전원 계통이 정상이다
ROLLBACK H1[A- c=0.4] 펌프 전원이 끊겼다  (nogood ['H1', 'H2'])
NEXT REPLAN(H1) by llm ...
  or TEST(H1) by llm
```

LLM에게 "다시 생각해 봐"라고 하지 않습니다. **어느 가정이 깨졌고, 그 위에 선 어느 결론이 무너졌는지**를 알려 줍니다.

## 기관별 동작

**TMS** (`tms.py`): 정당화 = in-list가 모두 IN이고 out-list가 모두 OUT일 때 결론이 IN이 됩니다. out-list가 있으면 비단조이므로, 라벨은 well-founded semantics의 교대 고정점으로 매깁니다. `p unless p` 같은 고리는 조용히 OUT으로 접지 않고 `UNDETERMINED`로 두며, 이것이 `LOOP` impasse가 됩니다.

**DDB** (`blackboard.maintain`): 모순 노드가 IN이 되면 그 밑에 깔린 가정 집합을 nogood로 기록하고, 확신도가 가장 낮은 가정(같으면 가장 최근 것)을 내립니다. nogood은 다시 세울 수 없습니다(`assume`이 거절됩니다). 관측끼리만 부딪힌 모순은 내릴 가정이 없으므로 고치지 않고 `REOBSERVE / ASK_USER`로 올립니다. 내려간 노드에 기대던 주장은 `RECHECK`로 표시되고, 무엇이 깨졌는지도 함께 기록됩니다.

**Impasse** (`impasse.py`): `CONFLICT` · `LOOP` · `STALL` · `RECHECK` · `REPEATED_ACTION` · `UNKNOWN` · `MISSING_PREMISE` · `NO_PLAN` · `WEAK_SUPPORT` · `UNSUPPORTED_CLAIM`. `UNKNOWN`과 `MISSING_PREMISE`는 정당화를 아래로 따라가 **맨 끝에서 빠진 노드**를 찾아 subgoal로 만듭니다(SOAR). 버린 목표 쪽에서 무너진 주장은 막힘으로 치지 않고 brief에만 적습니다.

**Verifier** (`verifier.py`): `GROUNDED`(관측에만 기댐) / `CONDITIONAL`(가정에 기댐: 어느 가정이고 확신도가 얼마인지) / `UNPROVEN`(빠진 것) / `UNSUPPORTED` / `CONTRADICTED` / `RETRACTED` / `UNDETERMINED`. 목표가 섰더라도 약한 가정(확신도 0.6 미만, 또는 출처가 llm)에 기대고 있으면 `WEAK_SUPPORT`로 시험을 요구합니다.

**Planner** (`planner.py`): STRIPS 연산자(pre / add / delete / cost)를 A*로 탐색합니다. 막힌 노드를 세울 `tool` 연산자가 있으면 **LLM을 부르지 않고** 그 도구를 다음 걸음으로 냅니다. LLM은 brief의 `TOOLS` 줄을 보고 `use`로 도구 실행을 요청할 수 있습니다.

**Memory** (`memory.py`): BM25로 검색합니다. 한국어는 조사 때문에 단어 단위로는 잘 맞지 않아("센서가"와 "센서") 글자 2-gram도 함께 색인합니다. nogood과 실패한 행동은 `failure` 기억으로 **자동으로** 쌓이고, impasse가 나면 관련 기억이 brief에 붙습니다.

**Control** (`brain.next`): 가장 높은 impasse 하나를 골라 연산 하나와 대안 몇 개를 냅니다. `call_llm`은 그 연산의 실행 주체가 llm일 때만 참입니다.

## 확인한 것

- `python3 -m unittest discover -s tests -t .`: 테스트 35개 통과. TMS(비단조, 홀수/짝수 고리), DDB, nogood 거절, subgoal 생성, STALL 차단, MCP stdio 왕복(하위 프로세스로 실제 서버를 띄워 확인)이 포함됩니다.
- **실제 Claude Code 클라이언트에 MCP로 붙여** `subbrain_ops → next → verify`를 호출하게 했고, `DONE` / `GROUNDED`가 돌아왔습니다.
- **실제 모델로 고리를 돌렸습니다**(`examples/machine_a.py`, `claude -p`, 2026-10-01). 두 번 돌렸습니다.
  - 1차(`use` 연산을 넣기 전): LLM은 모르는 두 가지를 확신도 0.4 가정으로 적고 "조건부 가동"을 주장했습니다. 보조-뇌는 이를 `CONDITIONAL`, `accept=false`로 판정해 받아들이지 않았습니다. 다만 등록된 온도계 도구를 LLM이 볼 수 없어서 쓰지 못했고, 그래서 brief에 `TOOLS` 줄과 `use` 연산을 추가했습니다.
  - 2차: LLM이 `use read_temperature`를 요청했고, 98도를 받았고, 보조-뇌가 "온도 정상" 쪽 길을 `CONFLICT(CONTRADICTED)`로 막았습니다. 최종 판정 "가동 중지"는 관측에만 기대어 섰습니다. **LLM 4회, 도구 1회.**

## ReAct와 비교 (2026-10-01, `bench/REPORT.md`)

**약한 모델(Haiku 4.5, 생각 끔)에서 보조-뇌가 졌습니다.**

| | ReAct | 보조-뇌 |
|---|---:|---:|
| 정답률 (48쌍) | **0.88** | 0.44 |
| 진단 과제 정답률 | **0.79** | 0.00 |
| 과제당 입력 토큰 | **6,669** | 15,095 |
| 과제당 출력 토큰 | **1,344** | 3,154 |

진단 과제는 보조-뇌에 유리하라고 고른 것인데도 더 크게 졌습니다. 틀린 답 24개 중 22개가 낡은 보고서의 미끼 부품이었습니다. 보조-뇌 프로토콜이 과제 문장을 `fact`로 옮겨 적게 만들었고, 약한 모델은 그 "사실"을 믿었습니다. 강한 모델(Sonnet·Opus) 비교는 API 안전 장치 오탐 때문에 망가져서 버렸습니다. 고칠 가설은 REPORT의 7절에 있고, 새 시드로 다시 재야 합니다.

선행 연구 1차 조사는 `docs/prior_art.md`에 있습니다(전부 검색 결과 조각 수준).

## 확인하지 않은 것
- 2차 실험에서 LLM은 원래 목표(`verdict`)를 버리고 `verdict_stop`을 요구하는 새 목표를 세웠습니다. 이 과제에서는 맞는 처리였지만, **목표를 옮겨 DONE을 만드는 길**이 열려 있다는 뜻이기도 합니다. "판정하라"처럼 어느 쪽이든 답이 되는 목표(either-of)는 아직 표현하지 못합니다.
- 실제 모델 실험은 과제 하나, 두 번뿐입니다.
- 선행 연구는 검색 결과 조각만 봤습니다(`docs/prior_art.md`). 가장 가까운 것은 DeepRewind(arXiv:2609.36344)로, 근거 그래프와 의존성 되돌림을 씁니다. 원문은 아직 읽지 못했습니다.

## 한계

- 라벨은 바뀔 때마다 처음부터 다시 매깁니다(증분 갱신 없음). 노드 수천 개까지는 문제가 없지만 그 이상은 재 보지 않았습니다.
- DDB의 가정 선택은 "확신도가 가장 낮은 것"이라는 단순한 규칙입니다. 목표에 유리한 쪽을 남기도록 하지 않은 것은 의도한 것입니다(동기화된 추론 방지).
- 노드 글귀의 의미는 비교하지 않습니다. "온도 정상"과 "온도가 정상이다"는 서로 다른 노드입니다. 같은 것을 가리키려면 id를 쓰세요.

"""실제 LLM(claude -p)으로 보조-뇌 고리를 돌리는 예. 모델 호출이 몇 번 일어난다.

    python3 examples/machine_a.py
"""
import json, sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from subbrain import AuxBrain, apply
from subbrain.loop import run, claude_cli

b = AuxBrain(memory=False)
apply(b, [
    {"op": "fact", "id": "sensor_ok", "text": "온도 센서 자가진단이 정상이다"},
    {"op": "fact", "id": "load_high", "text": "기계 A 의 부하가 정격의 95% 다"},
    {"op": "operator", "name": "read_temperature", "add": ["temp_reading"], "kind": "tool",
     "doc": "온도계를 읽는다"},
    {"op": "goal", "id": "G", "text": "기계 A 를 계속 가동해도 되는지 판정하라", "requires": ["verdict"]},
])
calls = []
def llm(system, user):
    out = claude_cli(timeout=240)(system, user)
    calls.append({"in": user, "out": out})
    return out
def read_temperature(brain, op):
    return [{"op": "fact", "id": "temp_reading", "text": "온도계 판독: 98도 (허용 한계 85도)"}]
res = run(b, llm, {"read_temperature": read_temperature}, max_steps=10,
          task="기계 A 를 계속 가동해도 되는가? verdict 노드를 근거와 함께 세워라.",
          on_step=lambda r: print(json.dumps({k: r.get(k) for k in ("step","status","impasse","op","llm_ops")}, ensure_ascii=False)[:300], flush=True))
print("STATS", res["stats"], "FINAL", res["final"])
for c in calls:
    print("----- IN\n" + c["in"] + "\n----- OUT\n" + c["out"][:1500])
print("VERIFY", json.dumps(b.verify("verdict"), ensure_ascii=False))
print(b.next()["brief"])

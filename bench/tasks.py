"""비교용 과제 생성기 -- 가상 세계라 모델이 외워서 풀 수 없다. 시드로 재현된다.

A  multihop   가상 지식베이스 다단계 질의 (HotpotQA 꼴, ReAct 의 본래 자리)
              절반은 '흔히 X 가 세웠다고 알려졌으나 등기부상 설립자는 Y' 같은 함정 문장을 넣는다
B  diagnose   직렬 설비의 고장 부품 찾기. 어제 보고서가 진짜 고장 부품을 '정상' 이라 하고
              엉뚱한 부품을 '교체 권고' 한다 -- 측정으로 뒤집어야 한다 (보조-뇌에 유리한 자리)

모든 도구는 인자 하나를 받는다. 인자 이름이 틀려도 첫 값을 쓴다(형식 실수로 지는 것을 줄인다).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

SYL = "가나다라마바사아자차카타파하고노도로모보소오조초코토포호구누두루무부수우주추쿠투푸후"
COUNTRY_SUF, CITY_SUF = ["니아", "란드", "스탄", "모라"], ["포르", "빌", "하임", "베르크", "도"]
PRODUCTS = ["유리 렌즈", "풍력 날개", "곡물 건조기", "광섬유", "전동 자전거", "종이 포장재", "해수 담수기", "목재 가구"]


def _name(rng, n=2, suf=""):
    return "".join(rng.choice(SYL) for _ in range(n)) + suf


@dataclass
class Task:
    id: str
    family: str
    question: str
    answer: str
    distractors: list[str]
    tools: dict                      # name -> (doc, fn(arg)->str)
    meta: dict = field(default_factory=dict)

    @staticmethod
    def _forms(x: str) -> list[str]:
        """'밸브V4' 는 'V4' 로 줄여 써도 같은 답이다(실측: ReAct 정답 둘이 이것 때문에 오답 처리됐다)."""
        m = re.search(r"[A-Z]\d+$", x)
        return [x, m.group()] if m else [x]

    def check(self, ans: str | None) -> bool:
        if not ans:
            return False
        a = ans.replace(" ", "")
        if not any(re.search(re.escape(f) + r"(?!\d)", a) for f in self._forms(self.answer)):
            return False
        return not any(re.search(re.escape(f) + r"(?!\d)", a)
                       for d in self.distractors if d != self.answer for f in self._forms(d))


def _first(kw):
    return str(next(iter(kw.values()), "")).strip().strip("'\"[]") if kw else ""


def multihop(seed: int) -> Task:
    rng = random.Random(seed)
    used: set[str] = set()

    def uniq(n, suf):
        while True:
            x = _name(rng, n, suf)
            if not any(x in u or u in x for u in used):      # 어느 이름도 다른 이름의 부분이 아니게
                used.add(x)
                return x
    countries = [uniq(2, rng.choice(COUNTRY_SUF)) for _ in range(3)]
    cities = {uniq(2, rng.choice(CITY_SUF)): countries[i // 2] for i in range(6)}
    city_list = list(cities)
    surname = rng.choice("김이박최정강조윤장임")
    people = {f"{surname}{uniq(2, '')}": rng.choice(city_list) for _ in range(5)}   # 같은 성 -- 헷갈리게
    plist = list(people)
    comps = {}
    for _ in range(4):
        c = uniq(2, rng.choice(["전자", "산업", "공방", "상사"]))
        comps[c] = {"founder": rng.choice(plist), "hq": rng.choice(city_list),
                    "year": rng.randint(1950, 2015), "product": rng.choice(PRODUCTS)}
    target = rng.choice(list(comps))
    rumor = seed % 2 == 0
    info = comps[target]
    wrong = rng.choice([p for p in plist if p != info["founder"]])
    # 함정이 함정이 되려면 틀린 사람의 출생 도시가 다른 나라여야 한다
    if rumor:
        others = [p for p in plist if cities[people[p]] != cities[people[info["founder"]]]]
        if others:
            wrong = rng.choice(others)

    docs: dict[str, str] = {}
    for p, c in people.items():
        docs[p] = f"{p}: {c}에서 태어난 기업인이다. " + rng.choice(["대학에서 화학을 공부했다.", "젊은 시절 항해사였다.", "두 번 결혼했다."])
    for c, k in cities.items():
        docs[c] = f"{c}: {k}의 항구 도시다. 인구는 약 {rng.randint(5, 90)}만 명." if rng.random() < .5 else f"{c}: {k} 북부의 도시다."
    for k in countries:
        docs[k] = f"{k}: 대륙 서쪽의 나라. 수도는 {[c for c, v in cities.items() if v == k][0]}."
    for c, v in comps.items():
        if c == target and rumor:
            docs[c] = (f"{c}: {v['year']}년 설립. 흔히 {wrong}가 세운 회사로 알려져 있으나, 등기부상 설립자는 "
                       f"{v['founder']}다. 본사는 {v['hq']}. 주력 제품은 {v['product']}.")
        else:
            docs[c] = f"{c}: {v['year']}년 {v['founder']}가 세운 회사. 본사는 {v['hq']}. 주력 제품은 {v['product']}."

    def lookup(**kw):
        q = _first(kw)
        if q in docs:
            return docs[q]
        near = [k for k in docs if q and (q in k or k in q or k[:2] == q[:2])]
        return "찾을 수 없음." + (f" 비슷한 이름: {', '.join(near[:4])}" if near else "")

    kind = seed % 4 in (0, 1)        # 둘 다 rumor 와 섞이게
    if kind:
        q = f"{target}를 세운 사람이 태어난 도시는 어느 나라에 있는가? 나라 이름만 답하라."
        ans, dis = cities[people[info["founder"]]], countries
    else:
        q = f"{target}의 설립자가 태어난 도시는 어디인가? 도시 이름만 답하라."
        ans, dis = people[info["founder"]], city_list
    return Task(f"A{seed}", "multihop", q, ans, dis,
                {"lookup": ("lookup(name): 사람·회사·도시·나라 이름 하나를 넣으면 그 항목 설명을 돌려준다", lookup)},
                {"rumor": rumor, "hops": 3 if kind else 2, "trap": cities[people[wrong]] if kind else people[wrong]})


KINDS = [("펌프", "P"), ("필터", "F"), ("밸브", "V"), ("열교환기", "H"), ("냉각기", "C"), ("분배기", "D"), ("정류기", "R")]


def diagnose(seed: int) -> Task:
    rng = random.Random(10_000 + seed)
    n = rng.randint(5, 7)
    kinds = rng.sample(KINDS, n)
    comps = [f"{k}{t}{i + 1}" for i, (k, t) in enumerate(kinds)]
    faulty = rng.randrange(1, n)              # 첫 부품은 고장 아님 -- 보고서 함정이 의미 있게
    decoy = rng.choice([i for i in range(n) if i != faulty])
    order = " -> ".join(comps)
    report = [f"{c}: 정상" for i, c in enumerate(comps) if i != decoy]
    rng.shuffle(report)
    report.append(f"{comps[decoy]}: 소음 있음, 교체 권고")
    q = (f"냉각 라인은 다음 순서로 직렬 연결돼 있다: 원수 -> {order} -> 출력. 지금 출력 유량이 낮다. "
         f"고장난 부품은 정확히 하나다. 판정 규칙: 어떤 부품의 입구 유량이 정상인데 출구 유량이 낮으면 그 부품이 고장이다 "
         f"(첫 부품의 입구는 원수라 늘 정상). 어제 작성된 점검 보고서: " + "; ".join(report) +
         ". 고장난 부품 이름 하나만 답하라.")

    def measure(**kw):
        p = _first(kw)
        hit = [i for i, c in enumerate(comps) if p and (c == p or c in p or c.endswith(p))]   # "F7" 처럼 줄여 불러도 된다
        if not hit:
            return f"측정 지점을 모른다. 지점: {', '.join(comps)} (각 부품의 출구)"
        i = hit[0]
        return f"{comps[i]} 출구 유량: {'낮음' if i >= faulty else '정상'}"
    return Task(f"B{seed}", "diagnose", q, comps[faulty], comps,
                {"measure": ("measure(point): 부품 이름을 넣으면 그 부품 출구의 현재 유량(정상/낮음)을 잰다", measure)},
                {"n": n, "faulty_pos": faulty, "decoy": comps[decoy]})


def by_id(tid: str) -> Task:
    return (multihop if tid[0] == "A" else diagnose)(int(tid[1:]))


def suite(n_per: int = 12, start: int = 0) -> list[Task]:
    out = []
    for i in range(start, start + n_per):
        out.append(multihop(i))
        out.append(diagnose(i))
    return out

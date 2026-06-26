"""60-second tour: watch append-only go stale while the bi-temporal graph stays
current — on the exact same conversation.

    python examples/quickstart.py
"""

from agent_memory_lab.backends import build_backend
from agent_memory_lab.stalebench import controlled_episode
from agent_memory_lab.stalebench.metrics import evaluate_probe

ep = controlled_episode()

print("Conversation (the user changes their mind over time):")
for t in ep.turns:
    print(f"  [{t.idx}] {t.text}")

for name in ("append-only", "bitemporal"):
    system = build_backend(name, "mock")
    system.ingest(ep.turns)
    print(f"\n[{name}]")
    for pr in ep.probes:
        rec = evaluate_probe(system, pr, ep.ask_turn)
        flag = "STALE" if rec["stale"] else ("ok" if rec["correct"] else "miss")
        print(f"  Q: {pr.predicate:<9}({pr.kind})  A: {str(rec['predicted']):<14}"
              f"  gold={str(rec['gold']):<9} [{flag}]")
    print(f"  store size={system.storage.size()}  edge-closes={system.prof.closes}")

"""System: test_agent_bounds module.

Strict validation of tool allow-lists and directive registries.
"""

from __future__ import annotations

# =============================================================================
#                        ********* TEST SUITE *********                        
#                 Unit verifications for pipeline correctness.                 
# =============================================================================

import pytest

from acv.knowledge import siesta
from acv.tools import registry


class TestToolAllowLists:
    def test_the_critic_may_inspect_but_not_spend(self):
        allowed = registry.ALLOWED["critic"]
        assert "read_structure" in allowed and "check_symmetry" in allowed
        assert not any(t.mutates for t in registry.for_agent("critic"))

    def test_an_agent_cannot_call_a_tool_outside_its_allow_list(self):
        # Enforce boundary separation between agents.
        with pytest.raises(registry.ToolRefused):
            registry.call("translator", "check_symmetry", {"a": 3.6, "b": 3.6})

    def test_declarations_are_restricted_to_the_allow_list(self):
        for agent, allowed in registry.ALLOWED.items():
            names = {d["name"] for d in registry.declarations(agent)}
            assert names <= allowed, f"{agent} was offered a tool it may not call"

    def test_every_allow_listed_tool_actually_exists(self):
        for agent, allowed in registry.ALLOWED.items():
            missing = {n for n in allowed if registry.get(n) is None}
            assert not missing, f"{agent} is allow-listed for missing tools: {missing}"


class TestSiestaDirectives:
    def test_a_real_directive_is_recognised(self):
        assert siesta.exists("MeshCutoff")
        assert siesta.exists("meshcutoff")          # fdf is case-insensitive

    def test_an_invented_directive_is_reported_not_silently_accepted(self):
        result = siesta.validate({"MeshCutoff": 350, "NotARealFlag": 1})
        assert result["known"] == {"MeshCutoff": 350}
        assert "NotARealFlag" in result["unknown"]
        assert "NotARealFlag" in result["note"]

    def test_lookup_of_an_unknown_directive_suggests_alternatives(self):
        record = siesta.lookup("MeshCutof")            # typo
        assert record["exists"] is False
        assert "MeshCutoff" in record["did_you_mean"]

    def test_the_index_is_populated(self):
        # Ensure the compiled binary index is populated.
        assert len(siesta.search("", limit=1)["matches"]) == 0
        assert siesta.exists("PAO.BasisSize")
        assert siesta.version() is not None

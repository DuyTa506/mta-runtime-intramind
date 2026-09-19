"""Actual Temporal replay for standard, agentic and custom DOCX drafting."""

import json
import os
import re

import pytest
from feature_harness import feature_environment

pytestmark = pytest.mark.integration


@pytest.mark.parametrize("kind", ["standard", "agentic", "custom"])
async def test_draft_recovers_every_model_boundary_and_replays(store, tmp_path, kind):
    if os.environ.get("RUNTIME_TEST_AI_FEATURES") != "yes":
        pytest.skip("explicit AI dependency environment and opt-in required")
    import yaml
    from agents.context_synthesis.durable.custom_template import LEAVES as CUSTOM_LEAVES
    from agents.context_synthesis.durable.draft_leaves import LEAVES as DRAFT_LEAVES
    from agents.context_synthesis.durable.leaves import LEAVES
    from agents.context_synthesis.pipeline.slot_filler import LEDGER_PATH
    from agents.context_synthesis.templates.docx import Blueprint, FieldSpan, docx_bytes
    from api.background.common.models import ModelActivities
    from api.background.draft.activities import DraftActivities
    from api.background.draft.agent import DraftAgentActivities
    from api.background.draft.workflows import WORKFLOWS
    from docx import Document
    from tests.test_context_synthesis.test_linch_synthesis import (
        FakeDocStore,
        FakeEmbedder,
        FakeLLM,
        make_template,
    )

    template = make_template()
    (tmp_path / "unit_test_template.yaml").write_text(yaml.safe_dump(template.model_dump()))
    document = Document()
    document.add_paragraph("Doanh thu: {{amount}}")
    blueprint = Blueprint(
        title="Báo cáo",
        fields=[
            FieldSpan(
                block_id="b0",
                match="{{amount}}",
                label="Doanh thu",
                instruction="Doanh thu kỳ báo cáo",
            )
        ],
    )
    source_docx = docx_bytes(document)
    llm, agent_turns, publications = FakeLLM(), [], []

    async def respond(reservation, payload):
        messages = payload["messages"]
        if payload.get("tools"):
            agent_turns.append(reservation.attempt_id)
            if len(agent_turns) == 1:
                name, arguments = "read_file", {"path": LEDGER_PATH}
            else:
                note = re.search(r"\[([0-9a-f-]{36})\]", messages[-1]["content"]).group(1)
                name, arguments = (
                    "emit_fill",
                    {"text": "Doanh thu đạt 120 tỷ đồng.", "cited_note_ids": [note]},
                )
            return {
                "choices": [
                    {
                        "message": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{len(agent_turns)}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(arguments)},
                                }
                            ],
                        }
                    }
                ]
            }
        if kind == "custom":
            args = json.loads(messages[-1]["content"])
            if "proposed_fills" in args:
                value = {key: "supported" for key in args["proposed_fills"]}
            else:
                value = {
                    "fields": {
                        key: {
                            "status": "filled",
                            "text": "120 tỷ",
                            "evidence": [
                                {
                                    "source_id": "s0",
                                    "quote": "doanh thu 120 tỷ",
                                }
                            ],
                        }
                        for key in args["fields"]
                    }
                }
            text = json.dumps(value)
        else:
            text = await llm.agenerate(
                user_prompt=messages[-1]["content"], system_prompt=messages[0]["content"]
            )
        return {"choices": [{"message": {"content": text}}]}

    def publication(*args):
        publications.append(args)
        if len(publications) == 1:
            raise RuntimeError("publication acknowledgement lost")

    def build_activities(factory):
        draft = DraftActivities(
            factory,
            model_profile="test",
            store_factory=FakeDocStore,
            embedder_factory=FakeEmbedder,
            template_directory=lambda: str(tmp_path),
        )
        draft.custom.template_loader = lambda *_: {
            "docx": source_docx,
            "blueprint": blueprint.model_dump(),
        }
        draft.custom.publication = publication
        return [
            *draft.registered(),
            *draft.custom.registered(),
            *DraftAgentActivities(
                factory, model_profile="test", embedder_factory=FakeEmbedder
            ).registered(),
            ModelActivities(factory, leaves=LEAVES | DRAFT_LEAVES | CUSTOM_LEAVES).plan,
        ]

    async with feature_environment(
        store,
        name="draft",
        workflows=WORKFLOWS,
        build_activities=build_activities,
        respond=respond,
        allow_tool_calls=True,
    ) as env:
        status, result = await env.submit(
            {
                "document_ids": ["d1"],
                "template_id": "custom_test" if kind == "custom" else template.template_id,
                "options": {
                    "single_call_enabled": False,
                    "template_fit_guard": False,
                    "fill_agentic": kind == "agentic",
                    "dedup_threshold": 0.99,
                },
            }
        )
        assert status["state"] == "SUCCEEDED"
        assert "120" in result["draft_markdown"]
        assert len(env.calls) == result["telemetry"]["llm_calls"]
        if kind == "custom":
            assert len(publications) == 2
            assert publications[0] == publications[1]
            assert publications[0][-1].startswith(b"PK")
        if kind == "agentic":
            assert len(agent_turns) == 2
        await env.replay(status["root_id"])
        assert len(env.calls) == result["telemetry"]["llm_calls"]

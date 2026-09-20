"""Temporal delivery integration. Broker identity survives activity retries."""

import asyncio
import logging
import random
from datetime import UTC, datetime
from typing import Any

from temporalio import activity
from temporalio.client import Client
from temporalio.common import WorkflowIDReusePolicy
from temporalio.exceptions import ApplicationError, CancelledError, WorkflowAlreadyStartedError
from temporalio.service import RPCError, RPCStatusCode

from .artifacts import ArtifactPort
from .contracts import (
    AdmissionDenied,
    Artifact,
    EmbeddingOperationSpec,
    InferenceOperation,
    NotFound,
    OperationSpec,
    RuntimeConflict,
    SpeechOperationSpec,
)
from .store import Store

logger = logging.getLogger(__name__)


class BrokerActivities:
    def __init__(self, store: Store, artifacts: ArtifactPort | None = None):
        self.store = store
        self.artifacts = artifacts

    @activity.defn(name="runtime.submit_or_attach_llm")
    async def submit_or_attach(self, payload: dict) -> dict:
        return await self._submit(OperationSpec.model_validate(payload))

    @activity.defn(name="runtime.submit_or_attach_speech")
    async def submit_speech(self, payload: dict) -> dict:
        return await self._submit(SpeechOperationSpec.model_validate(payload))

    @activity.defn(name="runtime.submit_or_attach_embedding")
    async def submit_embedding(self, payload: dict) -> dict:
        return await self._submit(EmbeddingOperationSpec.model_validate(payload))

    async def _submit(self, spec: InferenceOperation) -> dict:
        try:
            await self.store.submit_operation(spec)
            result = await self.store.operation(spec.operation_id, spec.tenant_id)
            if result["state"] == "SUCCEEDED":
                return result["result"]
            if result["state"] == "CANCELLED":
                raise CancelledError("operation cancelled")
            if result["state"] == "FAILED":
                raise ApplicationError(
                    result["wait_reason"] or "operation_failed",
                    type="OperationFailed",
                    non_retryable=True,
                )
            await self.store.bind_completion(
                spec.operation_id, spec.tenant_id, activity.info().task_token
            )
        except (RuntimeConflict, NotFound) as exc:
            raise ApplicationError(str(exc), non_retryable=True) from exc
        except AdmissionDenied as exc:
            # Durable activity retry applies producer backpressure; no permit
            # is held. Root deadline bounds retries, including attach retries.
            raise ApplicationError(
                str(exc),
                type="AdmissionBackpressure" if exc.retryable else "AdmissionRejected",
                non_retryable=not exc.retryable,
            ) from exc
        activity.raise_complete_async()

    @activity.defn(name="runtime.finish_run")
    async def finish(self, payload: dict[str, Any]) -> None:
        ref = Artifact.model_validate(payload["result"]) if payload.get("result") else None
        if ref and self.artifacts:
            await self.artifacts.get(ref)
        try:
            await self.store.finish_run(
                payload["root_id"],
                payload["tenant_id"],
                payload["state"],
                ref,
                payload.get("reason"),
            )
            committed = await self.store.run(payload["root_id"], payload["tenant_id"])
        except (RuntimeConflict, NotFound) as exc:
            raise ApplicationError(str(exc), non_retryable=True) from exc
        if payload["state"] in ("SUCCEEDED", "PARTIAL") and committed["state"] != payload["state"]:
            # A late result or concurrent cancellation must not let Temporal
            # publish workflow success after the ledger rejected that result.
            raise ApplicationError(
                committed["terminal_reason"] or "terminal_result_rejected",
                type="TerminalResultRejected",
                non_retryable=True,
            )


class OutboxPublisher:
    """Deliver independent events with durable retry times and bounded RPC concurrency.

    Events are never discarded after repeated failures. Their creation time and
    delivery count remain in the ledger for outbox-age alerts and investigation.
    """

    def __init__(
        self,
        store: Store,
        client: Client,
        owner_id: str,
        *,
        batch_size: int = 8,
        delivery_timeout_seconds: float = 20,
    ):
        if not 1 <= batch_size <= 32:
            raise ValueError("outbox batch size must be between 1 and 32")
        # Leave time to persist failure before the store's 30-second lease ends.
        if not 0 < delivery_timeout_seconds <= 20:
            raise ValueError("outbox delivery timeout must be positive and at most 20 seconds")
        self.store, self.client, self.owner_id = store, client, owner_id
        self.batch_size = batch_size
        self.delivery_timeout_seconds = delivery_timeout_seconds

    async def tick(self) -> int:
        events = await self.store.claim_events(self.owner_id, limit=self.batch_size)
        # Every claimed event starts immediately so none waits behind another
        # RPC until its lease has already expired.
        await asyncio.gather(*(self._publish(event) for event in events))
        return len(events)

    async def _publish(self, event: dict[str, Any]) -> None:
        try:
            async with asyncio.timeout(self.delivery_timeout_seconds):
                try:
                    await self._deliver(event)
                except RPCError as exc:
                    # A stale activity token is terminal for this binding only.
                    # Activity retry attaches a fresh token to the same operation.
                    if (
                        event["kind"] != "complete_activity"
                        or exc.status != RPCStatusCode.NOT_FOUND
                    ):
                        raise
                await self.store.delivered(event["event_id"], self.owner_id, event["deliveries"])
        except Exception as exc:
            # Do not log exception text or event payloads: RPCs may contain task
            # inputs. Only event identity and error category are operational data.
            logger.warning(
                "Outbox delivery deferred event_id=%s kind=%s delivery=%s error=%s",
                event["event_id"],
                event["kind"],
                event["deliveries"],
                type(exc).__name__,
            )
            cap = min(300, 2 ** min(max(0, event["deliveries"] - 1), 9))
            delay = random.uniform(1, cap)
            try:
                async with asyncio.timeout(5):
                    await self.store.defer_event(
                        event["event_id"], self.owner_id, event["deliveries"], delay
                    )
            except Exception as retry_error:
                # If the database is unavailable, the unacknowledged lease
                # expires and remains recoverable by any publisher.
                logger.error(
                    "Outbox retry persistence failed event_id=%s error=%s",
                    event["event_id"],
                    type(retry_error).__name__,
                )

    async def _deliver(self, event: dict[str, Any]) -> None:
        if event["kind"] == "start_workflow":
            spec = event["payload"]
            remaining = datetime.fromisoformat(spec["input"]["deadline"]) - datetime.now(UTC)
            if remaining.total_seconds() <= 0:
                await self.store.finish_run(
                    event["aggregate_id"],
                    spec["input"]["tenant_id"],
                    "FAILED",
                    reason="deadline_exceeded",
                )
                return
            try:
                await self.client.start_workflow(
                    spec["workflow_type"],
                    spec["input"],
                    id=event["aggregate_id"],
                    task_queue=spec["task_queue"],
                    execution_timeout=remaining,
                    id_reuse_policy=WorkflowIDReusePolicy.REJECT_DUPLICATE,
                )
            except WorkflowAlreadyStartedError:
                pass
        elif event["kind"] == "cancel_workflow":
            await self.client.get_workflow_handle(event["aggregate_id"]).cancel()
        elif event["kind"] == "complete_activity":
            binding = await self.store.binding(event["payload"]["binding_id"])
            handle = self.client.get_async_activity_handle(task_token=binding["task_token"])
            if binding["state"] == "SUCCEEDED":
                await handle.complete(binding["result"])
            elif binding["state"] == "CANCELLED":
                await handle.report_cancellation()
            elif binding["state"] == "FAILED":
                await handle.fail(
                    ApplicationError(
                        binding["wait_reason"] or "operation_failed",
                        type="OperationFailed",
                        non_retryable=True,
                    )
                )
            else:
                raise RuntimeConflict("completion binding is not terminal")
        else:
            raise ValueError("unsupported outbox event")

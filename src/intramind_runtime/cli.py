"""Separate API, broker activity worker, executors and outbox processes."""

import argparse
import asyncio
import json
import logging
import os
import shutil
import signal
import subprocess
from contextlib import suppress
from datetime import timedelta
from pathlib import Path
from urllib.parse import urlparse
from uuid import uuid4

import httpx
import uvicorn
from minio import Minio
from temporalio.client import Client
from temporalio.common import VersioningBehavior, WorkerDeploymentVersion
from temporalio.worker import Worker, WorkerDeploymentConfig

from .api import create_app
from .artifacts import MinioArtifacts
from .contracts import EmbeddingPoolSpec, PoolSpec, RerankPoolSpec, SpeechPoolSpec, parse_pool
from .drivers import OpenAICompletionDriver
from .embedding import EmbeddingPreparer, EmbeddingProfile, ServingEmbeddingDriver
from .epochs import epoch_lags_engine
from .executor import Executor
from .preparation import LlamaCppPromptSizer
from .rerank import RerankProfile
from .settings import Settings
from .speech import ServingSpeechDriver, SpeechPreparer, SpeechProfile
from .store import Store
from .temporal_adapter import BrokerActivities, OutboxPublisher
from .wakeup import Wakeup


def artifacts(settings):
    return MinioArtifacts(
        Minio(
            settings.minio_endpoint,
            access_key=settings.minio_access_key.get_secret_value(),
            secret_key=settings.minio_secret_key.get_secret_value(),
            secure=settings.minio_secure,
        ),
        settings.minio_bucket,
        max_bytes=settings.artifact_max_bytes,
    )


def preparers(config):
    result = {}
    for pool in config["pools"]:
        if pool["admission"].get("kind", "llm") != "llm":
            continue
        sizing = pool.get("prompt_sizing")
        if not sizing:
            continue
        spec = PoolSpec.model_validate(pool["admission"])
        if sizing.get("validated_profile_id") != spec.profile_id:
            raise ValueError("tokenizer/template contract must match the capacity profile")
        if spec.model_profile in result:
            raise ValueError(
                "ambiguous model profile: define one pinned sizing contract per model profile"
            )
        client = httpx.AsyncClient(
            base_url=sizing["base_url"].rstrip("/") + "/",
            headers={"Authorization": f"Bearer {os.environ[pool['api_key_env']]}"},
            timeout=30,
            follow_redirects=False,
            transport=httpx.AsyncHTTPTransport(retries=0),
        )
        result[spec.model_profile] = LlamaCppPromptSizer(
            client,
            model=pool["model"],
            capacity_profile_id=spec.profile_id,
            context_limit=spec.context_limit,
            token_margin=sizing["token_margin"],
            expected_output_tokens=sizing["expected_output_tokens"],
            response_formats=frozenset(sizing.get("response_formats", ["text"])),
            allow_tool_calls=sizing.get("allow_tool_calls", False),
        )
    return result


def speech_profiles(config):
    result = {}
    for pool in config["pools"]:
        spec = parse_pool(pool["admission"])
        if not isinstance(spec, SpeechPoolSpec):
            continue
        qualification = pool["speech"]
        if (qualification["validated_profile_id"] != spec.profile_id
            or qualification["termination_contract"] != "termination-v1"):
            raise ValueError("speech contract must match its qualified capacity profile")
        if spec.model_profile in result:
            raise ValueError("ambiguous speech profile")
        result[spec.model_profile] = SpeechProfile(
            model_profile=spec.model_profile, capacity_profile_id=spec.profile_id,
            character_limit=spec.character_limit, sample_rate=qualification["sample_rate"],
            max_audio_bytes=qualification["max_audio_bytes"], voices=qualification["voices"],
        )
    return result


def embedding_profiles(config):
    result = {}
    for pool in config["pools"]:
        spec = parse_pool(pool["admission"])
        if not isinstance(spec, EmbeddingPoolSpec):
            continue
        qualification = pool["embedding"]
        if (qualification["validated_profile_id"] != spec.profile_id
            or qualification["termination_contract"] != "termination-v1"):
            raise ValueError("embedding contract must match its qualified capacity profile")
        if spec.model_profile in result:
            raise ValueError("ambiguous embedding profile")
        result[spec.model_profile] = EmbeddingProfile(
            model_profile=spec.model_profile, capacity_profile_id=spec.profile_id,
            model=qualification["model"], model_revision=spec.model_revision,
            dimension=qualification["dimension"], max_batch_size=spec.max_batch_size,
            character_limit=spec.character_limit,
            max_text_characters=qualification["max_text_characters"],
            max_response_bytes=qualification["max_response_bytes"],
        )
    return result


def rerank_profiles(config):
    result = {}
    for pool in config["pools"]:
        spec = parse_pool(pool["admission"])
        if not isinstance(spec, RerankPoolSpec):
            continue
        qualification = pool["rerank"]
        if (qualification["validated_profile_id"] != spec.profile_id
            or qualification["termination_contract"] != "termination-v1"):
            raise ValueError("rerank contract must match its qualified capacity profile")
        if spec.model_profile in result:
            raise ValueError("ambiguous rerank profile")
        result[spec.model_profile] = RerankProfile(
            model_profile=spec.model_profile, capacity_profile_id=spec.profile_id,
            model_revision=spec.model_revision, character_limit=spec.character_limit,
            max_batch_size=spec.max_batch_size, max_response_bytes=qualification["max_response_bytes"])
    return result


def engine_driver(pool, speech, embeddings=None):
    spec = parse_pool(pool["admission"])
    if isinstance(spec, RerankPoolSpec):
        raise ValueError("rerank pools serve direct requests only")
    if isinstance(spec, EmbeddingPoolSpec):
        return ServingEmbeddingDriver(pool["base_url"], embeddings[spec.model_profile],
            api_key=os.environ[pool["api_key_env"]] if pool.get("api_key_env") else None)
    if isinstance(spec, SpeechPoolSpec):
        return ServingSpeechDriver(pool["base_url"], speech[spec.model_profile],
            api_key=os.environ[pool["api_key_env"]] if pool.get("api_key_env") else None)
    return OpenAICompletionDriver(pool["base_url"], os.environ[pool["api_key_env"]], pool["model"])


def direct_proxies(config, store, sizing):
    """Enable direct HTTP only for one qualified, pinned pool per model profile."""
    from .direct_proxy import DirectProxy

    selected = {}
    embeddings, rerankers = embedding_profiles(config), rerank_profiles(config)
    for pool in config["pools"]:
        enabled = pool.get("direct_enabled", False)
        if type(enabled) is not bool:
            raise ValueError("direct_enabled must be boolean")
        if not enabled:
            continue
        spec = parse_pool(pool["admission"])
        if spec.model_profile in selected:
            raise ValueError("ambiguous direct model profile")
        profile = None
        if spec.kind == "llm":
            preparer = sizing.get(spec.model_profile)
            if (preparer is None or preparer.profile_id != spec.profile_id
                or preparer.model != pool["model"]):
                raise ValueError("direct inference requires matching qualified prompt sizing")
        elif spec.kind in {"embedding", "rerank"}:
            profile = (embeddings if spec.kind == "embedding" else rerankers)[spec.model_profile]
        else:
            raise ValueError("no qualified direct contract for this pool kind")
        api_key = os.environ[pool["api_key_env"]] if pool.get("api_key_env") else None
        selected[spec.model_profile] = (pool, spec, profile, api_key)
    return {
        key: DirectProxy(store, pool=spec, model=pool.get("model"), profile=profile,
            timeout_seconds=pool.get("attempt_timeout_seconds", 1800),
            workload_deadline_seconds=pool.get("workload_deadline_seconds"),
            max_response_bytes=profile.max_response_bytes if profile else 16*1024*1024,
            client=httpx.AsyncClient(
            base_url=pool["base_url"].rstrip("/") + "/",
            headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
            timeout=httpx.Timeout(None, connect=10), follow_redirects=False,
            transport=httpx.AsyncHTTPTransport(retries=0)))
        for key, (pool, spec, profile, api_key) in selected.items()
    }


def engine_started_at(host: str) -> str | None:
    """Container start time when the docker CLI can see it, otherwise unknown."""
    if shutil.which("docker") is None or not host:
        return None
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.StartedAt}}", host],
        capture_output=True, text=True, timeout=15, check=False,
    )
    if result.returncode:
        return None
    started = result.stdout.strip()
    return started or None


def reject_stale_engine_epochs(config, started_at=engine_started_at) -> None:
    """Refuse to load a timestamp epoch older than the engine instance now running."""
    for pool in config.get("pools", []):
        admission = pool.get("admission") or {}
        epoch = admission.get("engine_epoch", "")
        host = urlparse(pool.get("base_url") or "").hostname
        started = started_at(host) if host else None
        if started and epoch_lags_engine(epoch, started):
            raise ValueError(
                f"Pool {admission.get('pool_id')} engine epoch {epoch} is older than "
                f"{host} start {started}"
            )


async def services(command, settings, config):
    reject_stale_engine_epochs(config)
    store = Store(settings.database_url.get_secret_value(), lease_seconds=settings.lease_seconds)
    tasks = []
    drivers = []
    wakeup = Wakeup(settings.database_url.get_secret_value())
    stop = asyncio.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        asyncio.get_running_loop().add_signal_handler(sig, stop.set)

    async def repeat(fn, delay, *, on_events=False):
        while not stop.is_set():
            generation = wakeup.generation
            try:
                if await fn() and on_events:
                    continue
            except Exception:
                logging.exception("runtime %s tick failed", command)
            try:
                if on_events:
                    await wakeup.wait(generation, timeout=delay)
                else:
                    await asyncio.wait_for(stop.wait(), timeout=delay)
            except TimeoutError:
                pass

    async def heartbeat_owner(owner_id, boot_generation):
        while not stop.is_set():
            try:
                await asyncio.wait_for(stop.wait(), timeout=max(1, settings.lease_seconds / 3))
            except TimeoutError:
                try:
                    await store.heartbeat_owner(owner_id, boot_generation)
                except Exception:
                    logging.exception("executor owner lease lost; stopping new dispatch")
                    stop.set()

    try:
        if command in ("executor", "outbox"):
            await wakeup.start()
        if command == "configure":
            for pool in config["pools"]:
                await store.configure_pool(
                    parse_pool(pool["admission"]), pool["group_ceiling"]
                )
            return
        if command == "executor":
            owner_id, boot_generation = "executor-" + str(uuid4()), str(uuid4())
            await store.register_owner(owner_id, boot_generation)
            tasks.append(asyncio.create_task(heartbeat_owner(owner_id, boot_generation)))
            blobs = artifacts(settings)
            await blobs.ready()
            speech = speech_profiles(config)
            embeddings = embedding_profiles(config)
            for pool in config["pools"]:
                if pool["admission"].get("kind") == "rerank":
                    continue
                spec = parse_pool(pool["admission"])
                driver = engine_driver(pool, speech, embeddings)
                drivers.append(driver)
                workers = min(settings.executor_count,
                    spec.background_transport_limit if spec.kind == "llm" else max(1, spec.target))
                for _ in range(workers):
                    worker = Executor(
                        store, blobs, driver, pool["admission"]["pool_id"], owner_id
                    )
                    tasks.append(asyncio.create_task(repeat(worker.tick, 5, on_events=True)))
        elif command == "reconciler":
            tasks.append(asyncio.create_task(repeat(store.reconcile_expired, 5)))
        else:
            client = await Client.connect(
                settings.temporal_address, namespace=settings.temporal_namespace
            )
            if command == "outbox":
                from .buffering import BufferedSubmissions

                publisher = OutboxPublisher(store, client, str(uuid4()))
                tasks.append(asyncio.create_task(repeat(publisher.tick, 5, on_events=True)))
                buffers = BufferedSubmissions(store, artifacts(settings),
                                              control_queue=settings.temporal_queue)
                tasks.append(asyncio.create_task(repeat(buffers.tick, 5, on_events=True)))
            elif command == "worker":
                activities = BrokerActivities(store, artifacts(settings))
                worker = Worker(
                    client,
                    task_queue=settings.temporal_queue,
                    activities=[activities.submit_or_attach, activities.submit_speech,
                                activities.submit_embedding, activities.finish],
                    max_concurrent_activities=32,
                    graceful_shutdown_timeout=timedelta(seconds=30),
                    deployment_config=WorkerDeploymentConfig(
                        version=WorkerDeploymentVersion(
                            "intramind-runtime", os.environ["RUNTIME_BUILD_ID"]
                        ),
                        use_worker_versioning=True,
                        default_versioning_behavior=VersioningBehavior.PINNED,
                    ),
                )
                async with worker:
                    await stop.wait()
                return
        await stop.wait()
        if command == "executor":
            # Stop fetching work, but retain live transports and any output bytes
            # until their current tick has persisted/settled or recorded UNKNOWN.
            logging.info("Executor stopping; waiting for active attempts to settle")
            await asyncio.gather(*tasks)
    finally:
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task
        for driver in drivers:
            await driver.close()
        await wakeup.close()
        await store.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=["api", "configure", "worker", "executor", "outbox", "reconciler"]
    )
    args = parser.parse_args()
    settings = Settings()
    logging.basicConfig(
        level=settings.log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    config = json.loads(Path(settings.pool_config).read_text())
    if args.command == "api":
        store = Store(settings.database_url.get_secret_value(), lease_seconds=settings.lease_seconds)
        sizing = preparers(config)
        app = create_app(
            store,
            artifacts(settings),
            settings.service_token.get_secret_value(),
            config["tasks"],
            settings.temporal_queue,
            sizing,
            manage_lifecycle=True,
            speech_preparers={key: SpeechPreparer(profile)
                              for key, profile in speech_profiles(config).items()},
            embedding_preparers={key: EmbeddingPreparer(profile)
                                 for key, profile in embedding_profiles(config).items()},
            artifact_max_bytes=settings.artifact_max_bytes,
            artifact_upload_concurrency=settings.artifact_upload_concurrency,
            direct_proxies=direct_proxies(config, store, sizing),
        )
        uvicorn.run(app, host="0.0.0.0", port=8070)
    else:
        asyncio.run(services(args.command, settings, config))


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from datetime import timedelta
import os
import time

import torch
import pathlib

from torch.distributed.elastic.multiprocessing.errors import record

from torchtitan.components.checkpoint import CheckpointManager, TrainState
from torchtitan.components.ft import FTParallelDims, init_ft_manager
from torchtitan.components.metrics import (
    build_metrics_processor,
    ensure_pp_loss_visible,
)
from torchtitan.config_manager import JobConfig
from torchtitan.datasets import build_hf_data_loader, build_tokenizer
from torchtitan.datasets.mixtera_datasets import build_mixtera_data_loader
from torchtitan.models.llama.model import PerDomainLoss

from torchtitan.distributed import ParallelDims, utils as dist_utils

from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.protocols.train_spec import get_train_spec

from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)


from mixtera.torch import MixteraTorchDataset
from mixtera.core.client import MixteraClient, QueryExecutionArgs, ResultStreamingArgs
from mixtera.core.query import Query
from mixtera.core.query.mixture import InferringMixture, StaticMixture, MixtureKey
from mixtera.core.query.mixture.dynamic_mixture import DynamicMixture
from mixtera.core.algo.ado.ado import AdoDynamicMixing
from mixtera.utils.feedback import handle_mixtera_feedback
from mixtera.utils.checkpoint import handle_mixtera_checkpoint

# Query execution in Mixtera takes long, and NCCL would time out otherwise.
os.environ["NCCL_TIMEOUT"] = str(30 * 60 * 1000)

# Enable debug tracing on failure: https://pytorch.org/docs/stable/elastic/errors.html
@record
def main(job_config: JobConfig):
    logger.info(f"Starting job: {job_config.job.description}")

    if job_config.experimental.custom_model_path:
        utils.import_module_from_path(job_config.experimental.custom_model_path)

    if job_config.job.print_args:
        logger.info(f"Running with args: {job_config.to_dict()}")

    # take control of garbage collection to avoid stragglers
    gc_handler = utils.GarbageCollection(gc_freq=job_config.training.gc_freq)

    device_module, device_type = utils.device_module, utils.device_type
    device = torch.device(f"{device_type}:{int(os.environ['LOCAL_RANK'])}")
    # Device has to be set before creating TorchFT manager.
    device_module.set_device(device)
    ft_manager = init_ft_manager(job_config)

    # init distributed
    world_size = int(os.environ["WORLD_SIZE"])
    if not ft_manager.enabled:
        parallel_dims = ParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
        )
    else:
        raise RuntimeError("TorchFT not support with Mixtera currently.")
        parallel_dims = FTParallelDims(
            dp_shard=job_config.training.data_parallel_shard_degree,
            dp_replicate=job_config.training.data_parallel_replicate_degree,
            cp=job_config.experimental.context_parallel_degree,
            tp=job_config.training.tensor_parallel_degree,
            pp=job_config.experimental.pipeline_parallel_degree,
            world_size=world_size,
            enable_loss_parallel=not job_config.training.disable_loss_parallel,
            ft_manager=ft_manager,
        )
    dist_utils.init_distributed(job_config)

    # build meshes
    world_mesh = parallel_dims.build_mesh(device_type=device_type)
    if parallel_dims.dp_enabled:
        dp_mesh = world_mesh["dp"]
        dp_degree, dp_rank = dp_mesh.size(), dp_mesh.get_local_rank()
        dp_group = dp_mesh.get_group()
    else:
        dp_degree, dp_rank = 1, 0
        dp_group = None

    if parallel_dims.pp_enabled:
        pp_mesh = world_mesh["pp"]

    if parallel_dims.tp_enabled:
        tp_mesh = world_mesh["tp"]
        tp_rank = tp_mesh.get_local_rank()
    else:
        tp_rank = 0

    # Set random seed, and maybe enable deterministic mode (mainly for debugging, expect perf loss)
    dist_utils.set_determinism(
        world_mesh, device, job_config.training.seed, job_config.training.deterministic
    )
    train_spec = get_train_spec(job_config.model.name)

    coordinate = world_mesh.get_coordinate()
    mesh_dim_names = world_mesh.mesh_dim_names
    mesh_sizes = world_mesh.mesh.shape
    dp_dim_names = ['dp_replicate', 'dp_shard'] # i dont think this is fully correct looking at device_mesh.py - this breaks down when ussing pp/cp/tp or sth, the names are not like that.

    # Map dimension names to indices and sizes
    coord_dict = dict(zip(mesh_dim_names, coordinate))
    size_dict = dict(zip(mesh_dim_names, mesh_sizes))

    dp_ranks = []
    dp_sizes = []
    for dp_dim_name in dp_dim_names:
        if dp_dim_name in mesh_dim_names:
            dp_ranks.append(coord_dict[dp_dim_name])
            dp_sizes.append(size_dict[dp_dim_name])
        else:
            dp_ranks.append(0)
            dp_sizes.append(1)

    # Compute dp_group_id
    dp_group_id = 0
    multiplier = 1
    for rank, size in zip(reversed(dp_ranks), reversed(dp_sizes)):
        dp_group_id += rank * multiplier
        multiplier *= size

    # Compute dp_degree
    dp_degree_mix = 1
    for size in dp_sizes:
        dp_degree_mix *= size # would 

    # Non-data-parallel dimensions
    non_dp_dims = [dim for dim in mesh_dim_names if dim not in dp_dim_names]

    node_id = 0
    multiplier = 1
    for dim_name in reversed(non_dp_dims):
        idx = coord_dict[dim_name]
        size = size_dict[dim_name]
        node_id += idx * multiplier
        multiplier *= size

    # Compute nodes_per_dp_group
    nodes_per_dp_group = 1
    for dim_name in non_dp_dims:
        nodes_per_dp_group *= size_dict[dim_name]

    logger.info(f"dp_group_id: {dp_group_id} nodes_per_dp_group: {nodes_per_dp_group} dp_degree: {dp_degree} dp_degree_mix: {dp_degree_mix} node_id: {node_id} ")

    assert nodes_per_dp_group == world_size // dp_degree
    assert nodes_per_dp_group == parallel_dims.non_data_parallel_size
    assert nodes_per_dp_group * dp_degree == world_size
    assert dp_degree_mix == dp_degree

    logger.info(f"dp_group_id: {dp_group_id}, dp_degree: {dp_degree}, node_id: {node_id}, nodes_per_dp_group: {nodes_per_dp_group}")
    
    dataloader_str = str(job_config.training.dataloader).lower()
    if dataloader_str == "mixtera":
        client = MixteraClient.from_remote(job_config.mixtera.ip, job_config.mixtera.port)
        job_id = job_config.mixtera.job_id
        chunk_size = job_config.mixtera.chunk_size
        tunnel_via_server = job_config.mixtera.tunnel_via_server
        chunk_reading_degree_of_parallelism = job_config.mixtera.chunk_reading_degree_of_parallelism
        num_workers = job_config.training.dl_worker
        tokenizer = job_config.training.tokenizer
        add_bos = job_config.training.add_bos
        add_eos = job_config.training.add_eos

        if add_bos:
            logger.info("Adding BOS. Are you sure you want that?")

        ## "Natural" baseline from ADO paper
        mixture_pile_static = StaticMixture(chunk_size=chunk_size, strict=False, mixture={
            MixtureKey({"pile_set_name": ["FreeLaw"]}): 0.04493927695030662,
            MixtureKey({"pile_set_name": ["Enron Emails"]}): 0.000998021865918546,
            MixtureKey({"pile_set_name": ["Github"]}): 0.12267758913758665,
            MixtureKey({"pile_set_name": ["OpenSubtitles"]}): 0.015835745965429738,
            MixtureKey({"pile_set_name": ["PubMed Central"]}): 0.12148621531516873,
            MixtureKey({"pile_set_name": ["OpenWebText2"]}): 0.10960682218906206,
            MixtureKey({"pile_set_name": ["StackExchange"]}): 0.049107965728456646,
            MixtureKey({"pile_set_name": ["Pile-CC"]}): 0.1824984780261193,
            MixtureKey({"pile_set_name": ["ArXiv"]}): 0.08862621733009907,
            MixtureKey({"pile_set_name": ["USPTO Backgrounds"]}): 0.02616577419097875,
            MixtureKey({"pile_set_name": ["Books3"]}): 0.10458626728299704,
            MixtureKey({"pile_set_name": ["Wikipedia (en)"]}): 0.04016661238580172,
            MixtureKey({"pile_set_name": ["PubMed Abstracts"]}): 0.02212837481440004,
            MixtureKey({"pile_set_name": ["NIH ExPorter"]}): 0.0018685647881937016,
            MixtureKey({"pile_set_name": ["BookCorpus2"]}): 0.006327357399975309,
            MixtureKey({"pile_set_name": ["EuroParl"]}): 0.008072738376112661,
            MixtureKey({"pile_set_name": ["HackerNews"]}): 0.004731183407655429,
            MixtureKey({"pile_set_name": ["DM Mathematics"]}): 0.019084626704901235,
            MixtureKey({"pile_set_name": ["YoutubeSubtitles"]}): 0.004027438721554198,
            MixtureKey({"pile_set_name": ["PhilPapers"]}): 0.0026731438901686708,
            MixtureKey({"pile_set_name": ["Ubuntu IRC"]}): 0.004850316881507234,
            MixtureKey({"pile_set_name": ["Gutenberg (PG-19)"]}): 0.0195412686476066,
        })

        ## Default pile weights
        mixture_pile_default = StaticMixture(chunk_size=chunk_size, strict=False, mixture={
            MixtureKey({"pile_set_name": ["Pile-CC"]}): 0.1121,
            MixtureKey({"pile_set_name": ["PubMed Central"]}): 0.1071,
            MixtureKey({"pile_set_name": ["Books3"]}): 0.0676,
            MixtureKey({"pile_set_name": ["OpenWebText2"]}): 0.1247,
            MixtureKey({"pile_set_name": ["ArXiv"]}): 0.1052,
            MixtureKey({"pile_set_name": ["Github"]}): 0.0427,
            MixtureKey({"pile_set_name": ["FreeLaw"]}): 0.0386,
            MixtureKey({"pile_set_name": ["StackExchange"]}): 0.0929,
            MixtureKey({"pile_set_name": ["USPTO Backgrounds"]}): 0.0420,
            MixtureKey({"pile_set_name": ["PubMed Abstracts"]}): 0.0845,
            MixtureKey({"pile_set_name": ["Gutenberg (PG-19)"]}): 0.0199,
            MixtureKey({"pile_set_name": ["OpenSubtitles"]}): 0.0124,
            MixtureKey({"pile_set_name": ["Wikipedia (en)"]}): 0.0919,
            MixtureKey({"pile_set_name": ["DM Mathematics"]}): 0.0198,
            MixtureKey({"pile_set_name": ["Ubuntu IRC"]}): 0.0074,
            MixtureKey({"pile_set_name": ["BookCorpus2"]}): 0.0044,
            MixtureKey({"pile_set_name": ["EuroParl"]}): 0.0043,
            MixtureKey({"pile_set_name": ["HackerNews"]}): 0.0075,
            MixtureKey({"pile_set_name": ["YoutubeSubtitles"]}): 0.0042,
            MixtureKey({"pile_set_name": ["PhilPapers"]}): 0.0027,
            MixtureKey({"pile_set_name": ["NIH ExPorter"]}): 0.0052,
            MixtureKey({"pile_set_name": ["Enron Emails"]}): 0.0030,
        })

        mixture_ado = DynamicMixture(strict=False, chunk_size=chunk_size, initial_mixture=mixture_pile_static, mixing_alg=AdoDynamicMixing(gamma2=0.1, count_normalizer=job_config.training.seq_len, use_same_step_size=True, delta_min=0.01, subsampling_interval=10, scaling_law_update_interval=1000, ignore_initial_steps=500, start_step=1000, logging_path=f"/capstor/store/cscs/swissai/a09/mixtera/adologs/{job_id}_seqfix.json", variant="vanilla"))
        
        # Set this to the mixture you want to use.
        if job_config.mixtera.pile == "ado":
            mixture = mixture_ado
            logger.info("Using ADO mixture")
        elif job_config.mixtera.pile == "default":
            mixture = mixture_pile_default
            logger.info("Using default mixture")
        elif job_config.mixtera.pile == "natural":
            mixture = mixture_pile_static
            logger.info("Using natural mixture")
        else:
            raise RuntimeError(f"Unknown pie mixture {job_config.mixtera.pile}")

        query_execution_args = QueryExecutionArgs(mixture=mixture, dp_groups=dp_degree, nodes_per_group=nodes_per_dp_group, num_workers=num_workers)
        # Please note that chunk_reading_mixture_type="token" is currently necessary in torchtitan, since we did not implement tokenization outside of Mixtera in torchtitan.
        # The torchtitan default is to apply BOS tokens. However, since this complicates evaluation (e.g., in the eval harness), we disable this here.
        streaming_args = ResultStreamingArgs(job_id=job_id, dp_group_id=dp_group_id, node_id=node_id, tunnel_via_server=tunnel_via_server, 
                                            chunk_reading_degree_of_parallelism=chunk_reading_degree_of_parallelism,
                                            chunk_reading_mixture_type="token", chunk_reading_tokenizer=tokenizer, chunk_reading_sequence_len=job_config.training.seq_len,
                                            chunk_reading_token_overlapping=False, chunk_reading_eos=add_eos, chunk_reading_bos=add_bos)
        
        query = Query.for_job(job_id).select(None) # TODO: Specify query in config file.

        return_key_id = isinstance(mixture, DynamicMixture) # not the best criterion to decide this on, but suffices for now.

        checkpoints_folder = os.path.join(job_config.job.dump_folder, job_config.checkpoint.folder)
        mix_step = job_config.checkpoint.load_step
        if mix_step == -1:
            mix_step = CheckpointManager._get_max_step(checkpoints_folder)
        step_folder = os.path.join(checkpoints_folder, f"step-{mix_step}")
        mixtera_id = os.path.join(step_folder, "mixtera.id")

        if os.path.isdir(step_folder) and not os.path.isfile(mixtera_id):
            logger.warning(f"Checkpoint directory {step_folder} exists but does not contain mixtera.id file - cannot load Mixtera checkpoint. If you load the model weights but not Mixtera, this may lead to unintended behavior. Please double check whether you are running on a checkpoint created using Mixtera.")

        should_load_checkpoint = job_config.checkpoint.enable_checkpoint and os.path.isdir(step_folder) and os.path.isfile(mixtera_id)
        checkpoint_mixtera_path = None if not should_load_checkpoint else pathlib.Path(step_folder)
        if should_load_checkpoint:
            logger.info(f"Loading Mixtera checkpoint from {step_folder}")
        else:
            logger.info("Will not load Mixtera checkpoint but run query from scratch.")

        raw_dataset = MixteraTorchDataset(client, query, query_execution_args, streaming_args, checkpoint_path=checkpoint_mixtera_path, return_key_id=return_key_id)

        # build dataloader
        data_loader = build_mixtera_data_loader(
            raw_dataset, job_config.training.batch_size, num_workers, return_key_id
        )
        vocab_size = job_config.mixtera.vocab_size
        if vocab_size < 1:
            raise RuntimeError(f"You did not provide mixtera.vocab_size!")
    elif dataloader_str in {"huggingface", "hf"}:
        tokenizer = build_tokenizer(job_config.training.tokenizer, job_config.model.tokenizer_path)
        # build dataloader
        streaming = not job_config.hf.disable_streaming
        data_loader = build_hf_data_loader(
            job_config.training.dataset,
            job_config.training.dataset_path,
            tokenizer,
            job_config.training.batch_size,
            job_config.training.seq_len,
            dp_degree,
            dp_rank,
            job_config.training.dl_worker,
            streaming,
            job_config.training.add_bos,
            job_config.training.add_eos
        )
        vocab_size = tokenizer.n_words
    else:
        raise RuntimeError(f"Unknown dataloader: {job_config.training.dataloader}")


    # build model (using meta init)
    model_cls = train_spec.cls
    model_config = train_spec.config[job_config.model.flavor]
    # set the model configs from training inputs:
    # 1. norm type to decide which norm layer to use
    # 2. vocab size from tokenizer
    # 3. max_seq_len base on inputs
    model_config.norm_type = job_config.model.norm_type
    model_config.vocab_size = vocab_size
    model_config.max_seq_len = job_config.training.seq_len

    logger.info(
        f"Building {train_spec.name} {job_config.model.flavor} with {model_config}"
    )
    with torch.device("meta"):
        model = model_cls.from_model_args(model_config)

    # Build the collection of model converters. No-op if `model.converters` empty
    model_converters = build_model_converters(job_config, parallel_dims)
    model_converters.convert(model)

    # metrics logging
    build_metrics_processor_fn = (
        build_metrics_processor
        if train_spec.build_metrics_processor_fn is None
        else train_spec.build_metrics_processor_fn
    )
    metrics_processor = build_metrics_processor_fn(job_config, parallel_dims)
    color = metrics_processor.color

    # log model size
    model_param_count = utils.get_num_params(model)
    metrics_processor.num_flop_per_token = utils.get_num_flop_per_token(
        utils.get_num_params(model, exclude_embedding=True),
        model_config,
        job_config.training.seq_len,
    )
    logger.info(
        f"{color.blue}Model {train_spec.name} {job_config.model.flavor} "
        f"{color.red}size: {model_param_count:,} total parameters{color.reset}"
    )

    # Note we ignore loss_fn from train spec because we hardcode the perdomainloss in this repo.
    per_domain_loss_module = PerDomainLoss(device=device) 

    # move sharded model to CPU/GPU and initialize weights via DTensor
    if job_config.checkpoint.create_seed_checkpoint:
        init_device = "cpu"
        buffer_device = None
    elif job_config.training.enable_cpu_offload:
        init_device = "cpu"
        buffer_device = device_type
    else:
        init_device = device_type
        buffer_device = None

    # apply parallelisms and initialization
    if parallel_dims.pp_enabled:
        raise NotImplementedError("do not support per domain loss with pp yet")
        # apply PT-D Pipeline Parallel
        (
            pp_schedule,
            model_parts,
            has_first_stage,
            has_last_stage,
        ) = train_spec.pipelining_fn(
            model,
            pp_mesh,
            parallel_dims,
            job_config,
            device,
            model_config,
            train_spec.loss_fn,
        )
        # when PP is enabled, `model` obj is no longer used after this point, model_parts is used instead
        del model

        # For PP with looped schedules, each item in model_parts is one stage-model-chunk.
        # We need to iterate through model_parts to apply SPMD parallelisms, compilation,
        # optimizer, and checkpointing
        for m in model_parts:
            # apply SPMD-style PT-D techniques
            train_spec.parallelize_fn(m, world_mesh, parallel_dims, job_config)
            m.to_empty(device=init_device)
            with torch.no_grad():
                m.init_weights(buffer_device=buffer_device)
            m.train()

        # confirm that user will be able to view loss metrics on the console
        ensure_pp_loss_visible(parallel_dims, job_config, color)

    else:
        # apply PT-D Tensor Parallel, activation checkpointing, torch.compile, Data Parallel
        train_spec.parallelize_fn(model, world_mesh, parallel_dims, job_config)
        model.to_empty(device=init_device)
        with torch.no_grad():
            model.init_weights(buffer_device=buffer_device)
        model.train()

        model_parts = [model]

    # initialize device memory monitor and get peak flops for MFU calculation
    device_memory_monitor = metrics_processor.device_memory_monitor
    gpu_peak_flops = utils.get_peak_flops(device_memory_monitor.device_name)
    logger.info(f"Peak FLOPS used for computing MFU: {gpu_peak_flops:.3e}")
    device_mem_stats = device_memory_monitor.get_peak_stats()
    logger.info(
        f"{device_type.upper()} memory usage for model: "
        f"{device_mem_stats.max_reserved_gib:.2f}GiB"
        f"({device_mem_stats.max_reserved_pct:.2f}%)"
    )

    # build optimizer after applying parallelisms to the model
    optimizers = train_spec.build_optimizers_fn(model_parts, job_config, ft_manager)
    lr_schedulers = train_spec.build_lr_schedulers_fn(optimizers, job_config)
    # Post optimizer step model converters hook.
    # e.g. calculate float8 dynamic amax/scale for all-parameter for FSDP2
    # where it issues a single all-reduce for all parameters at once for better performance
    optimizers.register_step_post_hook(
        lambda *args, **kwargs: model_converters.post_optimizer_hook(model_parts)
    )

    train_state = TrainState()

    # load initial checkpoint
    checkpoint = CheckpointManager(
        dataloader=data_loader,
        model_parts=model_parts,
        optimizers=optimizers,
        lr_schedulers=lr_schedulers,
        states={"train_state": train_state},
        job_config=job_config,
        ft_manager=ft_manager,
    )

    if job_config.checkpoint.create_seed_checkpoint:
        assert (
            world_size == 1
        ), "Must create seed checkpoint using a single device, to disable sharding"
        assert (
            job_config.checkpoint.enable_checkpoint
        ), "Must enable checkpointing when creating a seed checkpoint"
        checkpoint.save(curr_step=0, force=True)
        logger.info("Created seed checkpoint")
        return

    checkpoint.load(step=job_config.checkpoint.load_step)

    data_iterator = iter(data_loader)

    train_context = dist_utils.get_train_context(
        parallel_dims.loss_parallel_enabled,
        job_config.experimental.enable_compiled_autograd,
    )

    utils.global_barrier() # Global barrier necessary for mixtera to ensure query has been executed.

    # train loop
    logger.info(
        f"Training starts at step {train_state.step + 1}, "
        f"with local batch size {job_config.training.batch_size}, "
        f"global batch size {job_config.training.batch_size * dp_degree}, "
        f"sequence length {job_config.training.seq_len}, "
        f"total steps {job_config.training.steps} "
        f"(warmup {job_config.training.warmup_steps})"
    )
    with maybe_enable_profiling(
        job_config, global_step=train_state.step
    ) as torch_profiler, maybe_enable_memory_snapshot(
        job_config, global_step=train_state.step
    ) as memory_profiler:
        while train_state.step < job_config.training.steps:
            train_state.step += 1
            gc_handler.run(train_state.step)

            # get batch
            data_load_start = time.perf_counter()
            batch = next(data_iterator)

            if len(batch) == 3:
                input_ids, labels, key_ids = batch
            else:
                assert len(batch) == 2
                input_ids, labels = batch
                key_ids = None

            metrics_processor.ntokens_since_last_log += labels.numel()
            metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            input_ids = input_ids.to(device_type)
            labels = labels.to(device_type)
            key_ids = key_ids.to(device_type) if key_ids is not None else key_ids

            optimizers.zero_grad()

            handle_losses = None
            handle_counts = None
            losses_tensor = None
            counts_tensor = None
            init_async_start = 0
            init_async_time = 0

            # apply context parallelism if cp is enabled
            # ensure CP handles the separate freqs_cis buffer for each pp stage
            optional_context_parallel_ctx = (
                dist_utils.create_context_parallel_ctx(
                    cp_mesh=world_mesh["cp"],
                    cp_buffers=[input_ids, labels] + [m.freqs_cis for m in model_parts],
                    cp_seq_dims=[1, 1] + [0 for _ in model_parts],
                    cp_no_restore_buffers={input_ids, labels},
                    cp_rotate_method=job_config.experimental.context_parallel_rotate_method,
                )
                if parallel_dims.cp_enabled
                else None
            )

            if parallel_dims.pp_enabled:
                # todo mixtera support
                raise NotImplementedError("no mixtera support for pp yet")
                # Pipeline Parallel forward / backward inside step() call
                with train_context(optional_context_parallel_ctx):
                    targets, losses = (labels, []) if has_last_stage else (None, None)
                    if has_first_stage:
                        pp_schedule.step(input_ids, target=targets, losses=losses)
                    else:
                        pp_schedule.step(target=targets, losses=losses)

                # accumulate losses across pipeline microbatches
                # TODO: PP+FSDP unexpectedly puts the loss back to the CPU
                loss = (
                    torch.mean(torch.stack(losses)).to(device)
                    if has_last_stage
                    else torch.tensor([-1.0], device=device)
                )
            else:
                # Non-PP forward / backward
                with train_context(optional_context_parallel_ctx):
                    pred = model(input_ids)
                    loss = per_domain_loss_module(pred, labels, key_ids)
                    # pred.shape=(bs, seq_len, vocab_size)
                    # need to free to before bwd to avoid peaking memory
                    del pred

                    init_async_start = time.perf_counter()
                    if per_domain_loss_module.has_per_domain_loss:
                        with torch.no_grad():
                            losses_tensor, counts_tensor, max_id_tensor = per_domain_loss_module.get_per_domain_stats()
                            max_handle = torch.distributed.all_reduce(max_id_tensor, op=torch.distributed.ReduceOp.MAX, async_op=True, group=dp_group) # TODO: dp mesh vs dp group?
                            per_domain_loss_module.reset_per_domain_stats()
                            max_handle.wait()
                            max_domain_id = max_id_tensor.item()
                            # Resize tensors to the maximum domain ID
                            if losses_tensor.size(0) < max_domain_id + 1:
                                new_size = max_domain_id + 1 - losses_tensor.size(0)
                                losses_tensor = torch.cat(
                                    [losses_tensor, torch.zeros(new_size, dtype=losses_tensor.dtype, device=losses_tensor.device)], dim=0)
                                counts_tensor = torch.cat(
                                    [counts_tensor, torch.zeros(new_size, dtype=counts_tensor.dtype, device=counts_tensor.device)], dim=0)

                            handle_losses = torch.distributed.all_reduce(losses_tensor, op=torch.distributed.ReduceOp.SUM, async_op=True, group=dp_group)
                            handle_counts = torch.distributed.all_reduce(counts_tensor, op=torch.distributed.ReduceOp.SUM, async_op=True, group=dp_group)

                    init_async_time = time.perf_counter() - init_async_start

                    loss.backward()

            # clip gradients
            dist_utils.clip_grad_norm_(
                [p for m in model_parts for p in m.parameters()],
                job_config.training.max_norm,
                foreach=True,
                pp_mesh=pp_mesh if parallel_dims.pp_enabled else None,
            )

            # optimizer step
            checkpoint.maybe_wait_for_staging()
            optimizers.step()
            lr_schedulers.step()

            wait_mixtera_start = time.perf_counter()
            mixtera_feedback_time = 0
            wait_mixtera_time = 0
            if per_domain_loss_module.has_per_domain_loss:
                handle_losses.wait()
                handle_counts.wait()
                wait_mixtera_time = time.perf_counter() - wait_mixtera_start

                mixtera_feedback_start = time.perf_counter()

                handle_mixtera_feedback(
                    raw_dataset,
                    train_state.step,
                    losses_tensor,
                    counts_tensor,
                    dp_rank,
                    tp_rank,
                )

                mixtera_feedback_time = time.perf_counter() - mixtera_feedback_start
            # log metrics
            if metrics_processor.should_log(train_state.step):
                if (
                    parallel_dims.dp_replicate_enabled
                    or parallel_dims.dp_shard_enabled
                    or parallel_dims.cp_enabled
                ):
                    loss = loss.detach()
                    global_avg_loss, global_max_loss = (
                        dist_utils.dist_mean(loss, world_mesh["dp_cp"]),
                        dist_utils.dist_max(loss, world_mesh["dp_cp"]),
                    )
                else:
                    global_avg_loss = global_max_loss = loss.item()

                metrics_processor.log(
                    train_state.step, global_avg_loss, global_max_loss, init_async_time, wait_mixtera_time, mixtera_feedback_time
                )

            checkpoint_path = checkpoint.save(
                train_state.step, force=(train_state.step == job_config.training.steps)
            )
            if checkpoint_path is not None:
                handle_mixtera_checkpoint(data_loader, pathlib.Path(checkpoint_path), dp_rank, tp_rank, False)

            # signal the profiler that the next profiling step has started
            if torch_profiler:
                torch_profiler.step()
            if memory_profiler:
                memory_profiler.step()

            # reduce timeout after first train step for faster signal
            # (assuming lazy init and compilation are finished)
            if train_state.step == 1:
                dist_utils.set_pg_timeouts(
                    timeout=timedelta(seconds=job_config.comm.train_timeout_seconds),
                    world_mesh=world_mesh,
                )

    if torch.distributed.get_rank() == 0:
        logger.info("Sleeping 2 seconds for other ranks to complete")
        time.sleep(2)

    metrics_processor.close()
    logger.info("Training completed")


if __name__ == "__main__":
    init_logger()
    config = JobConfig()
    config.parse_args()
    main(config)
    torch.distributed.destroy_process_group()

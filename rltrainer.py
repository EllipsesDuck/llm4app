import os
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch

import torch
import torch.utils.data
import transformers
from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
from torch.utils.data import Sampler
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl import apply_chat_template, is_conversational, maybe_apply_chat_template
# from trl import is_vllm_available
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl import SyncRefModelCallback
from trl import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad, selective_log_softmax

import random

from transformers import (
        is_wandb_available, 
        AutoTokenizer, 
        AutoModelForCausalLM,
        TemperatureLogitsWarper, 
        LogitsProcessorList,
        Trainer
    )

from logitprocessor import ConstrainedLogitsProcessor
from transformers.generation import LogitsProcessor
import math

if is_peft_available():
    from peft import PeftConfig, get_peft_model

# if is_vllm_available():
    # from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class RepeatRandomSampler(Sampler):
    def __init__(self, data_source: Sized, repeat_count: int, seed: Optional[int] = None):
        self.data_source = data_source
        self.repeat_count = repeat_count
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()  # Create a local random generator
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        indexes = [
            idx
            for idx in torch.randperm(self.num_samples, generator=self.generator).tolist()
            for _ in range(self.repeat_count)
        ]
        return iter(indexes)

    def __len__(self):
        return self.num_samples * self.repeat_count

class ReReTrainer(Trainer):
    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        base_model: str,
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,

        #* sample
        add_gt: bool = False,
        dynamic_sampling: bool = False,
        beam_search: bool = False,
        length_penalty: float = 0.0,
        #* eval
        test_during_training: bool = True,
        test_beam: int = 20,

        #*loss
        dapo: bool = False,
        gspo: bool = False,

        #* others
        info_file: str = None,
        # logits_processor: Optional[LogitsProcessor] = None,
        prompt2history: dict[str, str] = None,
        history2target: dict[str, str] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        data_collator=None,
    ):
        # self.is_lazy_decoder = getattr(getattr(self.model, "config", None), "model_type", None) == "lazy_decoder"

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        self.base_model = base_model
        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        # if self.is_lazy_decoder:
        #     self.ref_model = None
        # else:
        #     if is_deepspeed_zero3_enabled():
        #         self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        #     elif not is_peft_model(model):
        #         self.ref_model = create_reference_model(model)
        #     else:
        #         self.ref_model = None

        # # Processing class
        # if not self.is_lazy_decoder:
        #     if processing_class is None:
        #         processing_class = AutoTokenizer.from_pretrained(self.base_model, padding_side="left")
        #         processing_class.pad_token = processing_class.eos_token
        # else:
        #     processing_class = None

        # self.processing_class = processing_class


        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(len(reward_funcs), dtype=torch.float32)

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        if data_collator is None:
            data_collator = lambda x: x

        self.data_collator = data_collator

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        print(f"max_completion_length: {self.max_completion_length}")
        self.num_generations = args.num_generations  # = G in the GRPO paper 
        self.use_vllm = args.use_vllm

        self.beta = args.beta
        

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)   
        self.log_completions = args.log_completions

        super().__init__(
            model=model,
            args=args,
            data_collator=self.data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # self.is_lazy_decoder = getattr(getattr(self.model, "config", None), "model_type", None) == "lazy_decoder"
        self.is_lazy_decoder = (
            hasattr(self.model, "config")
            and getattr(self.model.config, "model_type", None) == "lazy_decoder"
        )

        self.prompt2history = prompt2history
        self.history2target = history2target
        self.add_gt = add_gt
        self.beam_search = beam_search
        self.info_file = info_file
        self.temperature = args.temperature
        self.length_penalty = length_penalty
        self.test_during_training = test_during_training
        self.test_beam = test_beam
        self.dynamic_sampling = dynamic_sampling
        self.dapo = dapo
        self.gspo = gspo
        # self.logits_processor = logits_processor

        if self.is_lazy_decoder:
            self.ref_model = None
        else:
            if is_deepspeed_zero3_enabled():
                self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
            elif not is_peft_model(model):
                self.ref_model = create_reference_model(model)
            else:
                self.ref_model = None

        # Processing class
        if not self.is_lazy_decoder:
            if processing_class is None:
                processing_class = AutoTokenizer.from_pretrained(self.base_model, padding_side="left")
                processing_class.pad_token = processing_class.eos_token
        else:
            processing_class = None

        self.processing_class = processing_class


        # Check if the per_device_train/eval_batch_size * num processes can be divided by the number of generations
        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes
        possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        # Ensure each process receives a unique seed to prevent duplicate completions when generating with
        # transformers if num_generations exceeds per_device_train_batch_size. We could skip it if we use vLLM, but
        # it's safer to set it in all cases.
        set_seed(args.seed, device_specific=True)

        if self.is_lazy_decoder:
            self.generation_config = None
            self.test_generation_config = None
        else:
            if self.use_vllm:
                if not is_vllm_available():
                    raise ImportError(
                        "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                        "`pip install vllm` to use it."
                    )

                if self.accelerator.is_main_process:
                    vllm_device = self.args.vllm_device
                    if vllm_device == "auto":
                        if torch.cuda.device_count() == 1:
                            vllm_device = "cuda:0"  # particular case when training with onyl 1 GPU: share it
                        else:
                            vllm_device = f"cuda:{self.accelerator.num_processes}"  # take the next GPU idx
                    # Check that the requested device is available
                    if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                        raise ValueError(
                            f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                            "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                            "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                            f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                        )
                    # Check that the requested device is not also used for training
                    if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                        warnings.warn(
                            f"The requested device {vllm_device} is also being used for training. For higher throughput "
                            "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                            "If this is intentional, you may ignore this warning but should adjust "
                            "`vllm_gpu_memory_utilization` accordingly."
                        )
                    # vLLM is not compatible with accelerate. So we need to patch it to make sure we can (1) place the vLLM
                    # model on the desired device (world_size_patch) and (2) avoid a test that is not designed for our
                    # setting (profiling_patch).
                    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
                    profiling_patch = patch(
                        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                    )
                    with world_size_patch, profiling_patch:
                        self.llm = LLM(
                            model=model.name_or_path,
                            device=vllm_device,
                            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                            dtype=self.args.vllm_dtype,
                            # Automatic Prefix Caching caches the KV cache of existing queries, so that a new query can
                            # directly reuse the KV cache if it shares the same prefix with one of the existing queries.
                            # This is particularly useful here because we generate completions from the same prompts.
                            enable_prefix_caching=True,
                            max_model_len=self.args.vllm_max_model_len,
                        )
                    self.sampling_params = SamplingParams(
                        temperature=args.temperature,
                        max_tokens=self.max_completion_length,
                    )

                self._last_loaded_step = 0  # tag to avoid useless loading during grad accumulation

                # When using vLLM, the main process is responsible for loading the model weights. This can cause process
                # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
                # synchronize all processes after vLLM has been fully initialized.
                self.accelerator.wait_for_everyone()
            else:
                if self.beam_search:
                    #* temperature 默认为 1.0
                    print(f"self.temperature: {self.temperature}")
                    self.generation_config = GenerationConfig(
                        max_new_tokens=self.max_completion_length,
                        length_penalty=self.length_penalty,
                        num_beams=self.num_generations,
                        num_return_sequences=self.num_generations,
                        pad_token_id=processing_class.pad_token_id,
                        eos_token_id=processing_class.eos_token_id,
                        # temperature=self.temperature,
                        # do_sample=True, # if self.temperature > 1.0 else False,
                    )
                else:
                    self.generation_config = GenerationConfig(
                        max_new_tokens=self.max_completion_length,
                        length_penalty=self.length_penalty,
                        do_sample=True,
                        temperature=args.temperature,
                        pad_token_id=processing_class.pad_token_id,
                        eos_token_id=processing_class.eos_token_id,
                    )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        # Add tags to the model
        self.model.add_model_tags(self._tag_names)

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)

        if args.sync_ref_model:
            # print("Sync Begin")
            self.add_callback(SyncRefModelCallback(ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)


        if not self.is_lazy_decoder:
            if self.info_file is None:
                raise ValueError(
                    "info_file must be provided when using LLM mode because prefix constraints need it."
                )

            with open(self.info_file, 'r') as f:
                info = f.readlines()
                # Parse new format: semantic_id \t item_title \t item_id
                semantic_ids = [line.split('\t')[0].strip() + "\n" for line in info]
                item_titles = [line.split('\t')[1].strip() + "\n" for line in info if len(line.split('\t')) >= 2]

                info_semantic = [f'''### Response:\n{_}''' for _ in semantic_ids]
                info_titles = [f'''### Response:\n{_}''' for _ in item_titles]

                info = info_semantic

            tokenizer = AutoTokenizer.from_pretrained(self.base_model)

            if self.base_model.lower().find("llama") > -1:
                prefixID = [tokenizer(_).input_ids[1:] for _ in info]
            else:
                prefixID = [tokenizer(_).input_ids for _ in info]

            prefix_index = 4 if self.base_model.lower().find("gpt2") > -1 else 3

            self.hash_dict = dict()
            for index, ID in enumerate(prefixID):
                ID.append(tokenizer.eos_token_id)
                for i in range(prefix_index, len(ID)):
                    if i == prefix_index:
                        hash_number = self.get_hash(ID[:i])
                    else:
                        hash_number = self.get_hash(ID[prefix_index:i])

                    if hash_number not in self.hash_dict:
                        self.hash_dict[hash_number] = set()
                    self.hash_dict[hash_number].add(ID[i])

            for key in self.hash_dict:
                self.hash_dict[key] = list(self.hash_dict[key])

        else:
            self.hash_dict = {}
        if self.is_lazy_decoder:
            self.test_generation_config = None
        else:
            self.test_generation_config = GenerationConfig(max_new_tokens=self.max_completion_length,
                                                                length_penalty=self.length_penalty,
                                                                num_beams=self.test_beam,
                                                                num_return_sequences=self.test_beam,
                                                                do_sample=False,
                                                                pad_token_id=self.processing_class.pad_token_id,
                                                                eos_token_id=self.processing_class.eos_token_id,)

    def get_hash(self, x):
            x = [str(_) for _ in x]
            return '-'.join(x)

    def prefix_allowed_tokens_fn(self, batch_id, input_ids):
            hash_number = self.get_hash(input_ids)
            if hash_number in self.hash_dict:
                return self.hash_dict[hash_number]
            return []
    
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # def _get_train_sampler(self,  *args, **kwargs) -> Sampler:
    #     # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
    #     # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
    #     # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
    #     # preventing discrepancies in group formation.
    #     sampler = super()._get_train_sampler(*args, **kwargs)
    #     return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)
    
    def _get_train_sampler(self, train_dataset=None) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        if train_dataset is None:
            train_dataset = self.train_dataset
        return RepeatRandomSampler(self.train_dataset, self.num_generations, seed=self.args.seed)

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        # Returns a sampler that ensures each prompt is repeated across multiple processes. This guarantees that
        # identical prompts are distributed to different GPUs, allowing rewards to be computed and normalized correctly
        # within each prompt group. Using the same seed across processes ensures consistent prompt assignment,
        # preventing discrepancies in group formation.
        return RepeatRandomSampler(eval_dataset, self.num_generations, seed=self.args.seed)

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        logits = logits[:, -logits_to_keep:]
        return selective_log_softmax(logits, input_ids)  
    
    def _get_per_token_logps_lazy(
            self,
            prompt_ids,
            completion_ids,
            user_static,
            short_term,
            long_term,
            use_model,
        ):
        # 拼接 prompt + completion
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        out = use_model(
            input_ids=input_ids,
            user_static=user_static,
            short_term=short_term,
            long_term=long_term,
        )

        logits = out.logits                           # (B, T_prompt+T_comp, V)
        T_prompt = prompt_ids.size(1)
        logits_c = logits[:, T_prompt:, :]            # 只保留 completion 部分

        logps = torch.log_softmax(logits_c, dim=-1)   # (B, T_comp, V)
        per_token_logps = logps.gather(
            2, completion_ids.unsqueeze(-1)
        ).squeeze(-1)                                 # (B, T_comp)

        return per_token_logps



    def _move_model_to_vllm(self):
        with unwrap_model_for_generation(
            self.model, self.accelerator, gather_deepspeed3_params=self.args.ds3_gather_for_generation
        ) as unwrapped_model:
            if is_compiled_module(unwrapped_model):
                unwrapped_model = unwrapped_model._orig_mod
            if is_peft_model(unwrapped_model):
                unwrapped_model.merge_adapter()
                state_dict = unwrapped_model.state_dict()
                unwrapped_model.unmerge_adapter()
                # Remove base_model and base_layer prefixes
                state_dict = {
                    k.removeprefix("base_model.model.").replace(".base_layer", ""): v for k, v in state_dict.items()
                }
                # Remove values with adapter prefix (example: "_lora")
                state_dict = {k: v for k, v in state_dict.items() if unwrapped_model.prefix not in k}
                # When module to save, remove its prefix and discard the original module
                state_dict = {
                    k.replace("modules_to_save.default.", ""): v
                    for k, v in state_dict.items()
                    if "original_module" not in k
                }
            else:
                state_dict = unwrapped_model.state_dict()
        if self.accelerator.is_main_process:
            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(state_dict.items())

    def _prepare_inputs_llm(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]

        if self.add_gt or self.test_during_training or self.dynamic_sampling:
            histories = [self.prompt2history[x["prompt"]] for x in inputs]
            targets = [self.history2target[x] for x in histories]
            # print(f"targets: {targets}")
            num_categories = len(set(targets)) 
        # target_ids = self.processing_class(targets, return_tensors="pt", padding=True, padding_side="left")["input_ids"]
        # target_ids = target_ids.to(device)
        
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        
        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        ccc = ConstrainedLogitsProcessor(
                # guidance_scale=1.0,
                # cf_logits=None,
                prefix_allowed_tokens_fn=self.prefix_allowed_tokens_fn,
                # cf_dict=sasrec_dict,
                # unconditional_ids=None,
                num_beams=self.num_generations if self.beam_search else 1,
                base_model=self.base_model
            )
        self.logits_processor = LogitsProcessorList([TemperatureLogitsWarper(temperature=self.temperature), ccc])
        self.test_lp_list = LogitsProcessorList([ccc])

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                outputs = self.llm.generate(all_prompts_text, sampling_params=self.sampling_params, use_tqdm=False)
                completion_ids = [out.token_ids for completions in outputs for out in completions.outputs]
            else:
                completion_ids = [None] * len(all_prompts_text)
            # Broadcast the completions from the main process to all processes, ensuring each process receives its
            # corresponding slice.
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
            completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id)
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                topk = [3, 5, 10, 20]
                ndcg = [0 , 0, 0, 0]
                hr = [0, 0, 0, 0]

                if self.test_during_training:
                    dedup_prompt = []
                    dedup_mask = []
                    dedup_target = []

                    for i in range(len(prompt_ids)):
                        if i % self.num_generations == 0:
                            dedup_prompt.append(prompt_ids[i])
                            dedup_mask.append(prompt_mask[i])
                            dedup_target.append(targets[i])
                    
                    dedup_prompt_ids = torch.stack(dedup_prompt).to(device)
                    dedup_prompt_mask = torch.stack(dedup_mask).to(device)
                    # print(f"dedup_prompt_ids: {dedup_prompt_ids.shape}")
                
                    # print(f"test_beam: {self.test_beam}")
                    with torch.no_grad():
                        test_completion_ids = unwrapped_model.generate(
                            dedup_prompt_ids, attention_mask=dedup_prompt_mask, generation_config=self.test_generation_config,
                            logits_processor=self.test_lp_list,
                        )
                    
                    # print(f"test_completion_ids: {test_completion_ids.shape}")
                    if self.base_model.lower().find("llama")>-1:
                        test_completions = self.processing_class.batch_decode(test_completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    else:
                        test_completions = self.processing_class.batch_decode(test_completion_ids, skip_special_tokens=True)
                    test_completions = [_.split("Response:\n")[-1] for _ in test_completions]
                    test_comp_lis = [test_completions[i:i+self.test_beam] for i in range(0, len(test_completions), self.test_beam)]
                    for i, comp_lis in enumerate(test_comp_lis):
                        target = dedup_target[i]
                        for j in range(len(comp_lis)):
                            if comp_lis[j].strip("\n\"") == target.strip("\n\""):
                                for index, k in enumerate(topk):
                                    if j < k:
                                        hr[index] += 1
                                        ndcg[index] += 1 / math.log2(j+2) 
                                break
                    hr = [elm/len(dedup_target) for elm in hr]
                    ndcg = [elm/len(dedup_target) for elm in ndcg]

               

                if self.beam_search:
                    dedup_prompt = []
                    dedup_mask = []
                    for i in range(len(prompt_ids)):
                        if i % self.num_generations == 0:
                            dedup_prompt.append(prompt_ids[i])
                            dedup_mask.append(prompt_mask[i])
                    dedup_prompt_ids = torch.stack(dedup_prompt).to(device)
                    dedup_prompt_mask = torch.stack(dedup_mask).to(device)
                    # print(f"dedup_prompt_ids: {dedup_prompt_ids.shape}")
                    prompt_completion_ids = unwrapped_model.generate(
                        dedup_prompt_ids, attention_mask=dedup_prompt_mask, generation_config=self.generation_config,
                        logits_processor=self.logits_processor,
                    )
                    # print(f"prompt_ids: {prompt_ids.shape}")
                    # print(f"prompt_completion_ids: {prompt_completion_ids.shape}")
                else:
                    if self.dynamic_sampling:
                        lis1 = []
                        lis2 = []
                        extended_targets = []
                        for i in range(0, len(prompt_ids), self.num_generations):
                            lis1.extend([prompt_ids[i]]*int(1.5*self.num_generations))
                            lis2.extend([prompt_mask[i]]*int(1.5*self.num_generations))
                            extended_targets.extend([targets[i]]*int(1.5*self.num_generations))
                        extended_prompt_ids = torch.stack(lis1).to(device)
                        extended_prompt_mask = torch.stack(lis2).to(device)
                        # print(f"extended_prompt_ids: {extended_prompt_ids.shape}")
                        # print(f"extended_prompt_mask: {extended_prompt_mask.shape}")
                        prompt_completion_ids = unwrapped_model.generate(
                            extended_prompt_ids, attention_mask=extended_prompt_mask, generation_config=self.generation_config,
                            logits_processor=self.logits_processor,
                        )
                        prompt_length = prompt_ids.size(1)
                        extended_completion_ids = prompt_completion_ids[:, prompt_length:]
                        if self.base_model.lower().find("llama")>-1:
                            extended_completions_text = self.processing_class.batch_decode(extended_completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        else:
                            extended_completions_text = self.processing_class.batch_decode(extended_completion_ids, skip_special_tokens=True)
                        # print(f"extended_completions_text: {extended_completions_text}")

                        def select_completion(completions, target):
                            from collections import Counter
                            selected = []
                            completion_times = Counter(completions)
                            completion_times = dict(sorted(completion_times.items(), key=lambda x: x[1], reverse=True))
                            if target in completions:
                                selected.extend([target]*min(completion_times[target], self.num_generations))
                            if len(selected) == self.num_generations:
                                return selected
                            for item in completion_times:
                                if item != target and completion_times[item] > 0:
                                    selected.append(item)
                                    completion_times[item] -= 1
                                    if len(selected) == self.num_generations:
                                        return selected
                            while len(selected) < self.num_generations:
                                for item in completion_times:
                                    if item != target and completion_times[item] > 0:
                                        selected.append(item)
                                        completion_times[item] -= 1
                                        if len(selected) == self.num_generations:
                                            return selected
                        selected_completion = []
                        for i in range(0, len(extended_completions_text), int(self.num_generations*1.5)):
                            selected_completion.extend(select_completion(extended_completions_text[i:i+int(self.num_generations*1.5)], extended_targets[i]))
                        # print(f"selected_completion: {len(selected_completion)}")
                        selected_completion_ids = self.processing_class(selected_completion, return_tensors="pt", padding=True, padding_side="right", \
                            add_special_tokens=True)["input_ids"].to(device)
                        # print(f"selected_completion_ids: {selected_completion_ids.shape}")
                        prompt_completion_ids = torch.cat([prompt_ids, selected_completion_ids], dim=1)
                        # print(f"dynSam_prompt_completion_ids: {prompt_completion_ids.shape}")
                            
                    else:
                        prompt_completion_ids = unwrapped_model.generate(
                            prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config,
                            logits_processor=self.logits_processor,
                        )

            if self.add_gt:
                repeat = len(prompts) // num_categories
                new_prompt_completions = []
                flag = False
                # rep_ind = [random.randint(i, i+repeat-1) for i in range(0, len(prompts), repeat)]
                for i in range(len(prompts)):
                    if (i+1)%repeat == 0:
                        target_ids = self.processing_class(targets[i], return_tensors="pt", padding=True, padding_side="left", \
                            add_special_tokens=True)["input_ids"].squeeze()
                        # print(f"target_ids: {target_ids.shape}")
                        # print(f"prompt_ids: {prompt_ids[idx].shape}")
                        target_ids = target_ids.to(device)
                        added_ids = torch.cat([prompt_ids[i], target_ids], dim=0)
                        # print(f"added_ids: {added_ids.shape}")
                        new_prompt_completions.append(added_ids)
                    else:
                        new_prompt_completions.append(prompt_completion_ids[i])
                prompt_completion_ids = pad(new_prompt_completions, padding_value=self.processing_class.pad_token_id)
                
                    
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            
            completion_ids = prompt_completion_ids[:, prompt_length:]


        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        # completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # print(completions_text)
        
        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # Decode the generated completions
        if self.base_model.lower().find("llama")>-1:
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        else:
            completions_text = self.processing_class.batch_decode(completion_ids, skip_special_tokens=True)
        # print(completions_text)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append([{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        
        div_lis = [len(set(completions_text[i:i+self.num_generations]))/self.num_generations for i in range(0, len(completions_text), self.num_generations)]
        # cate_diversity = len(set(completions_text))/len(completions_text)
        cate_diversity = sum(div_lis)/len(div_lis)
        completion_ids_cpu = completion_ids.cpu().numpy()
        total_ids = set()
        num_tokens = 0
        for ids in completion_ids_cpu:
            ids = ids[ids != self.processing_class.pad_token_id]
            total_ids.update(set(ids))
            num_tokens += len(ids)        
        num_unique_tokens = len(total_ids)
        token_diversity = num_unique_tokens / num_tokens if num_tokens > 0 else 0.0

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c} for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)["text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[:, 0]  # Shape (B*G,)
            else:
                # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key] for example in inputs] for key in keys}
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        # print(f"advantages: {advantages}")

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]
        sliced_rewards = rewards[process_slice]



        # Log the metrics
        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())


        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics["categorical_diversity"].append(cate_diversity)
        self._metrics["token_diversity"].append(token_diversity)

        if self.test_during_training:
            for i in range(len(topk)):
                self._metrics[f"NDCG@{topk[i]}"].append(ndcg[i])
                self._metrics[f"HR@{topk[i]}"].append(hr[i])

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
            and "wandb" in self.args.report_to
        ):
            import pandas as pd

            # For logging
            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if wandb.run is not None and self.accelerator.is_main_process:
                wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
            "sliced_rewards": sliced_rewards,
        }
    
    def _prepare_inputs_lazy_decoder(self, inputs):

        device = self.accelerator.device

        input_ids = torch.stack([x["input_ids"] for x in inputs]).to(device)
        user_static = torch.stack([x["user_static"] for x in inputs]).to(device)
        short_term = torch.stack([x["short_term"] for x in inputs]).to(device)
        long_term = torch.stack([x["long_term"] for x in inputs]).to(device)
        rewards = torch.tensor([x["reward"] for x in inputs]).float().to(device)

        # with torch.no_grad():
        #     seqs = self.model.generate(
        #         input_ids=input_ids,
        #         user_static=user_static,
        #         short_term=short_term,
        #         long_term=long_term,
        #         max_new_tokens=self.max_completion_length,
        #         temperature=self.temperature,
        #     )

        # prompt_len = input_ids.size(1)
        # prompt_ids = seqs[:, :prompt_len]
        # completion_ids = seqs[:, prompt_len:]
        completion_ids = input_ids.clone()
        prompt_ids = input_ids.clone()

        # mask
        prompt_mask = torch.ones_like(prompt_ids)
        completion_mask = torch.ones_like(completion_ids)
        # completion_mask = (completion_ids != self.model.config.pad_token_id).int()
        # attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        # logits_to_keep = completion_ids.size(1)

        # if self.is_lazy_decoder:
        #     ref_logps = self._get_per_token_logps_lazy(
        #         prompt_ids,
        #         completion_ids,
        #         inputs.get("user_static"),
        #         inputs.get("short_term"),
        #         inputs.get("long_term"),
        #     )

        if self.ref_model is not None:
            ref_model = self.ref_model
        else:
            ref_model = self.model

        ref_logps = self._get_per_token_logps_lazy(
            prompt_ids,
            completion_ids,
            user_static,
            short_term,
            long_term,
            use_model=ref_model,
        )

        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-4)

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_logps,
            "advantages": advantages,
            "sliced_rewards": rewards,
            "user_static": user_static,
            "short_term": short_term,
            "long_term": long_term,
        }

    def _prepare_inputs(self, inputs):
        if self.is_lazy_decoder:
            return self._prepare_inputs_lazy_decoder(inputs)

        return self._prepare_inputs_llm(inputs)

    def _compute_loss_lazy(self, model, inputs, num_items_in_batch=None):
        # 这里 inputs 已经是 _prepare_inputs_lazy_decoder 返回的 dict
        prompt_ids = inputs["prompt_ids"]
        prompt_mask = inputs["prompt_mask"]
        completion_ids = inputs["completion_ids"]
        completion_mask = inputs["completion_mask"]
        user_static = inputs["user_static"]
        short_term = inputs["short_term"]
        long_term = inputs["long_term"]

        # 用当前 policy model 计算每个 token 的 logp
        per_token_logps = self._get_per_token_logps_lazy(
            prompt_ids,
            completion_ids,
            user_static,
            short_term,
            long_term,
            use_model=model,
        )

        ref_per_token_logps = inputs["ref_per_token_logps"]
        advantages = inputs["advantages"]

        # 和原 compute_loss 一样的 KL & loss 逻辑
        per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        if self.dapo:
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        elif self.gspo:
            per_token_ratio = per_token_logps - per_token_logps.detach()
            s_score = torch.exp((per_token_ratio * completion_mask).sum(dim=1) / completion_mask.sum(dim=1))
            sequence_kl = (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
            loss = -(s_score * advantages - self.beta * sequence_kl).mean()
        else:
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

        # 日志部分可以直接复用你原来的实现
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)

        mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        return loss


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        if self.is_lazy_decoder:
            return self._compute_loss_lazy(model, inputs, num_items_in_batch=num_items_in_batch)
        else:
            prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
            completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
            input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
            attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
            logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

            per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1


            advantages = inputs["advantages"]

            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
            per_token_loss = -(per_token_loss - self.beta * per_token_kl)

            if self.dapo:
                loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
            elif self.gspo:
                per_token_ratio = per_token_logps - per_token_logps.detach()
                s_score = torch.exp((per_token_ratio*completion_mask).sum(dim=1)/completion_mask.sum(dim=1)) 
                sequence_kl = (per_token_kl * completion_mask).sum(dim=1)/completion_mask.sum(dim=1)
                loss = -(s_score*advantages - self.beta*sequence_kl).mean()
            else:
                loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            # Log the metrics

            completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
            self._metrics["completion_length"].append(completion_length)
            
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())
                
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))




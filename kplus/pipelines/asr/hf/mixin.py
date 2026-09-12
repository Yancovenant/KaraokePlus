
import torch

class ASRMixin:
    def __init__(self, **kwargs):
        pass

    def get_inputs(self, audios: list | None = None, **kwargs):
        inputs = self.processor(
            text=hyp.latin,
            audio=audios,
            return_tensors="pt",
            sampling_rate=16000,
            padding=True,
        )
        inputs.to(self.model.device).to(self.model.dtype)
        return inputs

    @torch.no_grad()
    def _infer(self, inputs):
        return self.model(**inputs)

    @torch.inference_mode()
    def transcribe(self):
        pass

class QwenASR(ASRMixin):

    def get_inputs(self, audios = None, **kwargs):
        ProcessorKwargs = {
            "padding": True,
            "padding_side": "left",
            "sampling_rate": 16000,
            "truncation": False,
            "return_attention_mask": True,
            "n_window": 50,  # should match config.n_window
            "return_tensors": "pt"
        }
        ProcessorKwargs.update(**kwargs)
        kwargs = {k: v for k, v in kwargs if k not in ProcessorKwargs}
        inputs = self.processor.apply_transcription_request(
            audio=audios,                               # : AudioInput | list[AudioInput],
            language=None,                              # : str | list[str] | None = None,
            prompt=None,                                # : str | list[str] | None = None,
            processor_kwargs=ProcessorKwargs,
            **kwargs,
        )
        inputs.to(self.model.device).to(self.model.dtype)
        return inputs

    @torch.no_grad()
    def _infer(self, inputs, **kwargs):
        """ Language Model Qwen """
        return self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            generation_config=None,                     # : GenerationConfig | None = None,
            logits_processor=None,                      # : LogitsProcessorList | None = None,
            stopping_criteria=None,                     # : StoppingCriteriaList | None = None,
            prefix_allowed_tokens_fn=None,              # : Callable[[int, torch.Tensor], list[int]] | None = None,
            synced_gpus=None,                           # : bool | None = None,
            assistant_model=None,                       # : Optional["PreTrainedModel"] = None,
            streamer=None,                              # : Optional["BaseStreamer"] = None,
            negative_prompt_ids=None,                   # : torch.Tensor | None = None,
            negative_prompt_attention_mask=None,        # : torch.Tensor | None = None,
            custom_generate=None,                       # : str | Callable | None = None,
            **kwargs,

            # conversation: list[dict[str, str]] | list[list[dict[str, str]]],
            # chat_template: str | None = None,
            # tools: list[dict] | None = None,
            # documents: list[dict[str, str]] | None = None,
            # add_generation_prompt: bool = False,
            # continue_final_message: bool | str = False,
            # return_assistant_tokens_mask: bool = False,
            # tokenize: bool = False,
            # return_tensors: str | TensorType | None = None,
            # return_dict: bool = False,
            # load_audio_from_video: bool = False,
            # processor_kwargs: dict | None = None,
            # Other Kwargs:
            #   trust_remote_code: None,
            #   cache_implementation: "paged",
            #   input_ids: if inputs is None,
            #   num_beams: int = 1,
            #   max_length: int,
            #   min_length: int,
        )
    
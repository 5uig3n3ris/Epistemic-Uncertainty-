"""Model loading, generation, and hidden-state capture for R1-Distill models."""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .grading import extract_answer

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
PROMPT = "Answer the following question with a short answer (a few words).\nQuestion: {q}"


def load(model_id: str = DEFAULT_MODEL, load_in_8bit: bool = False):
    tok = AutoTokenizer.from_pretrained(model_id)
    kwargs = {"device_map": "auto"}
    if load_in_8bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    else:
        kwargs["torch_dtype"] = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs).eval()
    return tok, model


def build_inputs(tok, prompt: str, think: bool):
    """Chat-format the prompt. If think=False, prefill an empty <think></think> block."""
    text = tok.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True
    ).rstrip()
    if not text.endswith("<think>"):
        text += "<think>"
    text += "\n"
    if not think:
        text += "\n</think>\n\n"
    return tok(text, return_tensors="pt", add_special_tokens=False)


@torch.no_grad()
def generate_one(model, tok, question: str, think: bool = False, max_new_tokens: int = 256):
    """Greedy generation. Returns text, answer, per-token entropies, and the
    hidden state of the last prompt token at every layer, shape (L, d)."""
    inputs = build_inputs(tok, PROMPT.format(q=question), think).to(model.device)
    out = model(**inputs, output_hidden_states=True)
    # hidden_states[0] is the embedding output; [1:] are the transformer layers
    hidden = torch.stack([h[0, -1] for h in out.hidden_states[1:]]).float().cpu().numpy()

    gen = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        output_scores=True, return_dict_in_generate=True, pad_token_id=tok.eos_token_id,
    )
    new = gen.sequences[0, inputs["input_ids"].shape[1]:]
    ents = []
    for s in gen.scores:
        lp = torch.log_softmax(s[0].float(), -1)
        ents.append(float(-(lp.exp() * lp).sum()))
    text = tok.decode(new, skip_special_tokens=True)
    end_id = tok.convert_tokens_to_ids("</think>")
    ends = [i for i, t in enumerate(new.tolist()) if t == end_id]
    return {
        "text": text,
        "answer": extract_answer(text),
        "entropies": ents,
        "think_end_idx": ends[0] if ends else None,
        "hidden": hidden,
    }


@torch.no_grad()
def sample_answers(model, tok, question: str, k: int = 10, temperature: float = 1.0,
                   think: bool = False, max_new_tokens: int = 64):
    inputs = build_inputs(tok, PROMPT.format(q=question), think).to(model.device)
    gen = model.generate(
        **inputs, max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature,
        top_p=1.0, top_k=0, num_return_sequences=k, pad_token_id=tok.eos_token_id,
    )
    n = inputs["input_ids"].shape[1]
    return [extract_answer(tok.decode(s[n:], skip_special_tokens=True)) for s in gen]


@torch.no_grad()
def generate_text(model, tok, prompt: str, think: bool = False, max_new_tokens: int = 16):
    inputs = build_inputs(tok, prompt, think).to(model.device)
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    return extract_answer(tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))

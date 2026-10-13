# SheetSage2 Transformers compatibility issue

Status: open, checked 2026-10-13. Applies only to the experimental `sheetsage2`
extra. Tracking this issue locally keeps the experiment reviewable without
publishing an upstream issue or altering upstream code.

The pinned SheetSage2 revision
`b83c3d79a6d98bcffaf8f9ed112c90aa887887bc` constructs its BART decoder with
`BartDecoder(dc, embed_tokens=self.token_embedding)`. Transformers 5.19.0
rejects this with:

```
TypeError: BartDecoder.__init__() got an unexpected keyword argument 'embed_tokens'
```

Reproduced by loading the upstream config and constructing the model via
`AutoModel.from_config(..., trust_remote_code=True)` in an isolated environment.
Transformers 4.45.2 loads and transcribes real audio with NumPy 2.4.6; 4.57.6
constructs the model and performs cached decoder steps. The extra therefore
requires `transformers>=4.45.2,<5`; the legacy backend has no Transformers
requirement. This is an upper bound for a reproduced API incompatibility,
not a numerical equivalence claim for every version in the range.

Remove the upper bound once an upstream revision supports Transformers 5 and
passes `tools/verify_sheetsage2.py` on real music and silence, including raw
tokens/logits and ABC/MIDI/events. Revisit when updating the pinned model
revision or before promoting the backend out of experimental status.

Sources:
- https://huggingface.co/m-a-p/SheetSage2/blob/b83c3d79a6d98bcffaf8f9ed112c90aa887887bc/modeling_sheetsage2.py
- https://huggingface.co/m-a-p/SheetSage2/blob/b83c3d79a6d98bcffaf8f9ed112c90aa887887bc/requirements.txt

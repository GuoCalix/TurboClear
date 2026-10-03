# Example inputs

These 12 image/mask pairs were supplied from the ObjectClear example inputs.
They are a smoke test for inference, not a benchmark or the OBER training set.
Original project: https://github.com/jixin0101/ObjectClear

`imgs/` contains RGB images; `masks/` contains the corresponding object masks.
Match pairs by filename stem (extensions may differ). White selects the target
object; black preserves the surrounding region. TurboClear also predicts the
associated effects outside the object mask.

Example assets retain their original rights and are not relicensed by the
Apache-2.0 license on TurboClear source code.

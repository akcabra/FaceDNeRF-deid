# Vendored FaceNet model code

`inception_resnet_v1.py` and `utils/download.py` are pinned from
`timesler/facenet-pytorch` tag `v2.5.3` under the included MIT license. The
evaluation wrapper instantiates the architecture without pretrained downloads
and loads the repository-local VGGFace2 checkpoint explicitly.

Source: <https://github.com/timesler/facenet-pytorch/tree/v2.5.3>

Checkpoint:
`networks/20180402-114759-vggface2.pt`

Expected SHA-256:
`281cebca8662831adb987a874bdcb36e73f5b1c6dc5ee5878f305e985625d99b`

The recognizer is reserved for black-box evaluation and must not be introduced
into any optimization loss.

# ARCHITECTURE NOTES

## LIAE-ud (256x256 px, 256/64/64/22, mask layers not shown)

| **Encoder** | Act. | Output Shape | Params Formula | Params |
| ----------- | ---- | ------------ | -------------- | ------ |
| Input image | - | 3 x 256 x 256 | - | - |
| Conv 5 x 5 | LReLU | 64 x 128 x 128 | ((5\*5\*3)+1)*64 | 4.9k |
| Conv 5 x 5 | LReLU | 128 x 64 x 64 | ((5\*5\*64)+1)*128 | 205k |
| Conv 5 x 5 | LReLU | 256 x 32 x 32 | ((5\*5\*128)+1)*256 | 819k |
| Conv 5 x 5 | LReLU | 512 x 16 x 16 | ((5\*5\*256)+1)*512 | 3.3M |
| Flatten | - | 131,072 | - | - |

| **Inter** | Act. | Output Shape | Params Formula | Params |
| ----------- | ---- | ------------ | -------------- | ------ |
| Input vector | - | 131,072 | - | - |
| RMSNorm | - | 131,072 | - | - |
| Fully-connected | linear | 256 | (131,072+1)*256 | 33.6M |
| Fully-connected | linear | 32,768 | (256+1)*32,768 | 8.4M |
| Reshape | - | 512 x 8 x 8 | - | - |
| Conv 3 x 3 | LReLU | 2048 x 8 x 8 | ((3\*3\*512)+1)*2048 | 9.4M |
| Depth-to-space | - | 512 x 16 x 16 | - | - |

| **Decoder** | Act. | Output Shape | Params Formula | Params |
| ----------- | ---- | ------------ | -------------- | ------ |
| Input feature maps | - | 1024 x 16 x 16 | - | - |
| Conv 3 x 3 | LReLU | 2048 x 16 x 16 | ((3\*3\*1024)+1)*2048 | 170M |
| Depth-to-space | - | 512 x 32 x 32 | - | - |
| --- Conv 3 x 3 | LReLU | 512 x 32 x 32 | ((3\*3\*512)+1)*512 | 2.4M |
| --- Conv 3 x 3 | linear | 512 x 32 x 32 | ((3\*3\*512)+1)*512 | 2.4M |
| Residual | LReLU | 512 x 32 x 32 | - | - |
| Conv 3 x 3 | LReLU | 1024 x 32 x 32 | ((3\*3\*512)+1)*1024 | 4.7M |
| Depth-to-space | - | 256 x 64 x 64 | - | - |
| --- Conv 3 x 3 | LReLU | 256 x 64 x 64 | ((3\*3\*256)+1)*256 | 590k |
| --- Conv 3 x 3 | linear | 256 x 64 x 64 | ((3\*3\*256)+1)*256 | 590k |
| Residual | LReLU | 256 x 64 x 64 | - | - |
| Conv 3 x 3 | LReLU | 512 x 64 x 64 | ((3\*3\*256)+1)*512 | 1.2M |
| Depth-to-space | - | 128 x 128 x 128 | - | - |
| --- Conv 3 x 3 | LReLU | 128 x 128 x 128 | ((3\*3\*128)+1)*128 | 147k |
| --- Conv 3 x 3 | linear | 128 x 128 x 128 | ((3\*3\*128)+1)*128 | 147k |
| Residual | LReLU | 128 x 128 x 128 | - | - |
| (4x) Conv 3 x 3 | sigmoid | 3 x 128 x 128 | 4*((3\*3\*128)+1)*3 | 13.8k |
| Tile | - | 3 x 256 x 256 | - | - |

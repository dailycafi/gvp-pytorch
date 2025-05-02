# GVP: Geometric Vector Perceptron

## 原项目

https://github.com/drorlab/gvp-pytorch

## 安装指南

GVP (Geometric Vector Perceptron) 是一个处理 3D 分子结构的深度学习框架。本指南提供详细的安装步骤以避免常见问题。
因为过去的项目比较老， 所以对安装文件， 依赖等做了一些修改

### 环境要求

- Python 3.9+ (或者 3.10/3.11)
- 支持的平台: macOS (Intel/Apple Silicon)、Linux、Windows

### 方法一：通过 pip 安装（推荐）

```bash
# 1. 创建虚拟环境
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# 或 .venv\Scripts\activate  # Windows

# 2. 安装依赖
pip install -U pip  # 更新 pip
pip install .       # 安装项目及其依赖
```

### 方法二：通过 conda 安装

```bash
# 1. 创建环境
conda create -n gvp python=3.11 -y
conda activate gvp

# 2. 安装 PyTorch 2.1 及其依赖
conda install pytorch=2.1 -c pytorch -c conda-forge

# 3. 安装 PyG 依赖 (PyTorch Geometric 相关包)
conda install torch-scatter=2.1.1 torch-cluster=1.6.3 \
              torch-sparse=0.6.18 torch-spline-conv=1.2.2 \
              -c conda-forge

# 4. 安装本项目及其余依赖
pip install .
```

### 方法三：CUDA 12.4 用户安装指南（推荐）

对于使用 CUDA 12.4 的用户，请使用以下脚本进行安装：

```bash
#!/bin/bash

# 创建虚拟环境（如果使用 venv）
python -m venv .venv_gvp
source .venv_gvp/bin/activate

# 更新 pip
pip install --upgrade pip

# 安装 PyTorch 2.1.2 (CUDA 12.1 版本，兼容 CUDA 12.4)
pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121

# 安装基础依赖（指定版本以避免冲突）
pip install numpy==1.24.3 scipy==1.10.1 scikit-learn==1.3.2
pip install packaging pillow pyparsing pytz ninja

# 安装 PyG 相关依赖（使用预编译轮子）
pip install torch-scatter==2.1.1 torch-sparse==0.6.18 torch-cluster==1.6.3 torch-spline-conv==1.2.2 \
    -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# 安装 PyG 主包
pip install torch-geometric==2.5.1

# 安装 atom3d
pip install atom3d>=0.2.6

# 安装项目（开发模式）
pip install -e .

echo "GVP 安装完成！"
```

将上述内容保存为 `install_cuda124.sh`，然后执行：

```bash
chmod +x install_cuda124.sh
./install_cuda124.sh
```

### 特殊说明：Apple Silicon Mac (M1/M2/M3)

为避免在 Apple Silicon 上遇到编译问题，我们已经将所有依赖锁定到有预编译轮子的版本 (PyTorch 2.1.x 系列)。安装时不会触发复杂的 C++ 编译过程。

如果单独安装组件，请使用：

```bash
# Apple Silicon Mac 上安装 PyTorch CPU 版本
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# 安装 PyG 相关轮子 (确保使用 CPU 版本的 wheel)
pip install torch_scatter torch_cluster torch_sparse torch_spline_conv \
           -f https://data.pyg.org/whl/torch-2.1.0+cpu.html

# 安装主包
pip install torch_geometric==2.5.1
```

### 验证安装

安装完成后，可通过以下命令验证：

```bash
python -c "import torch, torch_scatter, torch_cluster, torch_geometric; \
           print(f'PyTorch: {torch.__version__}, PyG: {torch_geometric.__version__}')"
```

对于 CUDA 用户，可以使用以下脚本验证 CUDA 支持：

```python
import torch
import torch_geometric
import torch_scatter
import torch_cluster
import torch_sparse
import torch_spline_conv
import numpy
import scipy
import sklearn
import gvp

print("\n===== GVP 安装验证 =====\n")
print(f"PyTorch 版本: {torch.__version__}")
print(f"PyG 版本: {torch_geometric.__version__}")
print(f"NumPy 版本: {numpy.__version__}")
print(f"SciPy 版本: {scipy.__version__}")
print(f"scikit-learn 版本: {sklearn.__version__}")
print(f"CUDA 是否可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA 版本: {torch.version.cuda}")
    print(f"当前 CUDA 设备: {torch.cuda.get_device_name(0)}")
    
    # 测试 CUDA 功能
    x = torch.rand(5, 5).cuda()
    print(f"\nCUDA 张量测试: {x.device}")
```

将上述内容保存为 `verify_install.py` 并运行 `python verify_install.py`。

### 常见问题与排错

1. **NumPy 版本警告**：如果看到 NumPy ABI 相关警告，无需担心，我们已将依赖锁定到 `numpy<2.0`

2. **找不到预编译轮子**：确保使用的是 PyTorch 2.1.x 版本，该版本有完整的 arm64/x86_64 的 CPU 轮子

3. **Conda 安装时 PyG 包冲突**：尝试先只安装 PyTorch，然后用 pip 安装其余组件

4. **CUDA 版本兼容性**：PyTorch 官方预编译版本目前最高支持 CUDA 12.1，但通常与 CUDA 12.4 兼容

### 项目关键依赖

- torch ==2.1.2
- torch_geometric ==2.5.1
- torch_scatter ==2.1.1
- torch_cluster ==1.6.3 
- torch_sparse ==0.6.18
- torch_spline_conv ==1.2.2
- numpy ==1.24.3
- scipy ==1.10.1
- scikit-learn ==1.3.2
- atom3d >=0.2.6

### 原项目文档

Implementation of equivariant GVP-GNNs as described in [Learning from Protein Structure with Geometric Vector Perceptrons](https://openreview.net/forum?id=1YLJDvSx6J4) by B Jing, S Eismann, P Suriana, RJL Townshend, and RO Dror.

**UPDATE:** Also includes equivariant GNNs with vector gating as described in [Equivariant Graph Neural Networks for 3D Macromolecular Structure](https://arxiv.org/abs/2106.03843) by B Jing, S Eismann, P Soni, and RO Dror.

Scripts for training / testing / sampling on protein design and training / testing on all [ATOM3D](https://arxiv.org/abs/2012.04035) tasks are provided.

**Note:** This implementation is in PyTorch Geometric. The original TensorFlow code, which is not maintained, can be found [here](https://github.com/drorlab/gvp).

## General usage

We provide classes in three modules:
* `gvp`: core GVP modules and GVP-GNN layers
* `gvp.data`: data pipelines for both general use and protein design
* `gvp.models`: implementations of MQA and CPD models
* `gvp.atom3d`: models and data pipelines for ATOM3D

The core modules in `gvp` are meant to be as general as possible, but you will likely have to modify `gvp.data` and `gvp.models` for your specific application, with the existing classes serving as examples.

**Installation:** Download this repository and run `python setup.py develop` or `pip install . -e`. Be sure to manually install `torch_geometric` first!

**Tuple representation:** All inputs and outputs with both scalar and vector channels are represented as a tuple of two tensors `(s, V)`. Similarly, all dimensions should be specified as tuples `(n_scalar, n_vector)` where `n_scalar` and `n_vector` are the number of scalar and vector features, respectively. All `V` tensors must be shaped as `[..., n_vector, 3]`, not `[..., 3, n_vector]`.

**Batching:** We adopt the `torch_geometric` convention of absorbing the batch dimension into the node dimension and keeping track of batch index in a separate tensor.

**Amino acids:** Models view sequences as int tensors and are agnostic to aa-to-int mappings. Such mappings are specified as the `letter_to_num` attribute of `gvp.data.ProteinGraphDataset`. Currently, only the 20 standard amino acids are supported.

For all classes, see the docstrings for more detailed usage. If you have any questions, please contact bjing@cs.stanford.edu.

### Core GVP classes

The class `gvp.GVP` implements a Geometric Vector Perceptron.
```
import gvp

in_dims = scalars_in, vectors_in
out_dims = scalars_out, vectors_out
gvp_ = gvp.GVP(in_dims, out_dims)
```
To use vector gating, pass in `vector_gate=True` and the appropriate activations.
```
gvp_ = gvp.GVP(in_dims, out_dims,
            activations=(F.relu, None), vector_gate=True)
```
The classes `gvp.Dropout` and `gvp.LayerNorm` implement vector-channel dropout and layer norm, while using normal dropout and layer norm for scalar channels. Both expect inputs and return outputs of form `(s, V)`, but will also behave like their scalar-valued counterparts if passed a single tensor.
```
dropout = gvp.Dropout(drop_rate=0.1)
layer_norm = gvp.LayerNorm(dims)
```

The class `gvp.GVPConvLayer` implements a GVP-GNN layer, which includes a GVP for messages, a GVP for nodes, and residual connections. The class `gvp.GVPConv` implements the message passing step.
```
conv_layer = gvp.GVPConvLayer(node_dims, edge_dims, drop_rate=0.1)
```

### Protein design

The class `gvp.models.CPDModel` implements a conditional protein design model, which takes in a protein structure and predicts a sequence.
```
cpd_model = gvp.models.CPDModel(node_in_dim, node_h_dim, 
                        edge_in_dim, edge_h_dim)
out = cpd_model(nodes, batch.edge_index, 
                 edges, batch.seq) # shape (n_nodes, 20)
```

## Protein design
We provide a script `run_cpd.py` to train, validate, and test a `CPDModel` as specified in the paper using the CATH 4.2 dataset and TS50 dataset. If you want to use a trained model on new structures, see the section "Sampling" below.

### Fetching data
Run `getCATH.sh` in `data/` to fetch the CATH 4.2 dataset. If you are interested in testing on the TS 50 test set, also run `grep -Fv -f ts50remove.txt chain_set.jsonl > chain_set_ts50.jsonl` to produce a training set without overlap with the TS 50 test set. 

### Training / testing
To train a model, simply run `python run_cpd.py --train`. To test a trained model on both the CATH 4.2 test set and the TS50 test set, run `python run_cpd --test-r PATH` for perplexity or with `--test-p` for perplexity. Run `python run_cpd.py -h` for more detailed options.

```
$ python run_cpd.py -h

usage: run_cpd.py [-h] [--models-dir PATH] [--num-workers N] [--max-nodes N] [--epochs N] [--cath-data PATH] [--cath-splits PATH] [--ts50 PATH] [--train] [--test-r PATH] [--test-p PATH] [--n-samples N]

optional arguments:
  -h, --help          show this help message and exit
  --models-dir PATH   directory to save trained models, default=./models/
  --num-workers N     number of threads for loading data, default=4
  --max-nodes N       max number of nodes per batch, default=3000
  --epochs N          training epochs, default=100
  --cath-data PATH    location of CATH dataset, default=./data/chain_set.jsonl
  --cath-splits PATH  location of CATH split file, default=./data/chain_set_splits.json
  --ts50 PATH         location of TS50 dataset, default=./data/ts50.json
  --train             train a model
  --test-r PATH       evaluate a trained model on recovery (without training)
  --test-p PATH       evaluate a trained model on perplexity (without training)
  --n-samples N       number of sequences to sample (if testing recovery), default=100
```
**Confusion matrices:** Note that the values are normalized such that each row (corresponding to true class) sums to 1000, with the actual number of residues in that class printed under the "Count" column.

### Sampling
To sample from a `CPDModel`, prepare a `ProteinGraphDataset`, but do NOT pass into a `DataLoader`. The sequences are not used, so placeholders can be used for the `seq` attributes of the original structures dicts.

```
protein = dataset[i]
nodes = (protein.node_s, protein.node_v)
edges = (protein.edge_s, protein.edge_v)
    
sample = model.sample(nodes, protein.edge_index,  # shape = (n_samples, n_nodes)
                      edges, n_samples=n_samples)
```        
The output will be an int tensor, with mappings corresponding to those used when training the model.

## ATOM3D
We provide models and dataloaders for all ATOM3D tasks in `gvp.atom3d`, as well as a training and testing script in `run_atom3d.py`. This also supports loading pretrained weights for transfer learning experiments.

### Models / data loaders
The GVP-GNNs for ATOM3D are supplied in `gvp.atom3d` and are named after each task: `gvp.atom3d.MSPModel`, `gvp.atom3d.PPIModel`, etc. All of these extend the base class `gvp.atom3d.BaseModel`. These classes take no arguments at initialization, take in a `torch_geometric.data.Batch` representation of a batch of structures, and return an output corresponding to the task. Details vary based on the exact task---see the docstrings.
```
psr_model = gvp.atom3d.PSRModel()
```
`gvp.atom3d` also includes data loaders to produce `torch_geometric.data.Batch` objects from an underlying `atom3d.datasets.LMDBDataset`.  In the case of all tasks except PPI and RES, these are in the form of callable transform objects---`gvp.atom3d.SMPTransform`, `gvp.atom3d.RSRTransform`, etc---which should be passed into the constructor of a `atom3d.datasets.LMDBDataset`:
```
psr_dataset = atom3d.datasets.LMDBDataset(path_to_dataset,
                    transform=gvp.atom3d.PSRTransform())
```
On the other hand, `gvp.atom3d.PPIDataset` and `gvp.atom3d.RESDataset` take the place of / are wrappers around the `atom3d.datasets.LMDBDataset`:
```
ppi_dataset = gvp.atom3d.PPIDataset(path_to_dataset)
res_dataset = gvp.atom3d.RESDataset(path_to_dataset, path_to_split) # see docstring
```
All datasets must be then wrapped in a `torch_geometric.data.DataLoader`:
```
psr_dataloader = torch_geometric.data.DataLoader(psr_dataset, batch_size=batch_size)
```
The dataloaders can be directly iterated over to yield `torch_geometric.data.Batch` objects, which can then be passed into the models.
```
for batch in psr_dataloader:
    pred = psr_model(batch) # pred.shape = (batch_size,)
```

### Training / testing

To run training / testing on ATOM3D, download the datasets as described [here](https://www.atom3d.ai/). Modify the function `get_datasets` in `run_atom3d.py` with the paths to the datasets. Then run:
```
$ python run_atom3d.py -h

usage: run_atom3d.py [-h] [--num-workers N] [--smp-idx IDX]
                     [--lba-split SPLIT] [--batch SIZE] [--train-time MINUTES]
                     [--val-time MINUTES] [--epochs N] [--test PATH]
                     [--lr RATE] [--load PATH]
                     TASK

positional arguments:
  TASK                  {PSR, RSR, PPI, RES, MSP, SMP, LBA, LEP}

optional arguments:
  -h, --help            show this help message and exit
  --num-workers N       number of threads for loading data, default=4
  --smp-idx IDX         label index for SMP, in range 0-19
  --lba-split SPLIT     identity cutoff for LBA, 30 (default) or 60
  --batch SIZE          batch size, default=8
  --train-time MINUTES  maximum time between evaluations on valset,
                        default=120 minutes
  --val-time MINUTES    maximum time per evaluation on valset, default=20
                        minutes
  --epochs N            training epochs, default=50
  --test PATH           evaluate a trained model
  --lr RATE             learning rate
  --load PATH           initialize first 2 GNN layers with pretrained weights
```
For example:
```
# train a model
python run_atom3d.py PSR

# train a model with pretrained weights
python run_atom3d.py PSR --load PATH

# evaluate a model
python run_atom3d.py PSR --test PATH
```

## Acknowledgements
Portions of the input data pipeline were adapted from [Ingraham, et al, NeurIPS 2019](https://github.com/jingraham/neurips19-graph-protein-design). We thank Pratham Soni for portions of the implementation in PyTorch.

## Citation
```
@inproceedings{
    jing2021learning,
    title={Learning from Protein Structure with Geometric Vector Perceptrons},
    author={Bowen Jing and Stephan Eismann and Patricia Suriana and Raphael John Lamarre Townshend and Ron Dror},
    booktitle={International Conference on Learning Representations},
    year={2021},
    url={https://openreview.net/forum?id=1YLJDvSx6J4}
}

@article{jing2021equivariant,
  title={Equivariant Graph Neural Networks for 3D Macromolecular Structure},
  author={Jing, Bowen and Eismann, Stephan and Soni, Pratham N and Dror, Ron O},
  journal={arXiv preprint arXiv:2106.03843},
  year={2021}
}
```

# Backbones

To ensure a comprehensive benchmark across different architectural paradigms, we evaluate multiple distinct backbone configurations, encompassing CNNs, MLPs, and Vision Transformers.

## 2D Backbones

For our 2D medical imaging tasks (such as Chest X-Ray), we utilize the [`timm` (PyTorch Image Models)](https://huggingface.co/timm) library to load our feature extractors. We evaluate the following architectures:

* **ResNet-50 (Scratch)**
  * **Description:** Standard ResNet-50 architecture initialized from scratch (random weights).
  * **Load Code:**
    ```python
    import timm
    model = timm.create_model("resnet50", pretrained=False, num_classes=0)
    ```

* **ResNet-50 (Pretrained)**
  * **Description:** Standard ResNet-50 architecture initialized with ImageNet pre-trained weights.
  * **Load Code:**
    ```python
    import timm
    model = timm.create_model("resnet50", pretrained=True, num_classes=0)
    ```
    
* **ConvNeXt**
  * **Description:** A modernized standard ConvNet architecture (Tiny variant) initialized with ImageNet-22k pre-trained weights.
  * **Load Code:**
    ```python
    import timm
    model = timm.create_model("convnext_tiny.fb_in22k", pretrained=True, num_classes=0)
    ```

* **DeiT (Data-efficient Image Transformers)**
  * **Description:** A small, distilled Vision Transformer trained on ImageNet-1k. 
  * **Load Code:**
    ```python
    import timm
    model = timm.create_model("deit_small_distilled_patch16_224.fb_in1k", pretrained=True, num_classes=0)
    ```

* **ResMLP**
  * **Description:** A multi-layer perceptron (MLP) based architecture for image classification, initialized with ImageNet-1k weights.
  * **Load Code:**
    ```python
    import timm
    model = timm.create_model("resmlp_12_224.fb_in1k", pretrained=True, num_classes=0)
    ```

## 3D Backbones

For our 3D medical imaging tasks involving volumetric data (such as Brain MRI and Chest CT), we utilize 3D CNNs. The implementations for these architectures are sourced and adapted from the [`xmuyzz/3D-CNN-PyTorch`](https://github.com/xmuyzz/3D-CNN-PyTorch) repository.

For our current evaluations, we utilize the **3D ResNet-50** architecture from scratch. The evaluation of other 3D architectures provided by the repository could be open future work.

**Specialized Architecture for AD2A Algorithm:**
In particular, when evaluating the **AD2A** algorithm ([Paper Link](https://www.sciencedirect.com/science/article/pii/S1361841521001225)), we employ a specific 3D backbone based on 3D ResNet. The implementation for this specialized backbone is maintained separately and can be found in the `resnet_3d_for_ad2a.py` file.

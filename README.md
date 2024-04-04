# PointNet

## 1. Introduction

In this project, I re-implemented `PointNet` and tested its classification and segmentation performance. Please refer to my [report](report.pdf) to see full details, i.e., technical discussions, how to run the code, qualitative and quantitative results.

The models are trained and hosted on `Colab`:

- _Classification_ Network _with_ Feature Transformation: [Code](https://drive.google.com/drive/folders/1xUxI9VPA4Vp2u_Sh8BU5ZCxtuJgMXIZQ?usp=sharing), [Notebook](https://colab.research.google.com/drive/1FjOhSOztuWnt8hHVKg5FechXbFR2pbsd?usp=sharing)

- _Classification_ Network _without_ Feature Transformation: [Code](https://drive.google.com/drive/folders/1t625svLNEIpeDWmaXq5D6SAVNC70QF74?usp=sharing), [Notebook](https://colab.research.google.com/drive/1aHgOg6aq_M8HRdfYTWR_AjCBAZQfwjLe?usp=sharing)

- _Segmentation_ Network for Chairs _with_ Feature Transformation: [Code](https://drive.google.com/drive/folders/1xiGz62jRmxjQDR4xEzxo5llezZiqbD5R?usp=sharing), [Notebook](https://colab.research.google.com/drive/1kwGVVqXNd2aXZcTrrDLoZu2sysOpRLCP?usp=sharing)

- _Segmentation_ Network for Chairs _without_ Feature Transformation: [Code](https://drive.google.com/drive/folders/1_7i0SzVHMoZz5XcHP_owx1HEFSsL-K6v?usp=sharing), [Notebook](https://colab.research.google.com/drive/1-YrVgRn7t8DQgLZMieGFTz5yc5l9Prrn?usp=sharing)

- _Segmentation_ Network for Airplanes _with_ Feature Transformation: [Code](https://drive.google.com/drive/folders/1z00X8bY8b9JLUFimjE4gVWBm_GU00jzp?usp=sharing), [Notebook](https://colab.research.google.com/drive/1WpB4lwu8FhCUEVGqazXvTbZgG3FnN2Jn?usp=sharing)

- _Segmentation_ Network for Airplanes _without_ Feature Transformation: [Code](https://drive.google.com/drive/folders/1YmJwO0HiscNH3bqEwQY2NczjF9adqTrH?usp=sharing), [Notebook](https://colab.research.google.com/drive/1lqwNBY_ZfGeEQYNwWVRxiv-675k0L98C?usp=sharing)

**Topics:** _Computer Vision_, _PointNet_, _3D Classification and Segmentation_

**Skills:** _Pytorch_, _Python_, _Deep Neural Networks_, _Jupyter Lab_, _Colab_

## 2. Demo

Here are some sample results of `PointNet`'s 3D segmentation for airplanes and chairs:

Airplanes:

![Airplane](/demo/seg_airplane.png)

Chairs:

![Chair](/demo/seg_chair.png)
